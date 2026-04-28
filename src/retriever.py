"""
retriever.py
------------
Semantic retrieval engine for the RAG pipeline.

Pipeline:
    User query string
        ↓
    embed_text(query)  →  normalized 384-dim vector
        ↓
    faiss_index.search(vector, k)  →  (distances, indices)
        ↓
    Convert L2 distance → similarity score  (1 / 1 + distance)
        ↓
    Map indices → metadata entries
        ↓
    Return ranked list of top-k chunks
"""

import json
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.embedding_model import embed_text
from src.vector_store import load_index, load_metadata
import config  # read ENABLE_HYBRID_SEARCH / RRF_K at call-time, not import-time

logger = logging.getLogger(__name__)

# ── Module-level state (loaded once) ─────────────────────────────────────────
_index = None
_metadata: list[dict] = []
_loaded_index_path: str | None = None
_loaded_metadata_path: str | None = None
_bm25_index = None
_bm25_corpus: list[list[str]] = []
_loaded_bm25_path: str | None = None   # tracked so re-index forces a reload


def load_retriever(
    index_path: str | None = None,
    metadata_path: str | None = None,
    notebook_id: str | None = None,
) -> None:
    """
    Load the FAISS index and metadata into module-level state.

    Must be called once before retrieve().

    Args:
        index_path:    Path to vector_store/faiss_index.bin
        metadata_path: Path to processed/chunks.json
        notebook_id:   Notebook id for per-notebook index and chunks
    """
    global _index, _metadata, _loaded_index_path, _loaded_metadata_path
    global _bm25_index, _bm25_corpus, _loaded_bm25_path

    if notebook_id:
        index_path = str(config.get_index_path(notebook_id))
        metadata_path = str(config.get_chunks_path(notebook_id))

    if not index_path or not metadata_path:
        raise ValueError("index_path and metadata_path are required when notebook_id is not provided.")

    bm25_path = os.path.join(os.path.dirname(index_path), "bm25_corpus.pkl")

    # Skip reload only when FAISS *and* BM25 paths are all unchanged
    if (_index is not None
            and index_path    == _loaded_index_path
            and metadata_path == _loaded_metadata_path
            and bm25_path     == _loaded_bm25_path):
        return

    if not os.path.isfile(index_path):
        raise FileNotFoundError(
            f"FAISS index not found: {index_path}\n"
            "Run 'python main.py --phase 2' to generate it."
        )
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            "Run 'python main.py --phase 1' to generate it."
        )

    _index = load_index(index_path)
    if os.path.basename(metadata_path) == "chunks.json":
        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            _metadata = data.get("chunks", data)
        else:
            _metadata = data
    else:
        _metadata = load_metadata(metadata_path)
    _loaded_index_path = index_path
    _loaded_metadata_path = metadata_path

    logger.info("Index loaded: %s vectors (dim=%s)", _index.ntotal, _index.d)
    logger.info("Metadata loaded: %s entries", len(_metadata))

    # Attempt to load BM25 corpus from the same directory
    _bm25_index = None
    _bm25_corpus = []
    _loaded_bm25_path = bm25_path   # record path regardless of success
    if os.path.isfile(bm25_path):
        try:
            from src.bm25_retriever import load_bm25
            _bm25_index, _bm25_corpus = load_bm25(bm25_path)
            logger.info("BM25 loaded: %s documents", len(_bm25_corpus))
        except Exception as e:
            logger.warning("BM25 load failed (falling back to FAISS-only): %s", e)
            _bm25_index = None
            _bm25_corpus = []
    else:
        logger.warning("BM25 corpus not found (FAISS-only mode)")


def reciprocal_rank_fusion(faiss_results: list, bm25_results: list, k: int = 60) -> list:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    RRF score = sum(1 / (k + rank)) across all lists.
    """
    scores = {}
    # Assign RRF scores from FAISS ranking
    for rank, chunk in enumerate(faiss_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
    # Add RRF scores from BM25 ranking
    for rank, chunk in enumerate(bm25_results, start=1):
        cid = chunk["chunk_id"]
        scores[cid] = scores.get(cid, 0) + 1 / (k + rank)

    # Build merged result list (use chunk data from whichever list has it)
    all_chunks = {c["chunk_id"]: c for c in faiss_results + bm25_results}
    merged = sorted(all_chunks.values(), key=lambda c: scores[c["chunk_id"]], reverse=True)
    for chunk in merged:
        chunk["rrf_score"] = scores[chunk["chunk_id"]]
    return merged


def faiss_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed the query and return the top-k most relevant chunks (FAISS-only).
    """
    global _index, _metadata

    if _index is None:
        raise RuntimeError("Retriever not loaded. Call load_retriever() first.")

    # ── Step 1: Embed and normalize query ────────────────────────────────────────
    query_vector = embed_text(query)                      # shape (384,), normalized
    query_vector = np.array([query_vector], dtype=np.float32)  # shape (1, 384)

    # ── Step 2: FAISS search ──────────────────────────────────────────────────────
    k = min(top_k, _index.ntotal)           # don't ask for more than we have
    distances, indices = _index.search(query_vector, k)  # both shape (1, k)

    distances = distances[0]   # flatten to (k,)
    indices   = indices[0]     # flatten to (k,)

    # ── Step 3: Build FAISS results ───────────────────────────────────────────────
    faiss_results: list[dict] = []
    for dist, idx in zip(distances, indices):
        if idx == -1:          # FAISS returns -1 when fewer results than k exist
            continue

        # Convert L2 distance → similarity score in [0, 1]
        # Larger score = more similar
        score = float(1.0 / (1.0 + dist))

        chunk = _metadata[idx]
        faiss_results.append(
            {
                "score":       round(score, 4),
                "chunk_id":    chunk["chunk_id"],
                "document":    chunk["document"],
                "page":        chunk["page"],
                "token_count": chunk.get("token_count", 0),
                "text":        chunk["text"],
            }
        )

    return faiss_results


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """
    Embed the query and return the top-k most relevant chunks.

    Args:
        query:  Natural language question from the user.
        top_k:  Number of results to return (default 5).

    Returns:
        List of result dicts, ordered by descending similarity:
        [
          {
            "score":    0.82,         # similarity (higher = better)
            "chunk_id": "chunk_000001",
            "document": "ml_book.pdf",
            "page":     3,
            "token_count": 587,
            "text":     "Machine learning algorithms..."
          },
          ...
        ]
    """
    global _index, _metadata

    if _index is None:
        raise RuntimeError("Retriever not loaded. Call load_retriever() first.")

    # ── Step 1: Embed and normalize query ────────────────────────────────────────
    query_vector = embed_text(query)                      # shape (384,), normalized
    query_vector = np.array([query_vector], dtype=np.float32)  # shape (1, 384)

    # ── Step 2: FAISS search ──────────────────────────────────────────────────────
    k = min(top_k, _index.ntotal)           # don't ask for more than we have
    distances, indices = _index.search(query_vector, k)  # both shape (1, k)

    distances = distances[0]   # flatten to (k,)
    indices   = indices[0]     # flatten to (k,)

    # ── Step 3: Build FAISS results ───────────────────────────────────────────────
    faiss_results: list[dict] = []
    for dist, idx in zip(distances, indices):
        if idx == -1:          # FAISS returns -1 when fewer results than k exist
            continue

        # Convert L2 distance → similarity score in [0, 1]
        # Larger score = more similar
        score = float(1.0 / (1.0 + dist))

        chunk = _metadata[idx]
        faiss_results.append(
            {
                "score":       round(score, 4),
                "chunk_id":    chunk["chunk_id"],
                "document":    chunk["document"],
                "page":        chunk["page"],
                "token_count": chunk.get("token_count", 0),
                "text":        chunk["text"],
            }
        )

    # ── Step 4: Hybrid search with BM25 + RRF (if enabled) ───────────────────
    # Read flag at call-time so toggling config.ENABLE_HYBRID_SEARCH at runtime works.
    if not config.ENABLE_HYBRID_SEARCH or _bm25_index is None:
        # Results come pre-sorted by FAISS (ascending distance = descending similarity)
        return faiss_results

    # Run BM25 search
    from src.bm25_retriever import search_bm25
    bm25_hits = search_bm25(
        index=_bm25_index,
        tokenized_corpus=_bm25_corpus,
        query=query,
        top_k=top_k,
    )

    # Build ranked lists for RRF
    # faiss_ranking: chunk_id → rank (1-based)
    faiss_ranking: dict[str, int] = {}
    for rank, r in enumerate(faiss_results, start=1):
        faiss_ranking[r["chunk_id"]] = rank

    # bm25_ranking: chunk_id → rank (1-based)
    bm25_ranking: dict[str, int] = {}
    for rank, (corpus_idx, _score) in enumerate(bm25_hits, start=1):
        if corpus_idx < len(_metadata):
            cid = _metadata[corpus_idx]["chunk_id"]
            bm25_ranking[cid] = rank

    # Collect all chunk_ids from both systems
    all_chunk_ids = set(faiss_ranking.keys()) | set(bm25_ranking.keys())

    # Compute RRF score for each chunk
    # Read RRF_K at call-time so changes to config.RRF_K take effect without restart.
    rrf_k = config.RRF_K
    rrf_scores: dict[str, float] = {}
    for cid in all_chunk_ids:
        score = 0.0
        if cid in faiss_ranking:
            score += 1.0 / (rrf_k + faiss_ranking[cid])
        if cid in bm25_ranking:
            score += 1.0 / (rrf_k + bm25_ranking[cid])
        rrf_scores[cid] = score

    # Build a lookup for chunk data (from FAISS results + metadata)
    chunk_data: dict[str, dict] = {}
    for r in faiss_results:
        chunk_data[r["chunk_id"]] = r
    # Add BM25-only hits from metadata
    for corpus_idx, _ in bm25_hits:
        if corpus_idx < len(_metadata):
            m = _metadata[corpus_idx]
            cid = m["chunk_id"]
            if cid not in chunk_data:
                chunk_data[cid] = {
                    "score":       0.0,
                    "chunk_id":    cid,
                    "document":    m["document"],
                    "page":        m["page"],
                    "token_count": m.get("token_count", 0),
                    "text":        m["text"],
                }

    # Sort by RRF score descending, replace "score" with RRF score
    sorted_ids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)
    results: list[dict] = []
    for cid in sorted_ids[:top_k]:
        entry = chunk_data[cid].copy()
        entry["score"] = round(rrf_scores[cid], 4)
        results.append(entry)

    return results
