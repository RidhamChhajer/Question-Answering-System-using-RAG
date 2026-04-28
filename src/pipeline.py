"""
pipeline.py
-----------
Orchestrates the full document processing pipeline.

Phase 1: Load PDFs → Extract text → Clean text → Chunk → Save JSON
Phase 2: Load Chunks → Embed (batched) → Normalize → FAISS Index → Save

Phase 1 improvements applied:
  - Auto-creates `processed/` directory (no crash if it doesn't exist)
  - Empty pages are skipped (handled in document_loader)
  - Global sequential chunk IDs across all documents
  - token_count stored per chunk

Phase 2 improvements applied:
  - Embeddings are L2-normalized before being added to FAISS
  - token_count preserved in metadata.json
"""

import json
import os
import sys
import logging
from pathlib import Path

# Allow running from any working directory by adding project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_loader import load_documents
from src.text_cleaner import clean_text
from src.chunker import chunk_text, reset_counter
import config
# NOTE: embedding_model and vector_store are imported lazily inside
# run_embedding_pipeline() so that Phase 1 and --query start instantly
# without loading PyTorch / sentence-transformers.

logger = logging.getLogger(__name__)


def run_pipeline(
    input_folder: str | None = None,
    output_path: str | None = None,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
    notebook_id: str | None = None,
) -> list[dict]:
    """
    Run the complete document processing pipeline.

    Args:
        input_folder: Path to folder containing PDF files.
        output_path:  Full path for the output chunks.json file.
        chunk_size:   Target words per chunk.
        overlap:      Words of overlap between consecutive chunks.

    Returns:
        List of all chunk dicts written to output_path.
    """
    if notebook_id:
        input_folder = str(config.get_pdf_dir(notebook_id))
        output_path = str(config.get_chunks_path(notebook_id))

    if not input_folder or not output_path:
        raise ValueError("input_folder and output_path are required when notebook_id is not provided.")

    # Ensure the output directory exists (Improvement #1)
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reset chunk ID counter at the start of each pipeline run
    reset_counter()

    all_chunks: list[dict] = []

    # Step 1: Load all PDFs (empty pages already skipped inside loader)
    pages = load_documents(input_folder)

    if not pages:
        logger.warning("No content extracted. Nothing to process.")
        return []

    logger.info("Extracted %s non-empty page(s) from all documents.", len(pages))

    # Steps 2–4: Clean, chunk, and collect
    for page_data in pages:
        cleaned = clean_text(page_data["raw_text"])

        # After cleaning, the page might be empty (e.g., only had page numbers)
        if not cleaned:
            continue

        chunks = chunk_text(
            clean_text=cleaned,
            document_name=page_data["document"],
            page_number=page_data["page"],
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(chunks)

    # Step 5: Save output JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    return all_chunks


def run_embedding_pipeline(
    chunks_path: str | None = None,
    index_path: str | None = None,
    metadata_path: str | None = None,
    batch_size: int = config.BATCH_SIZE,
    notebook_id: str | None = None,
) -> int:
    """
    Phase 2 pipeline: embed all chunks and store in FAISS.

    Args:
        chunks_path:   Path to processed/chunks.json (output of Phase 1).
        index_path:    Destination for faiss_index.bin.
        metadata_path: Destination for metadata.json.
        batch_size:    Chunks per embedding batch (default 32).

    Returns:
        Number of vectors added to the index.
    """
    if notebook_id:
        chunks_path = str(config.get_chunks_path(notebook_id))
        index_path = str(config.get_index_path(notebook_id))
        metadata_path = str(config.get_metadata_path(notebook_id))

    if not chunks_path or not index_path or not metadata_path:
        raise ValueError("chunks_path, index_path, and metadata_path are required when notebook_id is not provided.")

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Load chunks ───────────────────────────────────────────────────
    if not os.path.isfile(chunks_path):
        raise FileNotFoundError(
            f"chunks.json not found at: {chunks_path}\n"
            "Run Phase 1 first (python main.py --phase 1)."
        )

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not chunks:
        logger.warning("chunks.json is empty. Nothing to embed.")
        return 0

    logger.info("Loaded %s chunks", len(chunks))

    # ── Step 2: Generate embeddings ───────────────────────────────────────────
    # Lazy imports — only loaded when Phase 2 actually runs.
    # This keeps Phase 1 and --query startup instant (no PyTorch load).
    from src.embedding_model import embed_batch, EMBEDDING_DIM
    from src.vector_store import create_index, train_index, add_embeddings, save_index, save_metadata

    logger.info("Generating embeddings (batch size %s)", batch_size)
    texts = [c["text"] for c in chunks]
    vectors = embed_batch(texts, batch_size=batch_size)  # shape (N, 384), normalized
    logger.info("Embeddings shape: %s", vectors.shape)


    # ── Step 3: Build FAISS index ─────────────────────────────────────────────
    n_vectors = len(vectors)
    logger.info("Creating FAISS index (n_vectors=%s)", n_vectors)
    index = create_index(EMBEDDING_DIM, n_vectors)
    train_index(index, vectors)   # no-op for IndexFlatL2; trains IndexIVFFlat
    add_embeddings(index, vectors)
    logger.info("Added %s vectors (dim=%s)", index.ntotal, EMBEDDING_DIM)

    # ── Step 4: Persist ───────────────────────────────────────────────────────
    save_index(index, index_path)
    save_metadata(chunks, metadata_path)

    logger.info("Saved index -> %s", index_path)
    logger.info("Saved metadata -> %s", metadata_path)

    # ── Step 5: Build and save BM25 corpus ────────────────────────────────────
    from src.bm25_retriever import build_bm25_index, save_bm25

    _, tokenized_corpus = build_bm25_index(chunks)
    bm25_path = os.path.join(os.path.dirname(index_path), "bm25_corpus.pkl")
    save_bm25(tokenized_corpus, bm25_path)
    logger.info("Saved BM25 -> %s", bm25_path)

    return index.ntotal
