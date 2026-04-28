"""
bm25_retriever.py
-----------------
BM25 keyword-based retrieval for the RAG pipeline.

Provides sparse (lexical) search to complement FAISS dense (semantic) search.
When combined via Reciprocal Rank Fusion in retriever.py, the system can find
chunks that match exact keyword phrases (e.g. "Pushdown automata") even when
the dense embedding doesn't rank them highly.

The BM25 index is built from the same chunks used for FAISS and persisted as
a pickled tokenized corpus alongside the FAISS index files.
"""

import os
import sys
import pickle
import re
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rank_bm25 import BM25Okapi
import config

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """
    Simple whitespace + lowercase tokenizer.

    Strips punctuation from token edges and filters out empty tokens.
    """
    return [
        tok
        for tok in re.sub(r"[^\w\s]", " ", text.lower()).split()
        if tok
    ]


def build_bm25_index(chunks: list[dict]) -> tuple["BM25Okapi", list[list[str]]]:
    """
    Tokenize chunk texts and build a BM25Okapi index.

    Args:
        chunks: List of chunk dicts (must have a "text" key).

    Returns:
        (bm25_index, tokenized_corpus)
        bm25_index       — the BM25Okapi object ready for queries
        tokenized_corpus — list of token lists (needed for reconstruction)
    """
    tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
    bm25_index = BM25Okapi(tokenized_corpus)
    return bm25_index, tokenized_corpus


def save_bm25(tokenized_corpus: list[list[str]], path: str) -> None:
    """Save the tokenized corpus to a pickle file."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(tokenized_corpus, f)


def load_bm25(path: str) -> tuple["BM25Okapi", list[list[str]]]:
    """
    Load tokenized corpus from pickle and rebuild the BM25 index.

    Args:
        path: Path to bm25_corpus.pkl.

    Returns:
        (bm25_index, tokenized_corpus)
    """
    with open(path, "rb") as f:
        tokenized_corpus = pickle.load(f)
    bm25_index = BM25Okapi(tokenized_corpus)
    return bm25_index, tokenized_corpus


def search_bm25(
    index: "BM25Okapi",
    tokenized_corpus: list[list[str]],
    query: str,
    top_k: int = 20,
) -> list[tuple[int, float]]:
    """
    Tokenize the query, score all documents, return top_k (index, score) pairs.

    Args:
        index:             BM25Okapi index object.
        tokenized_corpus:  The tokenized corpus used to build the index.
        query:             Raw query string.
        top_k:             Number of results to return.

    Returns:
        List of (corpus_index, bm25_score) tuples, sorted by descending score.
    """
    tokenized_query = _tokenize(query)
    scores = index.get_scores(tokenized_query)

    # Get top_k indices sorted by descending score
    top_indices = scores.argsort()[::-1][:top_k]

    return [
        (int(idx), float(scores[idx]))
        for idx in top_indices
        if scores[idx] > 0  # skip zero-score documents
    ]


class BM25Retriever:
    def __init__(self, notebook_id: str | None):
        chunks_path = (
            config.get_processed_dir(notebook_id) / "chunks.json"
            if notebook_id
            else config.CHUNKS_PATH
        )
        if not chunks_path.exists():
            logger.warning("BM25 warning: chunks.json not found at %s", chunks_path)
            self.chunks = []
            self.bm25 = None
            return

        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data.get("chunks", data) if isinstance(data, dict) else data

        corpus = [chunk["text"].lower().split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Return top_k chunks ranked by BM25 score."""
        if not self.bm25:
            return []

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)

        scored = [(score, chunk) for score, chunk in zip(scores, self.chunks)]
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, chunk in scored[:top_k]:
            c = dict(chunk)
            c["bm25_score"] = float(score)
            results.append(c)
        return results
