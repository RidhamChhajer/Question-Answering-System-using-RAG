"""
reranker.py
-----------
Cross-encoder re-ranking using BAAI/bge-reranker-base.

After FAISS retrieves a pool of candidates (e.g. top-20), this module
scores each (question, chunk) pair using a cross-encoder and returns
only the top-N most relevant chunks.

Why re-ranking?
  FAISS uses bi-encoder embeddings — fast, but approximate. A cross-encoder
  directly compares the question and each chunk together, giving much more
  accurate relevance scores at the cost of being slower.

  Running the cross-encoder over 20 chunks typically takes ~1–2 seconds on
  CPU, which is an acceptable overhead for significantly better precision.

Model: BAAI/bge-reranker-base (~550 MB, auto-downloaded on first use)
  - Strong re-ranking quality
  - CPU-compatible
  - No GPU required
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CrossEncoder is part of sentence-transformers (already installed)
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

# ── Module-level state (loaded once, reused across calls) ─────────────────────
_reranker: CrossEncoder | None = None
DEFAULT_MODEL = "BAAI/bge-reranker-base"


def load_reranker(model_name: str = DEFAULT_MODEL) -> None:
    """
    Load the CrossEncoder model into module-level state.

    Downloads the model on first call (~550 MB). Subsequent calls are instant
    because sentence-transformers caches models locally.

    Args:
        model_name: HuggingFace model ID (default: BAAI/bge-reranker-base).
    """
    global _reranker
    if _reranker is not None:
        return  # already loaded

    logger.info("Loading re-ranker: %s", model_name)
    _reranker = CrossEncoder(model_name, max_length=512)
    logger.info("Re-ranker ready")


def rerank_chunks(
    question: str,
    chunks: list[dict],
    top_n: int = 5,
) -> list[dict]:
    """
    Re-rank retrieved chunks by cross-encoder relevance to the question.

    Each (question, chunk_text) pair is scored. Chunks are returned sorted by
    descending relevance score (best first), capped at top_n.

    Args:
        question: The user's question string.
        chunks:   List of chunk dicts from retrieve() — each must have "text".
        top_n:    Maximum number of chunks to return after re-ranking.

    Returns:
        Subset of chunks sorted by descending reranker score.
        Each returned chunk dict gets an extra "rerank_score" field.

    Raises:
        RuntimeError: If load_reranker() has not been called.
    """
    global _reranker

    if _reranker is None:
        # Auto-load if not already done
        load_reranker()

    if not chunks:
        return []

    # Build (question, chunk_text) pairs for batch scoring
    pairs = [(question, chunk["text"]) for chunk in chunks]

    # Score all pairs in one batch — returns a numpy array of floats
    scores = _reranker.predict(pairs)

    # Attach rerank_score to each chunk and sort descending
    scored = sorted(
        [{"rerank_score": float(score), **chunk} for chunk, score in zip(chunks, scores)],
        key=lambda x: x["rerank_score"],
        reverse=True,
    )

    return scored[:top_n]
