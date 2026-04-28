"""
embedding_model.py
------------------
Loads the sentence-transformer model and generates embeddings for text chunks.

Model: all-MiniLM-L6-v2
  - Embedding dimension : 384
  - Fast CPU inference  : yes
  - Offline after first : yes (cached in ~/.cache/huggingface)

Improvements applied:
  1. Embeddings are L2-normalized after generation for better retrieval quality.
  2. Batch processing (default batch_size=32) prevents memory overflow.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import logging

from config import MODEL_NAME, EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once per process
_model: SentenceTransformer | None = None


def load_model() -> SentenceTransformer:
    """
    Load (or return cached) the sentence-transformer model.

    Returns:
        SentenceTransformer instance.
    """
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Model loaded (dim=%s)", EMBEDDING_DIM)
    return _model


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalize a 2-D array of vectors row-wise.

    Improvement #1: Normalized vectors improve cosine/L2 similarity quality.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero for any zero-length vectors
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def embed_text(text: str) -> np.ndarray:
    """
    Embed a single text string.

    Returns:
        Normalized numpy vector of shape (384,).
    """
    model = load_model()
    vector = model.encode([text], convert_to_numpy=True)  # shape (1, 384)
    return _normalize(vector)[0]


def embed_batch(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Embed a list of texts in mini-batches.

    Args:
        texts:      List of text strings to embed.
        batch_size: Number of texts per inference batch (default 32).

    Returns:
        Normalized numpy array of shape (N, 384).
    """
    model = load_model()

    all_vectors: list[np.ndarray] = []
    total = len(texts)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = texts[start:end]
        batch_vectors = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_vectors.append(batch_vectors)
        logger.debug("Embedding progress: %s/%s chunks", end, total)

    logger.debug("Embedding progress complete")

    combined = np.vstack(all_vectors).astype(np.float32)  # FAISS requires float32
    return _normalize(combined)
