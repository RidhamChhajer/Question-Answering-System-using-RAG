"""
vector_store.py
---------------
Manages the FAISS vector index and its associated metadata.

Index type : auto-selected based on vector count
  - IndexFlatL2   (n_vectors < IVF_THRESHOLD)
      Exact nearest-neighbour search via L2 distance.
      No training required. Works well with normalized vectors.
  - IndexIVFFlat  (n_vectors >= IVF_THRESHOLD)
      Inverted-file flat index; partitions the space into nlist clusters.
      Requires training before vectors can be added.
      nlist = int(sqrt(n_vectors)) — a common rule of thumb.

Metadata is stored separately in metadata.json.
  - Index position i  →  metadata["chunks"][i]
  - Includes: chunk_id, document, page, token_count, text  (Improvement #2)

Note: faiss.read_index() handles both IndexFlatL2 and IndexIVFFlat
transparently, so save_index / load_index require no changes.
"""

import json
import os

import faiss
import numpy as np

import config


def create_index(dim: int, n_vectors: int = 0) -> faiss.Index:
    """
    Create a new empty FAISS index, auto-selecting the type based on
    the expected number of vectors.

    - n_vectors < IVF_THRESHOLD  → IndexFlatL2 (exact, no training needed)
    - n_vectors >= IVF_THRESHOLD → IndexIVFFlat (approximate, requires training
                                   via train_index() before adding vectors)

    Args:
        dim:       Embedding dimension (384 for all-MiniLM-L6-v2).
        n_vectors: Expected number of vectors to be stored (default 0).

    Returns:
        An empty faiss.Index (IndexFlatL2 or IndexIVFFlat).
    """
    if n_vectors < config.IVF_THRESHOLD:
        print(
            f"   Index type : IndexFlatL2 ({n_vectors} vectors < threshold {config.IVF_THRESHOLD})"
        )
        return faiss.IndexFlatL2(dim)
    else:
        nlist = config.IVF_NLIST  # number of IVF clusters
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.nprobe = config.IVF_NPROBE
        print(
            f"   Index type : IndexIVFFlat "
            f"({n_vectors} vectors >= threshold {config.IVF_THRESHOLD}, nlist={nlist})"
        )
        return index


def build_index(embeddings: np.ndarray, dim: int) -> faiss.Index:
    """
    Build and populate a FAISS index from embeddings, auto-selecting
    IndexFlatL2 or IndexIVFFlat based on vector count.

    Args:
        embeddings: Float32 numpy array of shape (N, dim).
        dim:        Embedding dimension.

    Returns:
        A populated faiss.Index.
    """
    n_vectors = len(embeddings)
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    if n_vectors >= config.IVF_THRESHOLD:
        # Use IVF for large corpora
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, config.IVF_NLIST)
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = config.IVF_NPROBE
    else:
        # Use brute-force for small corpora
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

    return index


def train_index(index: faiss.Index, vectors: np.ndarray) -> None:
    """
    Train the FAISS index if it requires training (e.g. IndexIVFFlat).

    For index types that do not need training (e.g. IndexFlatL2),
    this function is a no-op — index.is_trained is already True.

    Args:
        index:   FAISS index returned by create_index().
        vectors: Float32 numpy array of shape (N, dim) used for training.
    """
    if not index.is_trained:
        print("   Training IVF index …")
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        index.train(vectors)
        print("   Training complete.")


def add_embeddings(index: faiss.IndexFlatL2, vectors: np.ndarray) -> None:
    """
    Add a matrix of embeddings to the FAISS index in-place.

    Args:
        index:   FAISS index to add to.
        vectors: Float32 numpy array of shape (N, dim).
    """
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    index.add(vectors)


def save_index(index: faiss.IndexFlatL2, path: str) -> None:
    """
    Persist the FAISS index to disk.

    Args:
        index: The FAISS index to save.
        path:  Full file path (e.g. vector_store/faiss_index.bin).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)


def load_index(path: str) -> faiss.IndexFlatL2:
    """
    Load a FAISS index from disk.

    Args:
        path: Full file path to the saved index.

    Returns:
        Loaded faiss.IndexFlatL2 index.
    """
    index = faiss.read_index(path)
    if hasattr(index, "nprobe"):
        index.nprobe = config.IVF_NPROBE
    return index


def save_metadata(chunks: list[dict], path: str) -> None:
    """
    Save chunk metadata to a JSON file.

    Each entry preserves: chunk_id, document, page, token_count, text.
    Improvement #2: token_count is explicitly carried through from chunks.

    Args:
        chunks: List of chunk dicts from the chunking step.
        path:   Full file path (e.g. vector_store/metadata.json).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    metadata_entries = [
        {
            "chunk_id":    chunk["chunk_id"],
            "document":    chunk["document"],
            "page":        chunk["page"],
            "token_count": chunk.get("token_count", 0),  # Improvement #2
            "text":        chunk["text"],
        }
        for chunk in chunks
    ]

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"chunks": metadata_entries}, f, ensure_ascii=False, indent=2)


def load_metadata(path: str) -> list[dict]:
    """
    Load chunk metadata from a JSON file.

    Returns:
        List of chunk metadata dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["chunks"]
