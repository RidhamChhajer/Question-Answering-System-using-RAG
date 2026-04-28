import os
from pathlib import Path

OLLAMA_HOST = "http://127.0.0.1:11434"
os.environ["OLLAMA_HOST"] = OLLAMA_HOST

# Paths
BASE_DIR        = Path(__file__).parent
DATA_DIR        = BASE_DIR / "data"
PROCESSED_DIR   = BASE_DIR / "processed"
VECTOR_DIR      = BASE_DIR / "vector_store"
DB_PATH         = BASE_DIR / "notebooks.db"

def get_pdf_dir(notebook_id: str) -> Path:
    return DATA_DIR / notebook_id / "raw_pdfs"

def get_processed_dir(notebook_id: str) -> Path:
    return PROCESSED_DIR / notebook_id

def get_vector_dir(notebook_id: str) -> Path:
    return VECTOR_DIR / notebook_id

def get_chunks_path(notebook_id: str) -> Path:
    return PROCESSED_DIR / notebook_id / "chunks.json"

def get_index_path(notebook_id: str) -> Path:
    return VECTOR_DIR / notebook_id / "faiss_index.bin"

def get_metadata_path(notebook_id: str) -> Path:
    return VECTOR_DIR / notebook_id / "metadata.json"

def get_bm25_path(notebook_id: str) -> Path:
    return VECTOR_DIR / notebook_id / "bm25_corpus.pkl"

# Aliases for main.py
INPUT_FOLDER    = DATA_DIR / "raw_pdfs"
PDF_DIR         = INPUT_FOLDER
CHUNKS_PATH     = PROCESSED_DIR / "chunks.json"
INDEX_PATH      = VECTOR_DIR / "faiss_index.bin"
METADATA_PATH   = VECTOR_DIR / "metadata.json"

# Chunking
CHUNK_SIZE      = 600   # words per chunk
CHUNK_OVERLAP   = 75    # words of overlap between chunks
OVERLAP         = CHUNK_OVERLAP  # alias for backwards compatibility

# Embedding & retrieval
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL_NAME      = EMBEDDING_MODEL # alias
EMBEDDING_DIM   = 384
TOP_K           = 5     # number of chunks to retrieve
RETRIEVAL_POOL  = 20
RERANK_TOP_N    = 5

# LLM
OLLAMA_MODEL    = "llama3.2:3b"
LLM_MODEL       = OLLAMA_MODEL # alias
OLLAMA_OPTIONS  = {
    "num_ctx": 8192,
    "num_gpu": 17,
    "num_thread": 8,
}
LLM_OPTIONS     = OLLAMA_OPTIONS # alias
CONTEXT_CAP     = 3000  # max words of context fed to LLM
MAX_CONTEXT_WORDS = CONTEXT_CAP # alias

# Conversation history
HISTORY_MESSAGES  = 4    # number of recent messages to include (2 user + 2 AI)
HISTORY_MAX_WORDS = 500  # max word budget for history; drop oldest if exceeded

# Indexing
BATCH_SIZE      = 32    # embedding batch size

# Multi-query expansion
ENABLE_MULTI_QUERY = True
MULTI_QUERY_COUNT  = 3

# Hybrid search
ENABLE_HYBRID_SEARCH = True
RRF_K                = 60
BM25_TOP_K_MULTIPLIER = 2

# FAISS index auto-selection
IVF_THRESHOLD = 5000
IVF_NLIST     = 100
IVF_NPROBE    = 10

import logging

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL  = logging.INFO

logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
