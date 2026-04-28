"""
main.py
-------
Entry point for the RAG pipeline — Phases 1, 2, 3, 4 and 5.

Usage:
    python main.py                              → Phase 1 + 2 (build pipeline)
    python main.py --phase 1                   → Phase 1 only: PDF → chunks.json
    python main.py --phase 2                   → Phase 2 only: chunks.json → FAISS
    python main.py --query "What is ML?"       → Phase 3: semantic search
    python main.py --ask   "What is ML?"       → Phase 4/5: full RAG answer (standard)
    python main.py --ask   "What is ML?" --mode mapreduce  → Phase 5: map-reduce
    python main.py --ask   "What is ML?" --mode mapreduce --top-k 8

Place PDF files in:  data/raw_pdfs/
Chunks output:       processed/chunks.json
Vector index:        vector_store/faiss_index.bin
Metadata:            vector_store/metadata.json
"""

import argparse
import os
import sys

# Ensure imports work regardless of where Python is invoked from
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.pipeline import run_pipeline, run_embedding_pipeline
from src.retriever import load_retriever, retrieve
from src.rag_engine import ask as rag_ask

# ── Configuration (imported from config.py) ───────────────────────────────────
import config
# ─────────────────────────────────────────────────────────────────────────────


def print_header(title: str) -> None:
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def phase1() -> list[dict]:
    """Run Phase 1: PDF → chunks.json"""
    print_header("Phase 1 — Document Processing Pipeline")
    print(f"  Input  : {config.INPUT_FOLDER}")
    print(f"  Output : {config.CHUNKS_PATH}")
    print(f"  Chunk size / overlap: {config.CHUNK_SIZE} / {config.CHUNK_OVERLAP} words")
    print("-" * 60)

    if not os.path.isdir(config.INPUT_FOLDER):
        print(f"❌ Input folder not found: {config.INPUT_FOLDER}")
        print("   Create 'data/raw_pdfs/' and place PDF files inside it.")
        sys.exit(1)

    chunks = run_pipeline(
        input_folder=config.INPUT_FOLDER,
        output_path=config.CHUNKS_PATH,
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP,
    )

    if chunks:
        docs = {c["document"] for c in chunks}
        print("-" * 60)
        print(f"✅ Processed {len(docs)} document(s) → {len(chunks)} chunk(s)")
        print(f"   Saved to: {config.CHUNKS_PATH}")
    else:
        print("⚠️  Pipeline produced no chunks. Check the input folder.")

    print("=" * 60)
    return chunks


def phase2() -> None:
    """Run Phase 2: chunks.json → FAISS index + metadata.json"""
    print_header("Phase 2 — Embedding & Vector Index")
    print(f"  Chunks input  : {config.CHUNKS_PATH}")
    print(f"  Index output  : {config.INDEX_PATH}")
    print(f"  Metadata      : {config.METADATA_PATH}")
    print(f"  Batch size    : {config.BATCH_SIZE}")
    print("-" * 60)

    try:
        n_vectors = run_embedding_pipeline(
            chunks_path=config.CHUNKS_PATH,
            index_path=config.INDEX_PATH,
            metadata_path=config.METADATA_PATH,
            batch_size=config.BATCH_SIZE,
        )
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    print("-" * 60)
    print(f"✅ FAISS index ready  →  {n_vectors} vectors (dim=384)")
    print("=" * 60)


def phase3(query: str, top_k: int = config.TOP_K) -> None:
    """Run Phase 3: semantic retrieval for a user query."""
    print_header("Phase 3 — Semantic Retrieval")
    print(f"  Query: {query}")
    print(f"  Top-K: {top_k}")
    print("-" * 60)

    try:
        load_retriever(index_path=config.INDEX_PATH, metadata_path=config.METADATA_PATH)
    except FileNotFoundError as e:
        print(f"\u274c {e}")
        sys.exit(1)

    results = retrieve(query=query, top_k=top_k)

    if not results:
        print("\u26a0\ufe0f  No results found. Is the vector store populated?")
        print("=" * 60)
        return

    print(f"\n  Found {len(results)} result(s):\n")
    for rank, res in enumerate(results, start=1):
        print(f"  {rank}. [{res['document']}  —  page {res['page']}]  "
              f"score={res['score']:.4f}  tokens={res['token_count']}")
        # Print up to 220 chars of the chunk text, indented
        preview = res['text'][:220].replace('\n', ' ')
        if len(res['text']) > 220:
            preview += '...'
        print(f"     \"{preview}\"")
        print()

    print("=" * 60)


def phase4(question: str, top_k: int = config.TOP_K, model: str = config.OLLAMA_MODEL,
           mode: str = "standard") -> None:
    """Run Phase 4: full RAG — retrieve + LLM answer generation."""
    print_header("Phase 4/5 — RAG Answer Generation")
    print(f"  Question : {question}")
    print(f"  Top-K    : {top_k}")
    print(f"  Model    : {model}")
    print(f"  Mode     : {mode}")
    print("-" * 60)

    # Load retriever singleton
    try:
        load_retriever(index_path=config.INDEX_PATH, metadata_path=config.METADATA_PATH)
    except FileNotFoundError as e:
        print(f"\u274c {e}")
        sys.exit(1)

    print()
    print(f"Question:\n{question}")
    print()
    print("Answer:")

    # rag_ask streams the answer to stdout as it generates
    try:
        result = rag_ask(question=question, top_k=top_k, model=model,
                         stream=True, rerank=True, mode=mode)
    except RuntimeError as e:
        print(f"\n{e}")
        sys.exit(1)

    print()
    print("-" * 60)

    # Print source citations
    if result["sources"]:
        print(f"  Sources used ({len(result['sources'])} chunk(s),"
              f" ~{result['context_words']} words of context):")
        seen = set()
        for src in result["sources"]:
            key = (src["document"], src["page"])
            if key not in seen:
                print(f"    • {src['document']}  —  page {src['page']}"
                      f"  (score={src['score']:.3f})")
                seen.add(key)
    else:
        print("  No source chunks were retrieved.")

    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Pipeline — Phases 1, 2, 3, 4 & 5",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=None,
        help=(
            "1 = document processing only (PDF → chunks.json)\n"
            "2 = embedding only          (chunks.json → FAISS)\n"
            "omit = run both phases sequentially"
        ),
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        metavar="QUESTION",
        help="Phase 3: semantic search (no LLM, returns raw chunks).",
    )
    parser.add_argument(
        "--ask",
        type=str,
        default=None,
        metavar="QUESTION",
        help="Phase 4: full RAG — retrieve + generate answer with the LLM.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.TOP_K,
        metavar="K",
        help=f"Number of chunks to retrieve (default: {config.TOP_K}).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.OLLAMA_MODEL,
        metavar="MODEL",
        help=f"Ollama model tag for --ask (default: {config.OLLAMA_MODEL}).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "mapreduce"],
        default="standard",
        metavar="MODE",
        help="standard = single LLM call (default)\nmapreduce = per-chunk answers + synthesis (slower, more complete).",
    )
    args = parser.parse_args()

    if args.ask:
        # Phase 4/5: full RAG answer
        phase4(question=args.ask, top_k=args.top_k, model=args.model, mode=args.mode)
    elif args.query:
        # Phase 3: raw retrieval
        phase3(query=args.query, top_k=args.top_k)
    elif args.phase == 1:
        phase1()
    elif args.phase == 2:
        phase2()
    else:
        # Default: build pipeline (Phase 1 + 2)
        phase1()
        print()
        phase2()


if __name__ == "__main__":
    main()
