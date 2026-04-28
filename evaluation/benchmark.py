"""
benchmark.py - Run test questions against multiple retrieval configurations.
Usage: python evaluation/benchmark.py [--notebook-id <id>]
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from api import database
from src.retriever import load_retriever, faiss_search, reciprocal_rank_fusion
from src.bm25_retriever import BM25Retriever
from src.reranker import rerank_chunks
from src.query_expander import expand_query

QUESTIONS_FILE = Path(__file__).parent / "test_questions.json"
RESULTS_FILE = Path(__file__).parent / "results.json"

# Define configurations to test
CONFIGS = [
    {"name": "A", "label": "FAISS only, no rerank, no multi-query", "hybrid": False, "rerank": False, "top_k": 5},
    {"name": "B", "label": "FAISS only, with rerank, no multi-query", "hybrid": False, "rerank": True, "top_k": 5},
    {"name": "C", "label": "Hybrid, with rerank, no multi-query", "hybrid": True, "rerank": True, "top_k": 5},
    {"name": "D", "label": "Hybrid, with rerank, with multi-query", "hybrid": True, "rerank": True, "top_k": 5, "multi_query": True},
    {"name": "E", "label": "Hybrid, with rerank, top_k=3", "hybrid": True, "rerank": True, "top_k": 3},
    {"name": "F", "label": "Hybrid, with rerank, top_k=8", "hybrid": True, "rerank": True, "top_k": 8}
]


def keyword_hit_rate(chunks: list[dict], keywords: list[str]) -> float:
    """Fraction of expected keywords found in retrieved chunk texts."""
    if not keywords:
        return 0.0
    combined = " ".join(c.get("text", "") for c in chunks).lower()
    hits = sum(1 for kw in keywords if kw.lower() in combined)
    return round(hits / len(keywords), 3)


def _rank_score(chunk: dict) -> float:
    return float(chunk.get("rrf_score", chunk.get("score", 0.0)))


def run_benchmark(notebook_id: str) -> list[dict]:
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    
    # Auto-generate index if missing
    index_path = config.get_index_path(notebook_id)
    if not index_path.exists():
        from src.pipeline import run_embedding_pipeline
        print(f"Index not found for notebook {notebook_id}. Auto-generating it...")
        run_embedding_pipeline(notebook_id=notebook_id)

    load_retriever(notebook_id=notebook_id)
    bm25 = BM25Retriever(notebook_id)
    results: list[dict] = []

    original_multi_query = config.ENABLE_MULTI_QUERY

    try:
        for q in questions:
            for cfg in CONFIGS:
                config.ENABLE_MULTI_QUERY = cfg.get("multi_query", False)
                entry = {
                    "question_id": q["id"],
                    "question": q["question"],
                    "category": q["category"],
                    "config": cfg["name"],
                    "config_label": cfg["label"],
                }
                try:
                    top_k = cfg.get("top_k", 5)
                    t0 = time.perf_counter()

                    queries = expand_query(question=q["question"], model=config.OLLAMA_MODEL)
                    seen: dict[str, dict] = {}
                    pool_k = max(top_k * 2, 1)

                    for query in queries:
                        faiss_chunks = faiss_search(query, top_k=pool_k)
                        if cfg.get("hybrid", False):
                            bm25_chunks = bm25.search(query, top_k=pool_k)
                            merged = reciprocal_rank_fusion(faiss_chunks, bm25_chunks)[:top_k]
                        else:
                            merged = faiss_chunks[:top_k]

                        for chunk in merged:
                            cid = chunk.get("chunk_id")
                            if not cid:
                                continue
                            score = _rank_score(chunk)
                            prev = seen.get(cid)
                            if not prev or score > _rank_score(prev):
                                candidate = dict(chunk)
                                candidate["_rank_score"] = score
                                seen[cid] = candidate

                    retrieved = sorted(seen.values(), key=_rank_score, reverse=True)
                    retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)

                    rerank_ms = 0.0
                    final_chunks = retrieved[:top_k]
                    if cfg.get("rerank", False) and final_chunks:
                        t1 = time.perf_counter()
                        final_chunks = rerank_chunks(q["question"], final_chunks, top_n=top_k)
                        rerank_ms = round((time.perf_counter() - t1) * 1000, 1)

                    entry.update({
                        "retrieval_latency_ms": retrieval_ms,
                        "rerank_latency_ms": rerank_ms,
                        "total_latency_ms": round(retrieval_ms + rerank_ms, 1),
                        "chunks_retrieved": len(final_chunks),
                        "unique_documents": len({c.get("document") for c in final_chunks if c.get("document")}),
                        "keyword_hit_rate": keyword_hit_rate(final_chunks, q["expected_keywords"]),
                        "error": None,
                    })
                except Exception as e:
                    entry.update({
                        "retrieval_latency_ms": 0,
                        "rerank_latency_ms": 0,
                        "total_latency_ms": 0,
                        "chunks_retrieved": 0,
                        "unique_documents": 0,
                        "keyword_hit_rate": 0,
                        "error": str(e),
                    })
                    print(f"  WARN: Error for Q{q['id']} Config {cfg['name']}: {e}")

                results.append(entry)
                print(
                    f"  Q{q['id']:02d} Config {cfg['name']} -> "
                    f"{entry.get('keyword_hit_rate', 0):.2f} hit rate, "
                    f"{entry.get('total_latency_ms', 0):.0f}ms"
                )
    finally:
        config.ENABLE_MULTI_QUERY = original_multi_query

    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_FILE}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook-id", help="Notebook ID to benchmark against")
    args = parser.parse_args()

    notebook_id = args.notebook_id
    if not notebook_id:
        database.init_db()
        nbs = database.list_notebooks()
        if not nbs:
            print("No notebooks found. Upload documents first.")
            sys.exit(1)
        notebook_id = nbs[0]["id"]
        print(f"Using first notebook: {nbs[0]['title']} ({notebook_id})")

    print(f"Running benchmark on notebook: {notebook_id}")
    run_benchmark(notebook_id)
