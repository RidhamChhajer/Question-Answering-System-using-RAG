"""
compare.py - Compare benchmark results across configurations.
Usage: python evaluation/compare.py
"""
import json
import sys
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "results.json"


def main() -> None:
    if not RESULTS_FILE.exists():
        print("No results.json found. Auto-running benchmark.py...")
        import os
        import subprocess
        # Run benchmark.py using the same python executable
        benchmark_script = Path(__file__).parent / "benchmark.py"
        subprocess.run([sys.executable, str(benchmark_script)], check=True)
        if not RESULTS_FILE.exists():
            print("results.json still not found. Benchmark might have failed.")
            sys.exit(1)

    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    configs = sorted({r["config"] for r in results})

    # Compute per-config averages
    stats: dict[str, dict] = {}
    for cfg in configs:
        rows = [r for r in results if r["config"] == cfg and r["error"] is None]
        if not rows:
            continue
        stats[cfg] = {
            "label": rows[0]["config_label"],
            "avg_latency": round(sum(r["total_latency_ms"] for r in rows) / len(rows), 1),
            "avg_hit_rate": round(sum(r["keyword_hit_rate"] for r in rows) / len(rows), 3),
            "avg_chunks": round(sum(r["chunks_retrieved"] for r in rows) / len(rows), 1),
            "avg_unique_docs": round(sum(r["unique_documents"] for r in rows) / len(rows), 1),
            "n": len(rows),
        }

    header = f"{'Config':<8} {'Avg Latency':>12} {'Hit Rate':>10} {'Avg Chunks':>12} {'Unique Docs':>12}  Description"
    print("\n" + "=" * 90)
    print("RAG Benchmark Results")
    print("=" * 90)
    print(header)
    print("-" * 90)

    best_latency = min(stats.values(), key=lambda x: x["avg_latency"])["avg_latency"]
    best_hit_rate = max(stats.values(), key=lambda x: x["avg_hit_rate"])["avg_hit_rate"]

    for cfg, s in stats.items():
        lat_mark = " *" if s["avg_latency"] == best_latency else "  "
        hit_mark = " *" if s["avg_hit_rate"] == best_hit_rate else "  "
        print(
            f"{cfg:<8}"
            f"{str(s['avg_latency']) + 'ms':>12}{lat_mark}"
            f"{str(s['avg_hit_rate']):>10}{hit_mark}"
            f"{str(s['avg_chunks']):>12}"
            f"{str(s['avg_unique_docs']):>12}  "
            f"{s['label']}"
        )

    print("=" * 90)
    print("* = best in category\n")

    best = max(stats.values(), key=lambda x: x["avg_hit_rate"])
    best_name = [k for k, v in stats.items() if v == best][0]
    print(f"Recommendation: Config {best_name} achieves the highest keyword hit rate ({best['avg_hit_rate']}).")
    print(f"  -> {best['label']}\n")


if __name__ == "__main__":
    main()
