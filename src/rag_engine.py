"""
rag_engine.py
-------------
Full RAG orchestration: retrieval → rerank → context assembly → prompt → LLM.

Phase 4 pipeline:
    retrieve(top_k) → assemble_context → build_prompt → LLM

Phase 5 pipeline (default):
    retrieve(top_20) → rerank(top_5) → assemble_context → build_prompt → LLM

Phase 5 pipeline (--mode mapreduce):
    retrieve(top_20) → rerank(top_5) → [compress?] → map_reduce_ask → LLM×N → combine

Context cap: ~3000 words (safe for most 4K token context windows).
Best-scoring chunks always make it in first.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import retrieve, faiss_search, reciprocal_rank_fusion
from src.bm25_retriever import BM25Retriever
from src.llm_engine import generate_answer
import config

# ── Constants ──────────────────────────────────────────────────────────────────
FALLBACK_ANSWER    = "I could not find the answer in the provided documents."

# ── Prompt Template (Phase 5 — synthesis-focused) ─────────────────────────────
PROMPT_TEMPLATE = """\
You are an expert teaching assistant helping a student understand material from a document.

Your task is to answer the question using ONLY the information contained in the provided context.

Important instructions:
1. Carefully read all context sections before answering.
2. Combine information from multiple sections when needed.
3. Do NOT copy large sentences directly from the context.
4. Rewrite explanations clearly in your own words while preserving the technical terminology used in the document.
5. Keep the explanation faithful to the document.
6. Do NOT introduce outside knowledge that is not supported by the context.
7. If the document only provides partial information, explain what is available and clearly state that the document does not contain further detail.

When writing the answer:
• Start with a clear explanation of the concept.
• If definitions are present, include them.
• Expand the explanation using supporting details from the context.
• Organise the explanation logically.
• Maintain the terminology used in the document.
• Avoid unnecessary repetition.

If the answer cannot be found in the context at all, respond with exactly:
"{fallback}"

Context from documents:
{{context}}

Question:
{{question}}

Answer:""".format(fallback=FALLBACK_ANSWER)



def assemble_context(chunks: list[dict], max_words: int = config.CONTEXT_CAP) -> tuple[str, list[dict]]:
    """
    Join retrieved chunk texts into a single context string.

    Chunks are already sorted by descending similarity (best first).
    We accumulate until we hit the word cap, then stop — so the most
    relevant content always appears in the context window.

    Args:
        chunks:    Ranked list of chunk dicts from retrieve().
        max_words: Word cap for the assembled context.

    Returns:
        (context_string, included_chunks)
        context_string  — formatted text ready for the prompt
        included_chunks — subset of chunks that fit within the cap
    """
    context_parts: list[str] = []
    included: list[dict] = []
    word_count = 0

    for chunk in chunks:
        chunk_words = len(chunk["text"].split())
        if word_count + chunk_words > max_words:
            break
        source_label = f"[Source: {chunk['document']}, page {chunk['page']}]"
        context_parts.append(f"{source_label}\n{chunk['text']}")
        included.append(chunk)
        word_count += chunk_words

    return "\n\n".join(context_parts), included


def format_history(history: list[dict], max_words: int) -> str:
    """
    Format conversation history as a string.
    history is a list of {"role": "user"/"ai", "content": "..."}
    Truncates by dropping OLDEST messages first until within max_words budget.
    """
    if not history:
        return ""
    formatted_lines = []
    for msg in history:
        prefix = "User" if msg["role"] == "user" else "Assistant"
        formatted_lines.append(f"{prefix}: {msg['content']}")

    # Drop oldest messages until within word budget
    while formatted_lines:
        combined = "\n".join(formatted_lines)
        if len(combined.split()) <= max_words:
            break
        formatted_lines.pop(0)  # drop oldest

    return "\n".join(formatted_lines)


def build_prompt(context: str, question: str, history: str | None = "") -> str:
    """
    Fill the prompt template with context and question.

    Args:
        context:  Assembled context string from assemble_context().
        question: Raw user question string.

    Returns:
        Fully-formed prompt string ready for the LLM.
    """
    prompt = PROMPT_TEMPLATE
    if history:
        prompt = prompt.replace(
            "Context from documents:\n{context}",
            "Previous conversation (for context only — answer based on the documents, not this history):\n"
            f"{history}\n\nContext from documents:\n{{context}}",
        )
    return prompt.replace("{context}", context).replace("{question}", question)


def ask(
    question: str,
    top_k: int = config.RERANK_TOP_N,
    model: str = config.OLLAMA_MODEL,
    history: list[dict] | None = None,
    stream: bool = True,
    rerank: bool = True,
    compress: bool = False,
    mode: str = "standard",      # "standard" | "mapreduce"
) -> dict:
    """
    Run the full RAG pipeline for a user question.

    Phase 5 pipeline (default, mode="standard"):
        retrieve(RETRIEVAL_POOL=20) → rerank(top_k=5) → assemble_context → LLM

    Phase 5 pipeline (mode="mapreduce"):
        retrieve(RETRIEVAL_POOL=20) → rerank(top_k=5) → [compress?] → map-reduce LLM

    Args:
        question: Natural language question from the user.
        top_k:    Final number of chunks sent to the LLM (after reranking).
        model:    Ollama model tag.
        stream:   Stream LLM output to stdout token-by-token.
        rerank:   If True, re-rank FAISS results with CrossEncoder (default True).
        compress: If True, compress each chunk with LLM before assembly (slow, opt-in).
        mode:     "standard" = single LLM call; "mapreduce" = per-chunk + reduce.

    Returns:
        {
          "question":      str,
          "answer":        str,
          "sources":       list of chunk dicts included in context,
          "context_words": int
        }
    """
    # ── Step 1: Retrieve a larger pool of candidates ───────────────────────────
    pool_size = config.RETRIEVAL_POOL if rerank else top_k

    # Multi-query expansion: generate alternative phrasings, retrieve for each,
    # and merge results (deduplicated by chunk_id, keeping highest score).
    from src.query_expander import expand_query
    queries = expand_query(question=question, model=model)

    seen: dict[str, dict] = {}   # chunk_id → best chunk dict
    bm25 = BM25Retriever(notebook_id=None) if config.ENABLE_HYBRID_SEARCH else None
    candidate_k = pool_size * config.BM25_TOP_K_MULTIPLIER

    for q in queries:
        if config.ENABLE_HYBRID_SEARCH:
            faiss_chunks = faiss_search(query=q, top_k=candidate_k)
            bm25_chunks = bm25.search(query=q, top_k=candidate_k) if bm25 else []
            merged = reciprocal_rank_fusion(faiss_chunks, bm25_chunks, k=config.RRF_K)
            candidates = merged[:pool_size]
        else:
            candidates = retrieve(query=q, top_k=pool_size)

        for chunk in candidates:
            cid = chunk["chunk_id"]
            candidate = dict(chunk)
            if "rrf_score" in candidate:
                candidate["score"] = candidate["rrf_score"]
            if cid not in seen or candidate["score"] > seen[cid]["score"]:
                seen[cid] = candidate
    chunks = sorted(seen.values(), key=lambda c: c["score"], reverse=True)

    if not chunks:
        return {
            "question":      question,
            "answer":        FALLBACK_ANSWER,
            "sources":       [],
            "context_words": 0,
        }

    # ── Step 2: Re-rank (optional but on by default) ───────────────────────────
    if rerank and len(chunks) > 1:
        from src.reranker import rerank_chunks
        chunks = rerank_chunks(question=question, chunks=chunks, top_n=top_k)
    else:
        chunks = chunks[:top_k]

    # ── Step 3: Compress chunks (opt-in — slow) ────────────────────────────────
    if compress:
        from src.context_compressor import compress_chunks
        chunks = compress_chunks(chunks=chunks, model=model)

    # ── Step 4: Generate answer ────────────────────────────────────────────────
    if mode == "mapreduce":
        from src.map_reduce_engine import map_reduce_ask
        answer = map_reduce_ask(question=question, chunks=chunks,
                                model=model, stream=stream)
        included_chunks = chunks
        context_words   = sum(len(c["text"].split()) for c in included_chunks)
    else:
        # Standard: assemble context → single LLM call
        context, included_chunks = assemble_context(chunks)
        context_words = sum(len(c["text"].split()) for c in included_chunks)
        history_str = format_history(history, config.HISTORY_MAX_WORDS) if history else ""
        prompt = build_prompt(context=context, question=question, history=history_str)
        answer = generate_answer(prompt=prompt, model=model, stream=stream)

    return {
        "question":      question,
        "answer":        answer.strip(),
        "sources":       included_chunks,
        "context_words": context_words,
    }
