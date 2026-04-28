"""
map_reduce_engine.py
--------------------
Two-stage answer generation for complex, multi-chunk topics.

The Problem:
  When a topic spans multiple document chunks, a single LLM call reading all
  chunks at once may "average out" the information, missing details from any
  individual chunk.

The Solution — Map-Reduce:
  Stage 1 (Map):   Ask the LLM one question per chunk independently.
                   Each chunk produces a focused partial answer.
  Stage 2 (Reduce): Combine all partial answers into one coherent final answer.

This mimics how a student would read a textbook: understand each section
individually, then synthesize a complete explanation.

Trade-off:
  Requires N+1 LLM calls (N = number of chunks).
  Slower than standard generation but produces more complete answers for
  multi-page topics like system design, proofs, or algorithms.

Use via: python main.py --ask "..." --mode mapreduce
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_engine import generate_answer
import config

logger = logging.getLogger(__name__)

# ── Map prompt ────────────────────────────────────────────────────────────────
MAP_PROMPT = """\
Using ONLY the text provided below, answer the question as completely as possible.
If the text does not contain relevant information, respond with "Not covered in this section."

Text:
{chunk}

Question:
{question}

Answer:"""

# ── Reduce prompt ─────────────────────────────────────────────────────────────
REDUCE_PROMPT = """\
You are an expert teaching assistant.

The following partial answers were each generated from a different section of the same document.
Some sections may say "Not covered in this section." — ignore those.

Combine the remaining partial answers into a single, well-structured, comprehensive explanation.
Eliminate repetition. Preserve all technical terminology. Organise the answer logically.
Do NOT introduce outside knowledge.

Partial answers:
{answers}

Question:
{question}

Final Answer:"""


def generate_partial_answers(
    question: str,
    chunks: list[dict],
    model: str = config.OLLAMA_MODEL,
) -> list[str]:
    """
    Map stage: generate one partial answer per chunk.

    Args:
        question: The user's question.
        chunks:   List of chunk dicts — each must have "text" and "chunk_id".
        model:    Ollama model tag.

    Returns:
        List of partial answer strings, one per chunk.
    """
    partial_answers: list[str] = []
    total = len(chunks)
    logger.info("[Map] Generating %s partial answer(s)...", total)

    for i, chunk in enumerate(chunks, start=1):
        logger.info("[%s/%s] Processing %s (%s, p.%s)...", i, total, chunk["chunk_id"], chunk["document"], chunk["page"])

        prompt = MAP_PROMPT.replace("{chunk}", chunk["text"]).replace("{question}", question)
        partial = generate_answer(prompt=prompt, model=model, stream=False)
        partial = partial.strip()
        partial_answers.append(partial)

        # Summarise the partial answer length for the user
        word_count = len(partial.split())
        logger.info("Partial answer length: %sw", word_count)

    return partial_answers


def combine_answers(
    question: str,
    partial_answers: list[str],
    model: str = config.OLLAMA_MODEL,
    stream: bool = True,
) -> str:
    """
    Reduce stage: synthesise partial answers into a final coherent answer.

    Args:
        question:        The original user question.
        partial_answers: List of strings from generate_partial_answers().
        model:           Ollama model tag.
        stream:          If True, stream final answer tokens to stdout.

    Returns:
        Final synthesised answer string.
    """
    # Filter out "not covered" responses to avoid polluting the reduce context
    useful = [a for a in partial_answers
              if "not covered in this section" not in a.lower() and len(a.strip()) > 20]

    if not useful:
        return "I could not find the answer in the provided documents."

    answers_block = "\n\n---\n\n".join(
        f"[Section {i+1}]:\n{ans}" for i, ans in enumerate(useful)
    )

    prompt = REDUCE_PROMPT.replace("{answers}", answers_block).replace("{question}", question)

    logger.info("[Reduce] Combining %s partial answer(s) into final response...", len(useful))
    final = generate_answer(prompt=prompt, model=model, stream=stream)
    return final.strip()


def map_reduce_ask(
    question: str,
    chunks: list[dict],
    model: str = config.OLLAMA_MODEL,
    stream: bool = True,
) -> str:
    """
    Run the full map-reduce pipeline for a question over a set of chunks.

    Args:
        question: The user's question.
        chunks:   Re-ranked chunks from retriever + reranker.
        model:    Ollama model tag.
        stream:   Stream the final reduce output to stdout.

    Returns:
        Final synthesised answer string.
    """
    partial_answers = generate_partial_answers(question=question, chunks=chunks, model=model)
    return combine_answers(question=question, partial_answers=partial_answers,
                           model=model, stream=stream)
