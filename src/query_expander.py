"""
query_expander.py
-----------------
Multi-query expansion for improved RAG retrieval.

When a user asks a short or ambiguous question (e.g. "What is FA?"),
the embedding may not match chunks that use different phrasing
(e.g. "finite automata").  This module calls the LLM to generate
alternative phrasings of the question, so that FAISS search can be
run against each variant and the results merged for better recall.

The expansion is controlled by two config flags:
  - ENABLE_MULTI_QUERY  (bool)  — master toggle
  - MULTI_QUERY_COUNT   (int)   — number of variants to request
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import ollama

logger = logging.getLogger(__name__)


EXPANSION_PROMPT = (
    "Generate {count} alternative phrasings of this question to improve document retrieval. "
    "Return ONLY the questions, one per line, no numbering, no explanations.\n\n"
    "Original: {question}"
)


def expand_query(
    question: str,
    model: str = config.OLLAMA_MODEL,
) -> list[str]:
    """
    Generate alternative phrasings of a user question via LLM.

    If ENABLE_MULTI_QUERY is False, or the LLM call fails, this
    gracefully falls back to returning only the original question.

    Args:
        question: The original user question.
        model:    Ollama model tag to use for expansion.

    Returns:
        A list starting with the original question followed by
        up to MULTI_QUERY_COUNT alternative phrasings.
        e.g. [original, variant1, variant2, variant3]
    """
    if not config.ENABLE_MULTI_QUERY:
        return [question]

    try:
        count = getattr(config, "MULTI_QUERY_COUNT", 3)
        prompt = EXPANSION_PROMPT.format(count=count, question=question)
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options=config.OLLAMA_OPTIONS,
            stream=False,
        )
        raw = response["message"]["content"].strip()

        # Parse: one question per line, strip whitespace, filter blanks
        variants = [line.strip() for line in raw.splitlines() if line.strip()]

        # Remove any lines that look like numbering artefacts
        cleaned: list[str] = []
        for v in variants:
            # Strip leading "1.", "2)", "- ", "* " etc. the LLM might add
            stripped = v.lstrip("0123456789.)- *").strip()
            if stripped:
                cleaned.append(stripped)

        # Cap at the configured count
        cleaned = cleaned[:count]

        if not cleaned:
            return [question]

        return [question] + cleaned

    except Exception:
        # Graceful fallback — expansion is best-effort
        return [question]
