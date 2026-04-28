"""
context_compressor.py
---------------------
Compress retrieved chunks before sending them to the LLM.

Problem:
  Chunks retrieved from FAISS often contain surrounding context that is
  not directly relevant to the question. This wastes precious context window
  space and can dilute the LLM's focus.

Solution:
  For each chunk, ask the LLM to extract a concise summary (100–200 words)
  that preserves all key technical information.

Trade-off:
  This is an *opt-in* feature (--mode advanced / compress=True).
  It makes N Ollama calls (one per chunk) BEFORE the final answer call,
  so total latency scales with the number of chunks.

  Typical cost: several seconds per chunk on local hardware.
  Use only when answer quality matters more than response speed.
"""

import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_engine import generate_answer
import config

logger = logging.getLogger(__name__)

# ── Compression prompt ────────────────────────────────────────────────────────
COMPRESSION_PROMPT = """\
You are a precise text compressor.

Extract and summarise the key technical information from the following text.
Your summary must:
- Be between 100 and 200 words.
- Preserve all technical terms, definitions, and important details exactly.
- Remove repetitive, redundant, or off-topic sentences.
- Be written in clear, concise prose.

Text to compress:
{text}

Compressed summary:"""


def compress_chunk(
    chunk_text: str,
    model: str = config.OLLAMA_MODEL,
) -> str:
    """
    Compress a single chunk to 100-200 words using the LLM.

    Args:
        chunk_text: The raw chunk text to compress.
        model:      Ollama model tag.

    Returns:
        Compressed text string.
    """
    prompt = COMPRESSION_PROMPT.replace("{text}", chunk_text)
    compressed = generate_answer(prompt=prompt, model=model, stream=False)
    return compressed.strip()


def compress_chunks(
    chunks: list[dict],
    model: str = config.OLLAMA_MODEL,
) -> list[dict]:
    """
    Compress a list of chunk dicts in-place (replaces "text" with compressed text).

    Progress is printed so the user knows compression is happening.

    Args:
        chunks: List of chunk dicts — each must have "text".
        model:  Ollama model tag.

    Returns:
        New list of chunk dicts with compressed "text" fields.
        Original "text" is preserved in "original_text" for debugging.
    """
    compressed_chunks = []
    total = len(chunks)

    logger.info("Compressing %s chunk(s) (this may take a moment)...", total)

    for i, chunk in enumerate(chunks, start=1):
        original = chunk["text"]
        compressed = compress_chunk(chunk_text=original, model=model)

        new_chunk = dict(chunk)
        new_chunk["original_text"] = original
        new_chunk["text"] = compressed
        compressed_chunks.append(new_chunk)

        original_words   = len(original.split())
        compressed_words = len(compressed.split())
        logger.info("[%s/%s] %s: %sw -> %sw", i, total, chunk["chunk_id"], original_words, compressed_words)

    return compressed_chunks
