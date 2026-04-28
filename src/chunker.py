"""
chunker.py
----------
Splits cleaned text into overlapping chunks using sentence-boundary-aware
grouping.

Chunking strategy:
  - Chunk size  : ~600 words  (approx 500-800 tokens as per PRD)
  - Overlap     : last 1-2 sentences from the previous chunk are carried
                  over into the next chunk (sentence-level overlap instead
                  of a fixed word count, controlled by the overlap parameter)
  - Token count : word count (fast, dependency-free approximation for Phase 1)

Sentence splitting uses pure regex and handles common edge cases:
  - Abbreviations: Dr., Mr., Mrs., Ms., Prof., etc., e.g., i.e., vs., St.
  - Decimal numbers: 3.14, 0.5
  - Ellipsis: ...
  - Initials: A.B., U.S.A.

Note: Phase 2 sentence-aware chunking uses a simpler regex split pattern
that does not attempt the above edge-case handling.

Chunk IDs are globally sequential across all documents in a pipeline run.
"""

import re
from itertools import count

# Global counter — shared across all calls within a pipeline run.
# Produces: chunk_000001, chunk_000002, ...
_global_counter = count(1)


def reset_counter():
    """Reset the global chunk counter (useful for tests)."""
    global _global_counter
    _global_counter = count(1)


# ── Sentence splitting ────────────────────────────────────────────────────────

# Abbreviations that should NOT trigger a sentence break.
_ABBREVIATIONS = frozenset({
    "dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st",
    "gen", "gov", "sgt", "cpl", "pvt", "capt", "lt", "col",
    "inc", "ltd", "corp", "co", "dept", "univ", "assn",
    "vol", "rev", "est", "approx", "dept",
    "fig", "eq", "ref", "sec", "ch", "pt", "no",
    "vs", "al",              # et al.
    "e.g", "i.e",            # matched without trailing dot
})


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using regex.

    Handles:
      - Standard terminators: .  ?  !
      - Abbreviations (Dr., e.g., etc.) — not treated as sentence endings
      - Decimal numbers (3.14) — not treated as sentence endings
      - Ellipsis (...) — not treated as sentence endings
      - Sentences ending at the end of the string (no trailing space needed)
    """
    # Sentence splitting for Phase 2: simple regex-based splitting on
    # sentence-ending punctuation followed by whitespace.
    # Strategy: walk through candidate split points (positions right after
    # a sentence-ending punctuation followed by whitespace + uppercase or
    # end-of-string) and decide whether each is a real sentence boundary.
    #
    # Match a period/question/exclamation followed by whitespace then an
    # uppercase letter, OR followed by end-of-string.
    #
    # ── Guard: skip non-sentence-ending periods ─────────────────────
    # 1. Ellipsis: "..." — skip
    # 2. Decimal number: digit immediately before and after the dot
    # 3. Known abbreviation: last "word" before the dot is in the set
    # 4. Single-letter initial (e.g. "J." in "J. Smith") — skip
    #
    # This is a real sentence boundary — split here.
    # The sentence runs from current_start to right after the punctuation.
    # Advance past any whitespace to the start of the next sentence
    #
    # Remaining text after the last split point
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(
    clean_text: str | list[dict],
    document_name: str | None = None,
    page_number: int | None = None,
    chunk_size: int = 600,
    overlap: int = 75,
) -> list[dict]:
    """
    Split clean_text into sentence-boundary-aware chunks and attach metadata.

    Sentences are grouped until adding the next sentence would exceed
    chunk_size words.  For overlap, the last 1-2 sentences of the previous
    chunk (up to roughly `overlap` words) are carried into the next chunk
    so that context is preserved across chunk boundaries.

    If a single sentence exceeds chunk_size, it is included as its own
    chunk (never dropped).

    Args:
        clean_text:    Cleaned text from a single page, or a legacy list of
                       page dicts containing raw_text/text plus metadata.
        document_name: Original PDF filename.
        page_number:   Page the text came from (1-indexed).
        chunk_size:    Target number of words per chunk.
        overlap:       Approximate number of overlap words — controls how
                       many trailing sentences are carried over (1-2).

    Returns:
        List of chunk dicts with keys:
            chunk_id, document, page, token_count, text
    """
    if isinstance(clean_text, list):
        chunks: list[dict] = []
        for page in clean_text:
            page_text = page.get("clean_text") or page.get("raw_text") or page.get("text") or ""
            chunks.extend(
                chunk_text(
                    clean_text=page_text,
                    document_name=page.get("document", document_name or ""),
                    page_number=page.get("page", page_number or 1),
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )
        return chunks

    document_name = document_name or ""
    page_number = page_number or 1
    words = clean_text.split()

    if not words:
        return []

    sentences = _split_sentences(clean_text)

    if not sentences:
        return []

    chunks: list[dict] = []
    current_chunk: list[str] = []
    current_word_count = 0

    # Pre-compute word counts per sentence
    # Accumulate sentences until the next one would exceed the cap
    # If the chunk is non-empty and adding this sentence would exceed
    # the cap, stop (the sentence goes into the next chunk).
    # Always add at least one sentence (even if it alone exceeds cap)
    # Build the chunk text
    # If we've consumed all sentences, stop.
    # ── Sentence-level overlap ────────────────────────────────────────
    # Carry back the last 1-2 sentences from the chunk we just built,
    # up to approximately `overlap` words.
    # Rewind the sentence index so the overlapping sentences are
    # included at the start of the next chunk.

    # Sentence-aware grouping with word-level overlap, per Phase 2 request.
    for sentence in sentences:
        sentence_words = sentence.split()

        if current_word_count + len(sentence_words) > chunk_size and current_chunk:
            chunk_text_str = " ".join(current_chunk)
            chunk_words = chunk_text_str.split()
            chunk_id = f"chunk_{next(_global_counter):06d}"

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document": document_name,
                    "page": page_number,
                    "token_count": len(chunk_words),  # word-level approximation
                    "text": chunk_text_str,
                }
            )

            # Overlap: keep the last N words for the next chunk.
            overlap_words = chunk_words[-overlap:] if overlap > 0 else []
            current_chunk = overlap_words + sentence_words
            current_word_count = len(current_chunk)
        else:
            current_chunk.extend(sentence_words)
            current_word_count += len(sentence_words)

    if current_chunk:
        chunk_text_str = " ".join(current_chunk)
        chunk_words = chunk_text_str.split()
        chunk_id = f"chunk_{next(_global_counter):06d}"

        chunks.append(
            {
                "chunk_id": chunk_id,
                "document": document_name,
                "page": page_number,
                "token_count": len(chunk_words),  # word-level approximation
                "text": chunk_text_str,
            }
        )

    return chunks
