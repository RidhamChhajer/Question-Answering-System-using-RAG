"""
text_cleaner.py
---------------
Cleans raw text extracted from a PDF page.

Goals:
  - Remove excessive whitespace and blank lines
  - Remove bare page-number lines (standalone integers, "Page N" patterns)
  - Preserve sentence meaning and readability
"""

import re


def clean_text(raw_text: str) -> str:
    """
    Clean raw PDF text and return a single readable string.

    Steps:
      1. Remove lines that are only a page number (e.g. "3", "Page 3", "- 3 -")
      2. Collapse multiple newlines into a single space
      3. Collapse multiple spaces into one
      4. Strip leading/trailing whitespace
    """
    # 1. Remove lines that look like page numbers
    #    Matches: bare integers, "Page N", "p. N", "- N -", "[ N ]"
    page_num_pattern = re.compile(
        r"^\s*(\[?\-?\s*)?(page\s*)?\d+\s*(\-?\s*\]?)?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    text = page_num_pattern.sub("", raw_text)

    # 2. Replace multiple newlines / carriage returns with a single space
    text = re.sub(r"[\r\n]+", " ", text)

    # 3. Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    # 4. Strip
    text = text.strip()

    return text
