"""
document_loader.py
------------------
Loads all PDF files from a given folder, extracts text page by page using
PyMuPDF (fitz), and skips blank pages. Returns a list of page-level dicts.
"""

import os
import fitz  # PyMuPDF
import docx  # python-docx
import logging

logger = logging.getLogger(__name__)


def _load_pdf(filepath: str, filename: str) -> list[dict]:
    """Load pages from a PDF file using PyMuPDF."""
    pages = []
    try:
        doc = fitz.open(filepath)
    except Exception as e:
        logger.error("❌ Could not open %s: %s", filename, e)
        return pages
    for page_index in range(len(doc)):
        raw_text = doc[page_index].get_text()
        if not raw_text.strip():
            continue
        pages.append({"document": filename, "page": page_index + 1, "raw_text": raw_text})
    doc.close()
    return pages


def _load_txt(filepath: str, filename: str) -> list[dict]:
    """Load a plain text file as a single page."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception as e:
            logger.error("❌ Could not read %s: %s", filename, e)
            return []
    except Exception as e:
        logger.error("❌ Could not read %s: %s", filename, e)
        return []
    if not content.strip():
        return []
    return [{"document": filename, "page": 1, "raw_text": content}]


def _load_docx(filepath: str, filename: str) -> list[dict]:
    """Load a Word document, splitting into synthetic pages of ~3000 chars."""
    try:
        doc = docx.Document(filepath)
    except Exception as e:
        logger.error("❌ Could not open %s: %s", filename, e)
        return []
    full_text = "\n".join(para.text for para in doc.paragraphs)
    if not full_text.strip():
        return []
    pages = []
    page_size = 3000
    for i in range(0, len(full_text), page_size):
        chunk = full_text[i:i + page_size]
        if chunk.strip():
            pages.append({"document": filename, "page": len(pages) + 1, "raw_text": chunk})
    return pages


def load_documents(folder_path: str) -> list[dict]:
    """
    Scan folder_path for PDF files and extract text from each page.

    Returns:
        List of dicts:
        {
            "document": <filename>,
            "page":     <1-indexed page number>,
            "raw_text": <extracted text string>
        }
    """
    supported_exts = {'.pdf', '.txt', '.docx'}
    files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f.lower())[1] in supported_exts
    ]

    if not files:
        logger.warning("⚠️  No supported files (.pdf, .txt, .docx) found in %s", folder_path)
        return []

    all_pages = []
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        logger.info("📄 Loading: %s", filename)
        ext = os.path.splitext(filename.lower())[1]
        if ext == '.pdf':
            all_pages.extend(_load_pdf(filepath, filename))
        elif ext == '.txt':
            all_pages.extend(_load_txt(filepath, filename))
        elif ext == '.docx':
            all_pages.extend(_load_docx(filepath, filename))

    return all_pages
