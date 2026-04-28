"""Unit tests for src/chunker.py"""
import pytest
from src.chunker import chunk_text, reset_counter


@pytest.fixture(autouse=True)
def reset():
    reset_counter()


def _make_pages(text, doc="test.pdf", page=1):
    return [{"document": doc, "page": page, "raw_text": text}]


def test_empty_input():
    result = chunk_text([], chunk_size=200, overlap=40)
    assert result == []


def test_single_short_sentence():
    pages = _make_pages("This is a short sentence.")
    chunks = chunk_text(pages, chunk_size=200, overlap=40)
    assert len(chunks) == 1
    assert "short sentence" in chunks[0]["text"]


def test_chunk_dict_format():
    pages = _make_pages("Hello world. This is a test sentence.")
    chunks = chunk_text(pages, chunk_size=200, overlap=40)
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "document" in chunk
        assert "page" in chunk
        assert "token_count" in chunk
        assert "text" in chunk


def test_global_ids_sequential():
    pages = _make_pages("A " * 50 + "B " * 50)
    chunks1 = chunk_text(pages, chunk_size=30, overlap=5)
    chunks2 = chunk_text(pages, chunk_size=30, overlap=5)
    all_ids = [c["chunk_id"] for c in chunks1 + chunks2]
    assert all_ids == sorted(all_ids)
    assert len(set(all_ids)) == len(all_ids)


def test_reset_counter():
    pages = _make_pages("Word " * 100)
    chunk_text(pages, chunk_size=30, overlap=5)
    reset_counter()
    chunks = chunk_text(pages, chunk_size=30, overlap=5)
    assert chunks[0]["chunk_id"] == "chunk_000001"


def test_document_and_page_preserved():
    pages = _make_pages("Hello world.", doc="myfile.pdf", page=7)
    chunks = chunk_text(pages, chunk_size=200, overlap=40)
    assert chunks[0]["document"] == "myfile.pdf"
    assert chunks[0]["page"] == 7
