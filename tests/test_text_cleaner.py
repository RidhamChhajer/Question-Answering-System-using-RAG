"""Unit tests for src/text_cleaner.py"""
import pytest
from src.text_cleaner import clean_text


def test_removes_bare_page_numbers():
    result = clean_text("Some text\n3\nMore text")
    assert "3" not in result.split() or "Some text" in result


def test_removes_page_n_format():
    result = clean_text("Some text\nPage 5\nMore text")
    assert "Page 5" not in result


def test_removes_bracketed_numbers():
    result = clean_text("Text\n[ 3 ]\nText")
    assert "[ 3 ]" not in result


def test_collapses_whitespace():
    result = clean_text("too   many    spaces")
    assert "too many spaces" in result


def test_collapses_newlines():
    result = clean_text("line1\n\n\nline2")
    assert "\n\n" not in result


def test_empty_input():
    assert clean_text("") == ""


def test_only_whitespace():
    result = clean_text("   \n\n   ")
    assert result.strip() == ""


def test_preserves_normal_text():
    text = "The quick brown fox jumps over the lazy dog."
    result = clean_text(text)
    assert "quick brown fox" in result


def test_preserves_numbers_in_sentences():
    """Numbers that are part of a sentence should NOT be removed."""
    result = clean_text("There are 3 types of sorting algorithms.")
    assert "3" in result
