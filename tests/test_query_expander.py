"""Unit tests for src/query_expander.py -- mocks Ollama calls."""
import pytest
from unittest.mock import patch, MagicMock
from src import query_expander
import config


@pytest.fixture(autouse=True)
def enable_multi_query(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_MULTI_QUERY", True)


def _mock_ollama(content: str):
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: {"message": {"content": content}}[key]
    return {"message": {"content": content}}


def test_parses_well_formatted_output():
    with patch("src.query_expander.ollama.chat", return_value=_mock_ollama("Q1\nQ2\nQ3")):
        result = query_expander.expand_query("original question")
    assert result[0] == "original question"
    assert "Q1" in result
    assert "Q2" in result


def test_fallback_on_empty_output():
    with patch("src.query_expander.ollama.chat", return_value=_mock_ollama("")):
        result = query_expander.expand_query("original question")
    assert result == ["original question"]


def test_fallback_on_exception():
    with patch("src.query_expander.ollama.chat", side_effect=Exception("connection refused")):
        result = query_expander.expand_query("original question")
    assert result == ["original question"]


def test_filters_empty_lines():
    with patch("src.query_expander.ollama.chat", return_value=_mock_ollama("Q1\n\n\nQ2\n")):
        result = query_expander.expand_query("original")
    assert "" not in result
    assert "Q1" in result and "Q2" in result


def test_disabled_config(monkeypatch):
    monkeypatch.setattr(config, "ENABLE_MULTI_QUERY", False)
    with patch("src.query_expander.ollama.chat") as mock_chat:
        result = query_expander.expand_query("original question")
    mock_chat.assert_not_called()
    assert result == ["original question"]
