"""Unit tests for api/database.py -- uses a temporary SQLite DB."""
import pytest
from api import database


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Each test gets a fresh isolated database."""
    db_path = str(tmp_path / "test.db")
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db()


def test_create_and_get_notebook():
    nb = database.create_notebook("My Notebook")
    fetched = database.get_notebook(nb["id"])
    assert fetched["title"] == "My Notebook"
    assert fetched["id"] == nb["id"]


def test_list_notebooks_empty():
    assert database.list_notebooks() == []


def test_list_notebooks():
    database.create_notebook("NB1")
    database.create_notebook("NB2")
    database.create_notebook("NB3")
    nbs = database.list_notebooks()
    assert len(nbs) == 3


def test_update_notebook():
    nb = database.create_notebook("Old Title")
    database.update_notebook(nb["id"], title="New Title")
    fetched = database.get_notebook(nb["id"])
    assert fetched["title"] == "New Title"


def test_get_nonexistent_notebook():
    assert database.get_notebook("nonexistent-id") is None


def test_create_conversation():
    nb = database.create_notebook("NB")
    conv = database.create_conversation(nb["id"])
    assert conv["notebook_id"] == nb["id"]
    assert conv["id"] is not None


def test_list_conversations():
    nb = database.create_notebook("NB")
    database.create_conversation(nb["id"])
    database.create_conversation(nb["id"])
    convs = database.list_conversations(nb["id"])
    assert len(convs) == 2


def test_add_and_list_messages():
    nb = database.create_notebook("NB")
    conv = database.create_conversation(nb["id"])
    database.add_message(conv["id"], "user", "Hello?")
    database.add_message(conv["id"], "ai", "Hi there!")
    msgs = database.list_messages(conv["id"])
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "ai"


def test_add_source():
    nb = database.create_notebook("NB")
    src = database.add_source(nb["id"], "file.pdf")
    srcs = database.list_sources(nb["id"])
    assert len(srcs) == 1
    assert srcs[0]["filename"] == "file.pdf"


def test_cascade_delete_notebook():
    """Deleting a notebook cascades to conversations and messages."""
    nb = database.create_notebook("NB")
    conv = database.create_conversation(nb["id"])
    database.add_message(conv["id"], "user", "test")
    database.delete_notebook(nb["id"])
    assert database.get_notebook(nb["id"]) is None
    assert database.list_conversations(nb["id"]) == []
