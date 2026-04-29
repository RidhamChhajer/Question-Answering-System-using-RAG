"""
database.py — SQLite persistence for the RAG web app.
Tables: notebooks, conversations, messages, sources
"""
import sqlite3, uuid, json, os, random, glob, re
from pathlib import Path
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks.db")
MIGRATIONS_DIR = Path(__file__).parent.parent / "migrations"

EMOJIS = ['\U0001f4d3','\U0001f4d4','\U0001f4d2','\U0001f4d5','\U0001f4d7','\U0001f4d8','\U0001f4d9','\U0001f4da','\U0001f5d2','\U0001f4cb','\U0001f4c4','\U0001f5c3','\U0001f4a1','\U0001f52c','\U0001f393','\u270f']

def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c

def _get_migrations():
    """Scan the migrations/ directory and return sorted (version, filename, path) tuples."""
    pattern = str(MIGRATIONS_DIR / "*.sql")
    files = sorted(glob.glob(pattern))
    migrations = []
    for filepath in files:
        basename = os.path.basename(filepath)
        match = re.match(r"^(\d+)", basename)
        if match:
            version = int(match.group(1))
            migrations.append((version, basename, filepath))
    return migrations

def _ensure_schema_version(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
    """)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(schema_version)").fetchall()}
    if "filename" not in cols:
        conn.execute("ALTER TABLE schema_version ADD COLUMN filename TEXT NOT NULL DEFAULT ''")
    if "applied_at" not in cols:
        conn.execute("ALTER TABLE schema_version ADD COLUMN applied_at TEXT NOT NULL DEFAULT ''")


def _get_schema_version(conn):
    _ensure_schema_version(conn)
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return row[0] or 0

def _apply_migration(conn, version: int, sql_path: Path):
    _ensure_schema_version(conn)
    sql = sql_path.read_text(encoding="utf-8")
    conn.executescript(sql)
    conn.execute(
        "INSERT INTO schema_version (version, filename, applied_at) VALUES (?, ?, datetime('now'))",
        (version, sql_path.name)
    )

def init_db():
    """Apply all pending migrations in order."""
    with _conn() as conn:
        # Create the schema_version tracking table
        conn.execute("PRAGMA foreign_keys = ON")
        current = _get_schema_version(conn)

        # Determine which migrations have already been applied
        migration_files = sorted(MIGRATIONS_DIR.glob("*.sql"))

        # Apply each pending migration in order
        for path in migration_files:
            version = int(path.stem.split("_")[0])  # e.g. 001 -> 1
            if version > current:
                _apply_migration(conn, version, path)


def _now(): return datetime.utcnow().isoformat()
def _uid(): return str(uuid.uuid4())

# ── Notebooks ──────────────────────────────────────────────────────────────────
def create_notebook(title="Untitled", emoji=None):
    nid, now = _uid(), _now()
    emoji = emoji or random.choice(EMOJIS)
    with _conn() as c:
        c.execute("INSERT INTO notebooks VALUES (?,?,?,?,?)", (nid, title, emoji, now, now))
    return get_notebook(nid)

def get_notebook(nid):
    with _conn() as c:
        r = c.execute("SELECT * FROM notebooks WHERE id=?", (nid,)).fetchone()
    return dict(r) if r else None

def list_notebooks():
    with _conn() as c:
        rows = c.execute("SELECT * FROM notebooks ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]

def update_notebook(nid, title=None, emoji=None):
    parts, vals = [], []
    if title is not None: parts.append("title=?"); vals.append(title)
    if emoji is not None: parts.append("emoji=?"); vals.append(emoji)
    parts.append("updated_at=?"); vals.append(_now())
    vals.append(nid)
    with _conn() as c:
        c.execute(f"UPDATE notebooks SET {','.join(parts)} WHERE id=?", vals)
    return get_notebook(nid)

def touch_notebook(nid):
    with _conn() as c:
        c.execute("UPDATE notebooks SET updated_at=? WHERE id=?", (_now(), nid))

def delete_notebook(nid):
    with _conn() as c: c.execute("DELETE FROM notebooks WHERE id=?", (nid,))

# ── Conversations ──────────────────────────────────────────────────────────────
def create_conversation(notebook_id, title="New conversation"):
    cid, now = _uid(), _now()
    with _conn() as c:
        c.execute("INSERT INTO conversations VALUES (?,?,?,?)", (cid, notebook_id, title, now))
        c.execute("UPDATE notebooks SET updated_at=? WHERE id=?", (now, notebook_id))
    return get_conversation(cid)

def get_conversation(cid):
    with _conn() as c:
        r = c.execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
    return dict(r) if r else None

def list_conversations(notebook_id):
    with _conn() as c:
        rows = c.execute("SELECT * FROM conversations WHERE notebook_id=? ORDER BY created_at DESC", (notebook_id,)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["message_count"] = c.execute(
                "SELECT COUNT(*) FROM messages WHERE conversation_id=?", (d["id"],)
            ).fetchone()[0]
            result.append(d)
    return result

def rename_conversation(cid, title):
    with _conn() as c:
        c.execute("UPDATE conversations SET title=? WHERE id=?", (title, cid))
    return get_conversation(cid)

def auto_title_conversation(cid: str, title: str):
    """Set the title of a conversation. Called after first answer is generated."""
    title = title.strip()[:100]   # clean and cap at 100 chars
    return rename_conversation(cid, title)

def delete_conversation(cid: str):
    """
    Delete a conversation by ID.
    SQLite CASCADE automatically deletes all messages and feedback for this conversation.
    """
    with _conn() as c:
        c.execute("DELETE FROM conversations WHERE id=?", (cid,))

# ── Messages ───────────────────────────────────────────────────────────────────
def add_message(conversation_id, role, content, sources=None):
    mid, now = _uid(), _now()
    sj = json.dumps(sources) if sources else None
    with _conn() as c:
        c.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)", (mid, conversation_id, role, content, sj, now))
    return get_message(mid)

def get_message(mid):
    with _conn() as c:
        r = c.execute("SELECT * FROM messages WHERE id=?", (mid,)).fetchone()
    if not r: return None
    d = dict(r); d["sources"] = json.loads(d["sources_json"]) if d["sources_json"] else []
    return d

def list_messages(conversation_id):
    with _conn() as c:
        rows = c.execute("SELECT * FROM messages WHERE conversation_id=? ORDER BY created_at ASC", (conversation_id,)).fetchall()
    result = []
    for r in rows:
        d = dict(r); d["sources"] = json.loads(d["sources_json"]) if d["sources_json"] else []
        result.append(d)
    return result

def count_messages(conversation_id):
    with _conn() as c:
        return c.execute("SELECT COUNT(*) FROM messages WHERE conversation_id=?", (conversation_id,)).fetchone()[0]

# ── Sources ────────────────────────────────────────────────────────────────────
def add_source(notebook_id: str, filename: str, status: str = "processing", file_size: int = 0):
    sid, now = _uid(), _now()
    with _conn() as c:
        c.execute(
            "INSERT INTO sources (id, notebook_id, filename, status, file_size, created_at) VALUES (?,?,?,?,?,?)",
            (sid, notebook_id, filename, status, file_size, now)
        )
    return get_source(sid)

def get_source(sid):
    with _conn() as c:
        r = c.execute("SELECT * FROM sources WHERE id=?", (sid,)).fetchone()
    return dict(r) if r else None

def update_source_status(sid, status):
    with _conn() as c:
        c.execute("UPDATE sources SET status=? WHERE id=?", (status, sid))

def update_source_metadata(sid: str, page_count: int):
    """Update the page count for a source after indexing."""
    with _conn() as c:
        c.execute("UPDATE sources SET page_count=? WHERE id=?", (page_count, sid))

def list_sources(notebook_id):
    with _conn() as c:
        rows = c.execute("SELECT * FROM sources WHERE notebook_id=? ORDER BY created_at ASC", (notebook_id,)).fetchall()
    return [dict(r) for r in rows]

def delete_source(sid: str):
    """Delete a source record by ID."""
    with _conn() as c:
        c.execute("DELETE FROM sources WHERE id=?", (sid,))

# ── Feedback ──────────────────────────────────────────────

def add_feedback(message_id: str, rating: str) -> dict:
    """Create or update feedback for a message."""
    import uuid, datetime
    existing = get_feedback(message_id)
    with _conn() as c:
        if existing:
            c.execute("UPDATE feedback SET rating=? WHERE message_id=?", (rating, message_id))
            return {**existing, "rating": rating}
        else:
            fid = str(uuid.uuid4())
            now = datetime.datetime.utcnow().isoformat()
            c.execute(
                "INSERT INTO feedback (id, message_id, rating, created_at) VALUES (?,?,?,?)",
                (fid, message_id, rating, now)
            )
            return {"id": fid, "message_id": message_id, "rating": rating, "created_at": now}

def get_feedback(message_id: str) -> dict | None:
    """Return feedback for a message, or None."""
    with _conn() as c:
        row = c.execute("SELECT * FROM feedback WHERE message_id=?", (message_id,)).fetchone()
        return dict(row) if row else None

def get_feedbacks_for_conversation(conversation_id: str) -> dict:
    """Return {message_id: rating} for all messages in a conversation."""
    with _conn() as c:
        rows = c.execute(
            "SELECT f.message_id, f.rating FROM feedback f "
            "JOIN messages m ON f.message_id=m.id "
            "WHERE m.conversation_id=?",
            (conversation_id,)
        ).fetchall()
        return {row["message_id"]: row["rating"] for row in rows}
