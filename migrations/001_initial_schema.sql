-- 001_initial_schema.sql
-- Initial schema: notebooks, conversations, messages, sources

CREATE TABLE IF NOT EXISTS notebooks (
    id         TEXT PRIMARY KEY,
    title      TEXT NOT NULL DEFAULT 'Untitled',
    emoji      TEXT NOT NULL DEFAULT '📓',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    title       TEXT NOT NULL DEFAULT 'New conversation',
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK(role IN ('user','ai')),
    content         TEXT NOT NULL,
    sources_json    TEXT,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources (
    id          TEXT PRIMARY KEY,
    notebook_id TEXT NOT NULL REFERENCES notebooks(id) ON DELETE CASCADE,
    filename    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'processing',
    created_at  TEXT NOT NULL
);
