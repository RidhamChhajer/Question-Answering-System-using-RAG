CREATE TABLE IF NOT EXISTS feedback (
    id         TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    rating     TEXT NOT NULL CHECK(rating IN ('up','down')),
    created_at TEXT NOT NULL
);
