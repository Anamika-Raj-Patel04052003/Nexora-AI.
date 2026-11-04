import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

# The database file lives in the backend folder: backend/nexora.db
DB_PATH = Path(__file__).resolve().parent.parent / "nexora.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL
                    REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id);
            """
        )


def create_conversation() -> dict:
    now = _now()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at) "
            "VALUES (?, ?, ?)",
            ("New chat", now, now),
        )
        return {
            "id": cur.lastrowid,
            "title": "New chat",
            "created_at": now,
            "updated_at": now,
        }


def list_conversations() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "ORDER BY updated_at DESC, id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_conversation(conversation_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations "
            "WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        return dict(row) if row else None


def get_messages(conversation_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, role, content, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def add_message(conversation_id: int, role: str, content: str) -> None:
    now = _now()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, now),
        )
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (now, conversation_id),
        )


def set_title(conversation_id: int, title: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            (title, conversation_id),
        )


def delete_conversation(conversation_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM conversations WHERE id = ?", (conversation_id,)
        )