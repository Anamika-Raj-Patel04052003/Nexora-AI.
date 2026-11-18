"""
NEXORA AI - Long-Term Memory Service

Persistent cross-conversation memory storing:
- User profile & identity (Name, role, background)
- User preferences (Tone, language, coding style, formatting)
- Active project details & technical stack preferences

Stored in SQLite (nexora.db) table 'user_memories'.
Automatically extracted from conversation turns and injected into AI system prompts.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from langchain_core.messages import HumanMessage
from app.services.llm_service import llm

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[2] / "nexora.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_long_term_memory_db() -> None:
    """
    Ensure the user_memories table exists in nexora.db.
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                value TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'preference',
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.commit()


def save_memory(key: str, value: str, category: str = "preference") -> None:
    """
    Store or update a persistent memory item.
    """
    init_long_term_memory_db()
    now = _now()
    clean_key = key.strip().lower().replace(" ", "_")
    clean_value = value.strip()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO user_memories (key, value, category, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                category = excluded.category,
                updated_at = excluded.updated_at
            """,
            (clean_key, clean_value, category, now)
        )
        conn.commit()
    logger.info("Saved long-term memory [%s]: %s", clean_key, clean_value[:50])


def get_all_memories() -> list[dict]:
    """
    Retrieve all stored long-term memories.
    """
    init_long_term_memory_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT key, value, category, updated_at FROM user_memories ORDER BY category, key"
        ).fetchall()
        return [dict(r) for r in rows]


def delete_memory(key: str) -> bool:
    """
    Delete a specific stored memory by key.
    """
    init_long_term_memory_db()
    clean_key = key.strip().lower().replace(" ", "_")
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.execute("DELETE FROM user_memories WHERE key = ?", (clean_key,))
        conn.commit()
        return cur.rowcount > 0


def format_long_term_memories() -> str:
    """
    Formats all persistent memories into a compact context block for LLM prompts.
    """
    memories = get_all_memories()
    if not memories:
        return ""

    lines = ["=== USER PROFILE & LONG-TERM MEMORY ==="]
    for m in memories:
        category = m["category"].capitalize()
        key = m["key"].replace("_", " ").capitalize()
        val = m["value"]
        lines.append(f"- [{category}] {key}: {val}")
    lines.append("=== END OF LONG-TERM MEMORY ===")

    return "\n".join(lines)


def extract_and_learn_facts(user_input: str) -> None:
    """
    Uses LLM to detect if the user shared personal facts, preferences, or instructions,
    and automatically saves them to long-term memory.
    Runs silently without disrupting chat flow.
    """
    if len(user_input.strip()) < 10:
        return

    extraction_prompt = f"""
Analyze the user statement below. Determine if the user is sharing an enduring personal fact, preference, technical skill, or specific instruction about themselves.
Examples:
- "Mera naam Anamika hai" -> key: "user_name", value: "Anamika", category: "identity"
- "I prefer answers in Hinglish" -> key: "preferred_language", value: "Hinglish", category: "preference"
- "I am building a React and Python project" -> key: "current_project", value: "React and Python", category: "project"

If NO lasting fact or preference is shared (e.g. general questions like "What is Docker?"), return strictly: {{"facts": []}}

User statement:
"{user_input}"

Return strictly valid JSON:
{{"facts": [{{"key": "string", "value": "string", "category": "identity|preference|skill|project"}}]}}
"""

    try:
        response = llm.invoke([HumanMessage(content=extraction_prompt)])
        cleaned = response.content.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(cleaned)
        facts = data.get("facts", [])

        for fact in facts:
            k = fact.get("key")
            v = fact.get("value")
            c = fact.get("category", "preference")
            if k and v:
                save_memory(key=k, value=v, category=c)

    except Exception as e:
        logger.debug("Memory fact extraction skipped: %s", e)