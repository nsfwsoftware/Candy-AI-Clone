"""
Check Full Demo at - https://tripleminds.co/white-label/candy-ai-clone/
database.py
SQLite database layer for Candy AI Clone chatbot.
- Manages users, chat history, and intent analytics
- Uses sqlite3 (no external dependency)
- Creates schema automatically on first run
"""

import sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
import time
import json

# -------- Paths --------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "logs" / "chatbot.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# -------- Connection Helper --------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

# -------- Schema Init --------
def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        created_at INTEGER
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        message TEXT,
        reply TEXT,
        intent TEXT,
        confidence REAL,
        latency_ms INTEGER,
        allowed INTEGER,
        meta TEXT,
        ts INTEGER,
        FOREIGN KEY(user_id) REFERENCES users(id)
    );
    """)

    conn.commit()
    conn.close()

# -------- User Ops --------
def ensure_user(user_id: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO users (id, created_at) VALUES (?, ?)", (user_id, int(time.time())))
    conn.commit()
    conn.close()

# -------- Chat Logging --------
def save_chat(
    user_id: Optional[str],
    message: str,
    reply: str,
    intent: Optional[str],
    confidence: Optional[float],
    latency_ms: Optional[int],
    allowed: bool,
    meta: Optional[Dict[str, Any]] = None
):
    conn = get_conn()
    cur = conn.cursor()

    meta_json = json.dumps(meta or {}, ensure_ascii=False)

    cur.execute("""
        INSERT INTO chats (user_id, message, reply, intent, confidence, latency_ms, allowed, meta, ts)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id,
        message,
        reply,
        intent,
        confidence,
        latency_ms,
        int(allowed),
        meta_json,
        int(time.time())
    ))

    conn.commit()
    conn.close()

# -------- Analytics --------
def count_chats_per_intent() -> Dict[str, int]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT intent, COUNT(*) FROM chats GROUP BY intent;")
    rows = cur.fetchall()
    conn.close()
    return {intent or "unknown": count for intent, count in rows}

def avg_latency() -> float:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT AVG(latency_ms) FROM chats WHERE latency_ms IS NOT NULL;")
    result = cur.fetchone()[0]
    conn.close()
    return float(result) if result is not None else 0.0

def recent_chats(limit: int = 10):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT user_id, message, reply, intent, ts FROM chats ORDER BY ts DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows

# -------- Initialize on import --------
init_db()
