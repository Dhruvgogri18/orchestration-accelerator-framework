"""SQLite-backed session store.

Same public functions as the earlier in-memory version
(create_session, get_session, append_message) plus two new ones
(set_latest_plan, get_latest_plan) needed for the /approve and
/plan/render guard. Callers outside this file (main.py) don't need to
know storage changed -- that was the whole point of keeping this
module's interface stable.

Why SQLite over the in-memory dict: sessions now survive an API
restart, and multiple people/processes hitting the same file see the
same data. Why SQLite over a bigger database: zero setup, one file,
plenty for local dev and a single-instance deployment. If this needs
to run across multiple API instances behind a load balancer later,
this file is still the only place that would need to change (e.g. to
Postgres) -- same reasoning as the original docstring.

Schema:
    sessions(session_id TEXT PRIMARY KEY, latest_plan_json TEXT NULL,
              locked_spec_json TEXT NULL, created_at TEXT)
    messages(id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT,
             role TEXT, content TEXT, created_at TEXT)
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

from orchestrator.extraction.models import Plan


def _db_path() -> str:
    """Read fresh on every call (not cached at import time) so tests
    can point this at a temp file via env var without needing to
    control import order -- see tests/conftest.py.
    """
    return os.environ.get("ORCHESTRATOR_DB_PATH", "data/sessions.db")


@contextmanager
def _connection():
    path = _db_path()
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                latest_plan_json TEXT,
                locked_spec_json TEXT,
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )"""
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


@dataclass
class Session:
    """In-memory representation of one row + its messages, handed back
    to callers. Mutating this object does NOT persist anything -- use
    append_message / set_latest_plan to write changes back to SQLite.
    """
    session_id: str
    history: list[dict] = field(default_factory=list)
    latest_plan: Plan | None = None
    locked_spec: dict | None = None


def create_session() -> Session:
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with _connection() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, created_at) VALUES (?, ?)",
            (session_id, now),
        )
    return Session(session_id=session_id)


def get_session(session_id: str) -> Session | None:
    with _connection() as conn:
        row = conn.execute(
            "SELECT latest_plan_json, locked_spec_json FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        latest_plan_json, locked_spec_json = row

        message_rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()

    history = [{"role": role, "content": content} for role, content in message_rows]
    latest_plan = Plan.model_validate_json(latest_plan_json) if latest_plan_json else None
    locked_spec = json.loads(locked_spec_json) if locked_spec_json else None

    return Session(
        session_id=session_id,
        history=history,
        latest_plan=latest_plan,
        locked_spec=locked_spec,
    )


def append_message(session_id: str, role: str, content: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if exists is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )


def set_latest_plan(session_id: str, plan: Plan | None) -> None:
    """Persist (or clear) the session's approvable plan. Called from
    main.py after every /message call -- set when status == 'ready',
    cleared otherwise (so a follow-up message that reopens the
    conversation can't leave a stale plan sitting around as
    "approvable").
    """
    plan_json = plan.model_dump_json() if plan is not None else None
    with _connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if exists is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        conn.execute(
            "UPDATE sessions SET latest_plan_json = ? WHERE session_id = ?",
            (plan_json, session_id),
        )


def set_locked_spec(session_id: str, spec: dict) -> None:
    """Persist the result of /approve -- the resolved plan (overrides
    applied) plus whatever placeholders are still unresolved. NOT the
    full workflow_spec.json yet (that's Stage 3.5 enrichment, still
    unbuilt) -- this is an intermediate "approved" checkpoint.
    """
    spec_json = json.dumps(spec)
    with _connection() as conn:
        exists = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if exists is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        conn.execute(
            "UPDATE sessions SET locked_spec_json = ? WHERE session_id = ?",
            (spec_json, session_id),
        )