import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_FILE = Path("orchestration_mvp.db")


def get_connection():
    connection = sqlite3.connect(DB_FILE, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deploy_status TEXT,
                last_state_machine_arn TEXT,
                run_id TEXT,
                stage TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                artifact_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
            )
            """
        )

        chat_cols = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(chats)").fetchall()
        }
        if "deploy_status" not in chat_cols:
            connection.execute("ALTER TABLE chats ADD COLUMN deploy_status TEXT")
        if "last_state_machine_arn" not in chat_cols:
            connection.execute(
                "ALTER TABLE chats ADD COLUMN last_state_machine_arn TEXT"
            )
        if "run_id" not in chat_cols:
            connection.execute("ALTER TABLE chats ADD COLUMN run_id TEXT")
        if "stage" not in chat_cols:
            connection.execute("ALTER TABLE chats ADD COLUMN stage TEXT")

        message_cols = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(messages)").fetchall()
        }
        if "artifact_json" not in message_cols:
            connection.execute(
                "ALTER TABLE messages ADD COLUMN artifact_json TEXT"
            )


def create_chat(title="New workflow chat"):
    now = datetime.now().isoformat()
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO chats (title, created_at, updated_at, deploy_status)
            VALUES (?, ?, ?, ?)
            """,
            (title, now, now, "draft"),
        )
        return cursor.lastrowid


def get_chats():
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, title, created_at, updated_at,
                   deploy_status, last_state_machine_arn,
                   run_id, stage
            FROM chats
            ORDER BY updated_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def get_chat(chat_id):
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, title, created_at, updated_at,
                   deploy_status, last_state_machine_arn,
                   run_id, stage
            FROM chats
            WHERE id = ?
            """,
            (chat_id,),
        ).fetchone()
    return dict(row) if row else None


def update_chat_title(chat_id, title):
    clean_title = title.strip() or "Untitled workflow"
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE chats
            SET title = ?, updated_at = ?
            WHERE id = ?
            """,
            (clean_title, datetime.now().isoformat(), chat_id),
        )


def update_chat_run_state(
    chat_id,
    run_id=None,
    stage=None,
    deploy_status=None,
    state_machine_arn=None,
):
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE chats
            SET run_id = COALESCE(?, run_id),
                stage = COALESCE(?, stage),
                deploy_status = COALESCE(?, deploy_status),
                last_state_machine_arn = COALESCE(?, last_state_machine_arn),
                updated_at = ?
            WHERE id = ?
            """,
            (
                run_id,
                stage,
                deploy_status,
                state_machine_arn,
                datetime.now().isoformat(),
                chat_id,
            ),
        )


def delete_chat(chat_id):
    with get_connection() as connection:
        connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,))


def add_message(chat_id, role, content, artifact=None):
    now = datetime.now().isoformat()
    artifact_json = json.dumps(artifact) if artifact is not None else None
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO messages (chat_id, role, content, artifact_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (chat_id, role, content, artifact_json, now),
        )
        message_id = cursor.lastrowid
        connection.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ?",
            (now, chat_id),
        )
    return message_id


def update_message_artifact(message_id, artifact):
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE messages
            SET artifact_json = ?
            WHERE id = ?
            """,
            (json.dumps(artifact), message_id),
        )


def update_message_content(message_id, content):
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE messages
            SET content = ?
            WHERE id = ?
            """,
            (content, message_id),
        )


def get_messages(chat_id):
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, chat_id, role, content, artifact_json, created_at
            FROM messages
            WHERE chat_id = ?
            ORDER BY id ASC
            """,
            (chat_id,),
        ).fetchall()

    messages = []
    for row in rows:
        message = dict(row)
        raw = message.pop("artifact_json", None)
        if raw:
            try:
                message["artifact"] = json.loads(raw)
            except json.JSONDecodeError:
                message["artifact"] = None
        else:
            message["artifact"] = None
        messages.append(message)
    return messages