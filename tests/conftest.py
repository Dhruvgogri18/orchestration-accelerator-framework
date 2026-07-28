"""Shared pytest fixtures.

isolate_db points ORCHESTRATOR_DB_PATH at a fresh temp file for every
single test, so tests never read/write the real dev database and never
leak state between tests. session_store.py reads this env var fresh on
every call (see _db_path()), so setting it per-test is enough -- no
import-order tricks needed.
"""
import pytest


@pytest.fixture(autouse=True)
def isolate_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_sessions.db"
    monkeypatch.setenv("ORCHESTRATOR_DB_PATH", str(db_path))
    yield