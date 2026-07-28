"""Shared pytest fixtures.

isolate_db points ORCHESTRATOR_DB_PATH at a fresh temp file for every
single test. This alone is NOT enough: orchestrator.api.config.settings
.get_api_settings() is @lru_cache'd, and orchestrator.api.main calls it
once at import time (before any test's monkeypatch.setenv runs) --
so without clearing that cache, every test after the first would
silently keep reusing the FIRST cached settings object (and its db
path), regardless of the env var this fixture sets. Confirmed this was
happening: running the suite without the cache_clear() below wrote 7
sessions / 14 messages directly into the real dev data/sessions.db.
"""
import pytest

from orchestrator.api.config.settings import get_api_settings


@pytest.fixture(autouse=True)
def isolate_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_sessions.db"
    monkeypatch.setenv("ORCHESTRATOR_DB_PATH", str(db_path))
    get_api_settings.cache_clear()
    yield
    get_api_settings.cache_clear()