"""Shared pytest fixtures.

isolate_db points ORCHESTRATOR_DB_PATH at a fresh temp file for every
single test. This alone is NOT enough: orchestrator.api.config.settings
.get_api_settings() is @lru_cache'd, and orchestrator.api.main calls it
once at import time (before any test's monkeypatch.setenv runs) --
so without clearing that cache, every test after the first would
silently keep reusing the FIRST cached settings object (and its db
path), regardless of the env var this fixture sets.

Part B fixtures:
  gateway_env          - dummy gateway settings; never touches the real .env/gateway
  fake_embedder        - deterministic, similarity-preserving embedder (1024-dim,
                         matching Titan Text Embeddings V2's default)
  use_fake_embeddings  - routes gateway_client.embed_texts() to the fake embedder
  pg_client            - live pgvector client (skips if Postgres is unreachable)
  pg_collection        - unique throwaway collection on pg_client (cleaned up)
"""
from __future__ import annotations

import hashlib
import math
import re
import uuid

import pytest

from orchestrator.api.config.settings import get_api_settings
from orchestrator.config import get_settings

FAKE_EMBEDDING_DIM = 1024  # Titan Text Embeddings V2 default output size
DEFAULT_PG_DSN = "postgresql://oaf:oaf_secret@127.0.0.1:5433/oaf_vectors"


def clean_settings_env(monkeypatch):
    """Remove every variable Settings/APISettings read from os.environ.

    gateway_client calls load_dotenv() at import, which copies the real
    .env into os.environ for the whole test process; without this, tests
    would silently pick up real values."""
    from orchestrator.config import Settings

    names = {"RETRIEVAL_ENABLED", "ORCHESTRATOR_DB_PATH", "CHROMA_PERSIST_DIR", "CHROMA_COLLECTION_NAME"}
    for field in Settings.model_fields.values():
        if field.alias:
            names.add(field.alias)
        va = field.validation_alias
        names.update(getattr(va, "choices", []) or ([va] if isinstance(va, str) else []))
    for name in names - {"ORCHESTRATOR_DB_PATH"}:
        monkeypatch.delenv(name, raising=False)


def _clear_caches():
    get_api_settings.cache_clear()
    get_settings.cache_clear()
    try:  # gateway_client may not be importable in every environment
        from orchestrator.indexing.gateway_client import get_client

        get_client.cache_clear()
    except Exception:  # pragma: no cover
        pass


@pytest.fixture(autouse=True)
def isolate_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test_sessions.db"
    monkeypatch.setenv("ORCHESTRATOR_DB_PATH", str(db_path))
    _clear_caches()
    yield
    _clear_caches()


@pytest.fixture
def gateway_env(tmp_path, monkeypatch):
    """Minimal, fake gateway configuration. Runs from an empty cwd so the
    real .env is never read."""
    monkeypatch.chdir(tmp_path)
    clean_settings_env(monkeypatch)
    monkeypatch.setenv("GATEWAY_API_KEY", "test-key")
    monkeypatch.setenv("GATEWAY_BASE_URL", "http://gateway.test/v1")
    monkeypatch.setenv("GATEWAY_CHAT_MODEL", "test/chat-model")
    monkeypatch.setenv("GATEWAY_EMBEDDING_MODEL", "bedrock/amazon.titan-embed-text-v2:0")
    _clear_caches()
    yield
    _clear_caches()


def _fake_vector(text: str, dim: int = FAKE_EMBEDDING_DIM) -> list[float]:
    """Bag-of-words hashing embedder: deterministic, unit length, and texts
    that share words end up closer together -- good enough to test ranking
    without calling the gateway."""
    vec = [0.0] * dim
    tokens = re.findall(r"[a-z0-9_]+", text.lower()) or [text]
    for tok in tokens:
        h = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:8], "big")
        vec[h % dim] += 1.0 if (h >> 63) & 1 else -1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


@pytest.fixture
def fake_embedder():
    def embed(texts: list[str]) -> list[list[float]]:
        return [_fake_vector(t) for t in texts]

    embed.dim = FAKE_EMBEDDING_DIM
    return embed


@pytest.fixture
def use_fake_embeddings(monkeypatch, fake_embedder):
    """Replace gateway_client.embed_texts() with the fake embedder.

    Code under test must call it as `gateway_client.embed_texts(...)`
    (module attribute lookup), not `from ... import embed_texts`, for the
    patch to take effect."""
    from orchestrator.indexing import gateway_client

    monkeypatch.setattr(gateway_client, "embed_texts", fake_embedder)
    return fake_embedder


def _can_connect_pg(dsn: str) -> bool:
    try:
        import psycopg

        with psycopg.connect(dsn, connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.fixture
def pg_dsn(monkeypatch):
    """Resolve PGVECTOR_DSN and skip the test if Postgres is unreachable.

    Start the DB with: docker compose -f docker/docker-compose.yaml up -d
    """
    dsn = (
        __import__("os").environ.get("PGVECTOR_DSN")
        or __import__("os").environ.get("DATABASE_URL")
        or DEFAULT_PG_DSN
    )
    if not _can_connect_pg(dsn):
        pytest.skip(
            f"Postgres/pgvector not reachable at {dsn}. "
            "Run: docker compose -f docker/docker-compose.yaml up -d"
        )
    monkeypatch.setenv("PGVECTOR_DSN", dsn)
    _clear_caches()
    return dsn


@pytest.fixture
def pg_client(pg_dsn):
    from orchestrator.indexing import vectorstore as vs

    return vs.get_client()


@pytest.fixture
def pg_collection(pg_client):
    """Unique throwaway collection; dropped after the test."""
    from orchestrator.indexing import vectorstore as vs

    name = f"test_{uuid.uuid4().hex[:12]}"
    col = vs.get_collection(
        client=pg_client,
        name=name,
        embedding_model="fake-embed",
        embedding_dim=FAKE_EMBEDDING_DIM,
        create=True,
    )
    yield col
    vs.drop_collection(pg_client, name)


# Backwards-compatible aliases used by older test names
@pytest.fixture
def tmp_chroma_dir(tmp_path):
    """Legacy name: no filesystem store under pgvector; returns a temp path only."""
    path = tmp_path / "pgvector_placeholder"
    path.mkdir()
    return path


@pytest.fixture
def chroma_collection(pg_collection):
    """Legacy name: returns a pgvector collection."""
    return pg_collection
