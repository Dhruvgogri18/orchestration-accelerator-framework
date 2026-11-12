"""Postgres + pgvector vector store for repository chunks.

Public surface:
  IndexMismatchError, Hit,
  get_client, get_collection, drop_collection,
  upsert_chunks, delete_ids, delete_repo,
  existing_hashes, build_where, get_by_metadata, query

Requires:
  pip install "psycopg[binary]>=3.1"
  Docker: docker/docker-compose.yaml (pgvector on host port 5433)
  env: PGVECTOR_DSN (or DATABASE_URL); optional VECTOR_COLLECTION_NAME
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from orchestrator.ingestion.models import SCHEMA_VERSION, Chunk

UPSERT_BATCH_SIZE = 100
GET_PAGE_SIZE = 1000

# Default DSN matches docker/docker-compose.yaml (port 5433 on host)
_DEFAULT_DSN = "postgresql://oaf:oaf_secret@127.0.0.1:5433/oaf_vectors"


class IndexMismatchError(RuntimeError):
    """Collection was built with a different model or schema."""


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None = None


def _dsn() -> str:
    try:
        s = _settings()
        if getattr(s, "pgvector_dsn", None):
            return str(s.pgvector_dsn).strip()
    except Exception:  # noqa: BLE001 - settings may be incomplete in tests
        pass
    return (
        os.environ.get("PGVECTOR_DSN")
        or os.environ.get("DATABASE_URL")
        or _DEFAULT_DSN
    ).strip()


def _settings():
    from orchestrator.config import get_settings

    return get_settings()


def _default_collection_name() -> str:
    try:
        return _settings().vector_collection_name
    except Exception:  # noqa: BLE001
        return os.environ.get("VECTOR_COLLECTION_NAME") or os.environ.get(
            "CHROMA_COLLECTION_NAME", "repo_context"
        )


# ---------------------------------------------------------------------------
# Connection / collection handles
# ---------------------------------------------------------------------------


class PgClient:
    """Thin client handle — holds DSN only."""

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def _connect(self):
        import psycopg

        return psycopg.connect(self.dsn, autocommit=False)


class PgCollection:
    """One named collection (logical partition) + embedding config."""

    def __init__(
        self,
        client: PgClient,
        name: str,
        *,
        schema_version: str | int,
        embedding_model: str,
        embedding_dim: int,
    ) -> None:
        self.client = client
        self.name = name
        self.metadata = {
            "schema_version": str(schema_version),
            "embedding_model": embedding_model,
            "embedding_dim": str(embedding_dim),
        }
        self.embedding_dim = embedding_dim

    def _connect(self):
        return self.client._connect()

    def count(self) -> int:
        """Number of chunks in this collection."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM repo_chunks WHERE collection_name = %s",
                    (self.name,),
                )
                return int(cur.fetchone()[0])

    def get(self, ids: list[str] | None = None) -> dict[str, list]:
        """Return chunks in a Chroma-like shape: ids / documents / metadatas.

        If ``ids`` is given, only those rows (missing ids are omitted).
        Order follows ``ids`` when provided, otherwise id ascending.
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                if ids is not None:
                    if not ids:
                        return {"ids": [], "documents": [], "metadatas": []}
                    cur.execute(
                        """
                        SELECT id, document, metadata FROM repo_chunks
                        WHERE collection_name = %s AND id = ANY(%s)
                        """,
                        (self.name, list(ids)),
                    )
                    by_id = {
                        row[0]: (row[1], row[2]) for row in cur.fetchall()
                    }
                    ordered_ids = [i for i in ids if i in by_id]
                    docs, metas = [], []
                    for i in ordered_ids:
                        doc, meta = by_id[i]
                        if isinstance(meta, str):
                            meta = json.loads(meta)
                        docs.append(doc or "")
                        metas.append(meta or {})
                    return {"ids": ordered_ids, "documents": docs, "metadatas": metas}

                cur.execute(
                    """
                    SELECT id, document, metadata FROM repo_chunks
                    WHERE collection_name = %s
                    ORDER BY id
                    """,
                    (self.name,),
                )
                rows = cur.fetchall()
        out_ids, docs, metas = [], [], []
        for i, doc, meta in rows:
            if isinstance(meta, str):
                meta = json.loads(meta)
            out_ids.append(i)
            docs.append(doc or "")
            metas.append(meta or {})
        return {"ids": out_ids, "documents": docs, "metadatas": metas}


def get_client(persist_dir: str | Path | None = None) -> PgClient:
    """Return a pgvector client.

    ``persist_dir`` is accepted for call-site compatibility with the old
    Chroma API and is ignored — storage is entirely in Postgres.
    """
    _ = persist_dir
    dsn = _dsn()
    _ensure_schema(dsn)
    return PgClient(dsn)


def _ensure_schema(dsn: str) -> None:
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS repo_collections (
                    name            TEXT PRIMARY KEY,
                    schema_version  TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dim   INTEGER NOT NULL,
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS repo_chunks (
                    id              TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL
                        REFERENCES repo_collections(name) ON DELETE CASCADE,
                    repo_id         TEXT NOT NULL DEFAULT '',
                    content_hash    TEXT NOT NULL DEFAULT '',
                    document        TEXT NOT NULL DEFAULT '',
                    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
                    embedding       vector,
                    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            # If an older init.sql created vector(1024), widen to unbounded.
            cur.execute(
                """
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'repo_chunks' AND column_name = 'embedding'
                    ) THEN
                        BEGIN
                            ALTER TABLE repo_chunks ALTER COLUMN embedding TYPE vector;
                        EXCEPTION WHEN others THEN
                            NULL;
                        END;
                    END IF;
                END $$;
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_repo_chunks_collection
                    ON repo_chunks (collection_name)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_repo_chunks_repo
                    ON repo_chunks (collection_name, repo_id)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_repo_chunks_metadata
                    ON repo_chunks USING GIN (metadata)
                """
            )


def drop_collection(client: PgClient, name: str) -> None:
    """Remove a collection and all of its chunks (for tests / rebuilds)."""
    with client._connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM repo_chunks WHERE collection_name = %s", (name,))
            cur.execute("DELETE FROM repo_collections WHERE name = %s", (name,))
        conn.commit()


def _check_compatible(collection: PgCollection, expected: dict[str, Any]) -> None:
    stored = collection.metadata or {}
    for key in ("schema_version", "embedding_model", "embedding_dim"):
        if key not in expected or key not in stored:
            continue
        left = str(stored[key])
        right = str(expected[key])
        if left != right:
            raise IndexMismatchError(
                f"Collection '{collection.name}' was built with {key}={left!r} but this run "
                f"uses {right!r}. Re-run `orchestrator index --rebuild` or use a new "
                "VECTOR_COLLECTION_NAME."
            )


def get_collection(
    *,
    client: PgClient | None = None,
    name: str | None = None,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    create: bool = True,
) -> PgCollection | None:
    client = client or get_client()
    name = name or _default_collection_name()
    expected: dict[str, Any] = {"schema_version": str(SCHEMA_VERSION)}
    if embedding_model:
        expected["embedding_model"] = embedding_model
    if embedding_dim:
        expected["embedding_dim"] = embedding_dim

    import psycopg

    with client._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT schema_version, embedding_model, embedding_dim "
                "FROM repo_collections WHERE name = %s",
                (name,),
            )
            row = cur.fetchone()

            if row is None:
                if not create:
                    return None
                if not embedding_model:
                    raise RuntimeError(
                        "No embedding model to record on the collection: "
                        "set GATEWAY_EMBEDDING_MODEL."
                    )
                dim = int(embedding_dim or 1024)
                cur.execute(
                    """
                    INSERT INTO repo_collections
                        (name, schema_version, embedding_model, embedding_dim)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (name) DO NOTHING
                    """,
                    (name, str(SCHEMA_VERSION), embedding_model, dim),
                )
                conn.commit()
                col = PgCollection(
                    client,
                    name,
                    schema_version=SCHEMA_VERSION,
                    embedding_model=embedding_model,
                    embedding_dim=dim,
                )
                _check_compatible(col, expected)
                return col

            schema_version, model, dim = row[0], row[1], int(row[2])
            col = PgCollection(
                client,
                name,
                schema_version=schema_version,
                embedding_model=model,
                embedding_dim=dim,
            )
            if embedding_model or embedding_dim is not None:
                _check_compatible(col, expected)
            return col


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def upsert_chunks(
    collection: PgCollection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    *,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    if not chunks:
        return

    dim = collection.embedding_dim
    for emb in embeddings:
        if len(emb) != dim:
            raise ValueError(
                f"Embedding dim {len(emb)} != collection dim {dim}. "
                "Rebuild index with matching GATEWAY_EMBEDDING_MODEL / EMBEDDING_DIMENSIONS."
            )

    with collection._connect() as conn:
        with conn.cursor() as cur:
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start : start + batch_size]
                emb_batch = embeddings[start : start + batch_size]
                ids = [c.id for c in batch]
                cur.execute(
                    "DELETE FROM repo_chunks WHERE id = ANY(%s) AND collection_name = %s",
                    (ids, collection.name),
                )
                for c, emb in zip(batch, emb_batch, strict=True):
                    meta = c.stored_metadata()
                    repo_id = str(meta.get("repo_id") or "")
                    content_hash = str(meta.get("content_hash") or "")
                    cur.execute(
                        """
                        INSERT INTO repo_chunks (
                            id, collection_name, repo_id, content_hash,
                            document, metadata, embedding, updated_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s::jsonb, %s::vector, NOW()
                        )
                        ON CONFLICT (id) DO UPDATE SET
                            collection_name = EXCLUDED.collection_name,
                            repo_id = EXCLUDED.repo_id,
                            content_hash = EXCLUDED.content_hash,
                            document = EXCLUDED.document,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding,
                            updated_at = NOW()
                        """,
                        (
                            c.id,
                            collection.name,
                            repo_id,
                            content_hash,
                            c.text,
                            json.dumps(meta),
                            _vec_literal(emb),
                        ),
                    )
        conn.commit()


def delete_ids(
    collection: PgCollection, ids: list[str], *, batch_size: int = UPSERT_BATCH_SIZE
) -> None:
    if not ids:
        return
    with collection._connect() as conn:
        with conn.cursor() as cur:
            for start in range(0, len(ids), batch_size):
                batch = ids[start : start + batch_size]
                cur.execute(
                    "DELETE FROM repo_chunks WHERE collection_name = %s AND id = ANY(%s)",
                    (collection.name, batch),
                )
        conn.commit()


def delete_repo(collection: PgCollection, repo_id: str) -> None:
    with collection._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM repo_chunks WHERE collection_name = %s AND repo_id = %s",
                (collection.name, repo_id),
            )
        conn.commit()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def existing_hashes(collection: PgCollection, repo_id: str) -> dict[str, str]:
    out: dict[str, str] = {}
    offset = 0
    with collection._connect() as conn:
        with conn.cursor() as cur:
            while True:
                cur.execute(
                    """
                    SELECT id, content_hash FROM repo_chunks
                    WHERE collection_name = %s AND repo_id = %s
                    ORDER BY id
                    LIMIT %s OFFSET %s
                    """,
                    (collection.name, repo_id, GET_PAGE_SIZE, offset),
                )
                rows = cur.fetchall()
                for chunk_id, content_hash in rows:
                    out[chunk_id] = content_hash or ""
                if len(rows) < GET_PAGE_SIZE:
                    return out
                offset += GET_PAGE_SIZE


def build_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Normalize caller filters into an internal where dict."""
    clauses: list[dict[str, Any]] = []
    for key, value in (filters or {}).items():
        if key.endswith("__contains"):
            clauses.append({key.removesuffix("__contains"): {"$contains": value}})
        elif isinstance(value, (list, tuple, set)):
            clauses.append({key: {"$in": list(value)}})
        else:
            clauses.append({key: value})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _is_operator_where(where: dict[str, Any]) -> bool:
    return any(str(k).startswith("$") for k in where)


def _flatten_where(where: dict[str, Any] | None) -> list[tuple[str, str, Any]]:
    """Return list of (key, op, value) with op in eq|in|contains."""
    if not where:
        return []
    if "$and" in where:
        out: list[tuple[str, str, Any]] = []
        for part in where["$and"]:
            out.extend(_flatten_where(part))
        return out
    out = []
    for key, value in where.items():
        if isinstance(value, dict):
            if "$in" in value:
                out.append((key, "in", list(value["$in"])))
            elif "$contains" in value:
                out.append((key, "contains", value["$contains"]))
            elif "$eq" in value:
                out.append((key, "eq", value["$eq"]))
            else:
                out.append((key, "eq", value))
        else:
            out.append((key, "eq", value))
    return out


_SAFE_KEY = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _sql_where(where: dict[str, Any] | None, params: list[Any]) -> str:
    parts: list[str] = []
    for key, op, value in _flatten_where(where):
        if not _SAFE_KEY.match(key):
            raise ValueError(f"Unsafe metadata key: {key!r}")
        if op == "eq":
            parts.append("(metadata ->> %s) = %s")
            params.extend([key, str(value)])
        elif op == "in":
            placeholders = ", ".join(["%s"] * len(value))
            parts.append(f"(metadata ->> %s) IN ({placeholders})")
            params.append(key)
            params.extend([str(v) for v in value])
        elif op == "contains":
            parts.append(
                """(
                    (jsonb_typeof(metadata -> %s) = 'array' AND metadata -> %s ? %s)
                    OR (metadata ->> %s) ILIKE %s
                )"""
            )
            params.extend([key, key, str(value), key, f"%{value}%"])
    if not parts:
        return "TRUE"
    return " AND ".join(parts)


def get_by_metadata(
    where: dict[str, Any] | None,
    *,
    collection: PgCollection,
    limit: int | None = None,
) -> list[Hit]:
    w = build_where(where) if where and not _is_operator_where(where) else where
    params: list[Any] = [collection.name]
    clause = _sql_where(w, params)
    sql = f"""
        SELECT id, document, metadata FROM repo_chunks
        WHERE collection_name = %s AND ({clause})
        ORDER BY id
    """
    if limit is not None:
        sql += " LIMIT %s"
        params.append(int(limit))

    with collection._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    hits: list[Hit] = []
    for i, doc, meta in rows:
        if isinstance(meta, str):
            meta = json.loads(meta)
        hits.append(Hit(id=i, text=doc or "", metadata=meta or {}))
    return hits


def query(
    text: str,
    *,
    collection: PgCollection,
    n_results: int = 5,
    filters: dict[str, Any] | None = None,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[Hit]:
    if embed is None:
        from orchestrator.indexing import gateway_client

        embed = gateway_client.embed_texts
    vector = embed([text])[0]
    if len(vector) != collection.embedding_dim:
        raise ValueError(
            f"Query embedding dim {len(vector)} != collection dim {collection.embedding_dim}"
        )

    where = build_where(filters)
    params: list[Any] = [_vec_literal(vector), collection.name]
    clause = _sql_where(where, params)
    params.append(_vec_literal(vector))
    params.append(int(n_results))
    sql = f"""
        SELECT id, document, metadata,
               (embedding <=> %s::vector) AS distance
        FROM repo_chunks
        WHERE collection_name = %s AND embedding IS NOT NULL AND ({clause})
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    with collection._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    hits: list[Hit] = []
    for i, doc, meta, dist in rows:
        if isinstance(meta, str):
            meta = json.loads(meta)
        hits.append(
            Hit(
                id=i,
                text=doc or "",
                metadata=meta or {},
                distance=float(dist) if dist is not None else None,
            )
        )
    return hits
