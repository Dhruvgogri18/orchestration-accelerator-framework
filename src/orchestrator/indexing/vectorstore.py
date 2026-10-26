"""Persistent Chroma collection used as the repo-context vector store.

Behaviour verified against chromadb 1.5.0 and 1.5.9 (array metadata needs
>= 1.5.0):

* Cosine distance, embeddings supplied by us (``embedding_function=None``) --
  Chroma never calls an embedding model on its own.
* ``upsert`` MERGES metadata: a key that disappears from a chunk would keep its
  old value. ``upsert_chunks`` therefore deletes a chunk's id first, so a
  re-indexed chunk is exactly what the chunker produced.
* Empty id lists, empty metadata arrays and mixed-type arrays raise; the
  helpers here never send them (the chunker drops empty values).
* The collection records the embedding model and schema version it was built
  with. Opening it with a different model/schema raises ``IndexMismatchError``
  instead of silently mixing vector spaces.
* Collection names must be 3-512 characters from [a-zA-Z0-9._-].
* The collection itself is the index manifest: every chunk stores its
  ``content_hash`` and ``repo_id``, so re-indexing diffs against what is
  actually stored (``existing_hashes``), with no second state file to drift.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from orchestrator.ingestion.models import SCHEMA_VERSION, Chunk

UPSERT_BATCH_SIZE = 100
GET_PAGE_SIZE = 1000
_COLLECTION_CONFIG = {"hnsw": {"space": "cosine"}}


class IndexMismatchError(RuntimeError):
    """The existing collection was built with a different model or schema."""


@dataclass(frozen=True)
class Hit:
    id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None = None  # cosine distance; None for exact lookups


def _settings():
    from orchestrator.config import get_settings

    return get_settings()


def get_client(persist_dir: str | Path | None = None) -> chromadb.ClientAPI:
    """Persistent client. Telemetry is off: repo content never leaves the box
    except through the configured embeddings gateway."""
    path = Path(persist_dir) if persist_dir else Path(_settings().chroma_persist_dir)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
    )


def _check_compatible(collection: Any, expected: dict[str, Any]) -> None:
    stored = collection.metadata or {}
    for key in ("schema_version", "embedding_model", "embedding_dim"):
        if key in expected and key in stored and stored[key] != expected[key]:
            raise IndexMismatchError(
                f"Collection '{collection.name}' was built with {key}={stored[key]!r} but this run "
                f"uses {expected[key]!r}. Vectors from different models/schemas must not be mixed: "
                "re-run `orchestrator index --rebuild` after deleting the collection, or point "
                "CHROMA_COLLECTION_NAME at a new one."
            )


def get_collection(
    *,
    client: chromadb.ClientAPI | None = None,
    name: str | None = None,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    create: bool = True,
):
    """Open (and with ``create=True``, create) the repo-context collection.

    Returns None when ``create=False`` and it does not exist yet.
    """
    client = client or get_client()
    name = name or _settings().chroma_collection_name
    expected: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    if embedding_model:
        expected["embedding_model"] = embedding_model
    if embedding_dim:
        expected["embedding_dim"] = embedding_dim

    if create:
        if "embedding_model" not in expected:
            raise RuntimeError(
                "No embedding model to record on the collection: set GATEWAY_EMBEDDING_MODEL."
            )
        collection = client.get_or_create_collection(
            name,
            configuration=_COLLECTION_CONFIG,
            embedding_function=None,
            metadata=expected,
        )
    else:
        try:
            collection = client.get_collection(name, embedding_function=None)
        except Exception:  # noqa: BLE001 - not-found type differs across chroma versions
            return None
    _check_compatible(collection, expected)
    return collection


# ------------------------------------------------------------------ writes
def upsert_chunks(
    collection: Any,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    *,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> None:
    """Write chunks (ids replaced, not merged -- see module docstring)."""
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        ids = [c.id for c in batch]
        collection.delete(ids=ids)
        collection.upsert(
            ids=ids,
            documents=[c.text for c in batch],
            embeddings=embeddings[start : start + batch_size],
            metadatas=[c.stored_metadata() for c in batch],
        )


def delete_ids(collection: Any, ids: list[str], *, batch_size: int = UPSERT_BATCH_SIZE) -> None:
    for start in range(0, len(ids), batch_size):
        collection.delete(ids=ids[start : start + batch_size])


def delete_repo(collection: Any, repo_id: str) -> None:
    collection.delete(where={"repo_id": repo_id})


# ------------------------------------------------------------------ reads
def existing_hashes(collection: Any, repo_id: str) -> dict[str, str]:
    """{chunk id: content_hash} for everything stored for ``repo_id``."""
    out: dict[str, str] = {}
    offset = 0
    while True:
        page = collection.get(
            where={"repo_id": repo_id},
            include=["metadatas"],
            limit=GET_PAGE_SIZE,
            offset=offset,
        )
        ids = page["ids"]
        for chunk_id, meta in zip(ids, page["metadatas"] or [], strict=False):
            out[chunk_id] = (meta or {}).get("content_hash", "")
        if len(ids) < GET_PAGE_SIZE:
            return out
        offset += GET_PAGE_SIZE


def build_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Turn simple filters into a Chroma ``where``.

    ``{"environment": "dev"}``                  -> equality
    ``{"resource_type": ["lambda", "glue"]}``   -> $in
    ``{"resource_names__contains": "x"}``       -> array membership ($contains)
    Several filters are AND-ed.
    """
    clauses: list[dict[str, Any]] = []
    for key, value in (filters or {}).items():
        if key.endswith("__contains"):
            clauses.append({key.removesuffix("__contains"): {"$contains": value}})
        elif isinstance(value, list | tuple | set):
            clauses.append({key: {"$in": list(value)}})
        else:
            clauses.append({key: value})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def get_by_metadata(
    where: dict[str, Any] | None,
    *,
    collection: Any,
    limit: int | None = None,
) -> list[Hit]:
    """Exact lookup (no embedding): e.g. the chunk for one resource_name."""
    page = collection.get(
        where=build_where(where) if where and not _is_chroma_where(where) else where,
        include=["documents", "metadatas"],
        limit=limit,
    )
    return [
        Hit(id=i, text=t, metadata=m or {})
        for i, t, m in zip(
            page["ids"], page["documents"] or [], page["metadatas"] or [], strict=False
        )
    ]


def _is_chroma_where(where: dict[str, Any]) -> bool:
    return any(k.startswith("$") for k in where)


def query(
    text: str,
    *,
    collection: Any,
    n_results: int = 5,
    filters: dict[str, Any] | None = None,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[Hit]:
    """Semantic search; ``filters`` as in build_where()."""
    if embed is None:
        from orchestrator.indexing import gateway_client

        embed = gateway_client.embed_texts  # looked up per call so tests can patch it
    vector = embed([text])[0]
    res = collection.query(
        query_embeddings=[vector],
        n_results=n_results,
        where=build_where(filters),
        include=["documents", "metadatas", "distances"],
    )
    return [
        Hit(id=i, text=t, metadata=m or {}, distance=d)
        for i, t, m, d in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            strict=False,
        )
    ]
