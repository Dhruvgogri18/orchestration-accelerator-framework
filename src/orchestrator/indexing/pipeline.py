"""Index a repository: load -> chunk -> diff against what is stored -> embed
only what changed -> write -> remove what disappeared.

Idempotent and resumable: chunk ids and content hashes are deterministic, and
the collection is the source of truth for what is already indexed, so

* re-running on an unchanged repo makes zero embedding calls,
* editing one file re-embeds only the chunks whose text or metadata changed,
* a run that dies half-way is finished by simply running it again
  (new chunks are written before stale ones are deleted).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from orchestrator.indexing import vectorstore
from orchestrator.ingestion.chunker import chunk_files
from orchestrator.ingestion.loader import load_repository
from orchestrator.ingestion.models import Chunk, SkippedFile

EMBED_GROUP_SIZE = 64  # chunks embedded + written per step (progress survives a crash)


@dataclass
class IndexReport:
    repo_id: str
    repo_path: str
    files_loaded: int = 0
    files_skipped: list[SkippedFile] = field(default_factory=list)
    chunks_total: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    embedded: int = 0
    redacted_chunks: int = 0
    dry_run: bool = False
    rebuilt: bool = False
    embedding_model: str | None = None
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def default_repo_id(repo_path: str | Path) -> str:
    name = Path(repo_path).resolve().name or "repo"
    slug = re.sub(r"[^a-z0-9._-]+", "-", name.lower()).strip("-.")
    return slug or "repo"


def _log(event: str, **fields) -> None:
    try:
        from orchestrator.api.services.llm_trace import log_event

        log_event(event, **fields)
    except Exception:  # noqa: BLE001 - logging must never break indexing
        pass


def index_repository(
    repo_path: str | Path | None = None,
    *,
    repo_id: str | None = None,
    rebuild: bool = False,
    dry_run: bool = False,
    persist_dir: str | Path | None = None,
    collection_name: str | None = None,
    include_globs: list[str] | None = None,
    exclude_globs: list[str] | None = None,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
    embedding_model: str | None = None,
    embedding_dim: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
) -> IndexReport:
    """Build or refresh the index for one repository.

    Every argument defaults to the matching setting (REPO_PATH, INCLUDE_GLOBS,
    CHROMA_PERSIST_DIR, ...), so tests and callers can pass explicit values and
    never touch the real configuration or gateway.
    """
    t0 = time.perf_counter()
    settings = None

    def cfg():
        nonlocal settings
        if settings is None:
            from orchestrator.config import get_settings

            settings = get_settings()
        return settings

    repo_path = repo_path or cfg().repo_path
    include_globs = include_globs if include_globs is not None else cfg().include_glob_list
    exclude_globs = exclude_globs if exclude_globs is not None else cfg().exclude_glob_list
    repo_id = repo_id or default_repo_id(repo_path)
    embedding_model = embedding_model or (None if dry_run else cfg().gateway_embedding_model)
    embedding_dim = embedding_dim or (None if dry_run else cfg().embedding_dimensions)

    sources, skipped = load_repository(repo_path, include_globs, exclude_globs)
    if not sources:
        raise ValueError(
            f"No indexable files found under {Path(repo_path).resolve()} "
            f"(include: {include_globs}). Refusing to continue: an empty result would delete "
            "the existing index for this repo."
        )
    chunks = chunk_files(
        sources,
        repo_id=repo_id,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        count_tokens=count_tokens,
    )
    new = {c.id: c for c in chunks}

    report = IndexReport(
        repo_id=repo_id,
        repo_path=str(Path(repo_path).resolve()),
        files_loaded=len(sources),
        files_skipped=skipped,
        chunks_total=len(chunks),
        redacted_chunks=sum(1 for c in chunks if c.metadata.get("redacted")),
        dry_run=dry_run,
        rebuilt=rebuild,
        embedding_model=embedding_model,
    )

    client = None
    try:
        store_path = Path(persist_dir) if persist_dir else Path(cfg().chroma_persist_dir)
    except Exception:  # noqa: BLE001 - a dry run must work without any setup
        if not dry_run:
            raise
        store_path = None
    if store_path is not None and not (dry_run and not store_path.exists()):
        client = vectorstore.get_client(store_path)  # a dry run never creates the store
    collection = None
    if client is not None:
        collection = vectorstore.get_collection(
            client=client,
            name=collection_name,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
            create=not dry_run,
        )

    if rebuild and collection is not None and not dry_run:
        vectorstore.delete_repo(collection, repo_id)
    existing: dict[str, str] = (
        {} if rebuild or collection is None else vectorstore.existing_hashes(collection, repo_id)
    )

    to_write: list[Chunk] = []
    for chunk_id, chunk in new.items():
        if chunk_id not in existing:
            report.added += 1
            to_write.append(chunk)
        elif existing[chunk_id] != chunk.content_hash:
            report.updated += 1
            to_write.append(chunk)
        else:
            report.unchanged += 1
    stale = [chunk_id for chunk_id in existing if chunk_id not in new]
    report.deleted = len(stale)

    if not dry_run:
        if embed is None:
            from orchestrator.indexing import gateway_client

            embed = gateway_client.embed_texts  # looked up per run so tests can patch it
        for start in range(0, len(to_write), EMBED_GROUP_SIZE):
            group = to_write[start : start + EMBED_GROUP_SIZE]
            vectors = embed([c.text for c in group])
            if len(vectors) != len(group):
                raise RuntimeError(
                    f"Embedder returned {len(vectors)} vectors for {len(group)} chunks"
                )
            vectorstore.upsert_chunks(collection, group, vectors)
            report.embedded += len(group)
        # New chunks are in; only now drop the ones that no longer exist.
        vectorstore.delete_ids(collection, stale)

    report.duration_s = round(time.perf_counter() - t0, 3)
    _log(
        "index_run",
        **{k: v for k, v in report.to_dict().items() if k != "files_skipped"},
        files_skipped=len(skipped),
    )
    return report
