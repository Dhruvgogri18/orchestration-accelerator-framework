"""Data model shared by the loader, chunker and vector store.

Chunk text and metadata are deliberately plain JSON-friendly values:
str / int / float / bool or a non-empty list of str (stored as JSONB in
Postgres). Everything here is deterministic -- the same source text always
yields byte-identical chunks, ids and hashes -- so re-indexing an unchanged
repo costs zero embedding calls.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# Bump when chunk layout or metadata semantics change. The vector store refuses
# to mix schema versions in one collection; re-run `orchestrator index --rebuild`.
SCHEMA_VERSION = 1

MetadataValue = str | int | float | bool | list[str]


@dataclass(frozen=True)
class SourceFile:
    """One repository file, loaded and normalised."""

    path: str  # repo-relative, POSIX separators
    text: str  # LF line endings, BOM stripped
    file_type: str  # "py" | "yaml" | "json" | "md" | "other"
    file_hash: str  # sha256 of the normalised text


@dataclass(frozen=True)
class SkippedFile:
    path: str
    reason: str


@dataclass(frozen=True)
class Chunk:
    """The unit that is embedded and stored.

    ``text`` is header + body: exactly what is embedded and what Chroma stores
    as the document. ``metadata`` holds the descriptive fields (see
    docs/part_b_indexing.md); ``stored_metadata()`` adds the bookkeeping ones.
    """

    id: str
    text: str
    metadata: dict[str, MetadataValue] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        """Hash of everything that is embedded or filterable: text + metadata."""
        payload = json.dumps(
            {"text": self.text, "metadata": self.metadata},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def stored_metadata(self) -> dict[str, Any]:
        return {
            **self.metadata,
            "content_hash": self.content_hash,
            "schema_version": SCHEMA_VERSION,
        }
