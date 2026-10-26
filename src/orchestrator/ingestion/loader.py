"""Repository file discovery and loading.

Walks the repo, applies the include/exclude globs, and returns normalised
``SourceFile`` records ready for chunking. Normalisation (LF line endings,
BOM stripped) matters: hashes and chunk ids must not change just because a file
was checked out on Windows.

Glob syntax: ``*`` and ``?`` stay inside one path segment, ``**`` crosses
segments, ``**/`` also matches zero directories. Brace expansion is not
supported. Matching is case-sensitive on POSIX-style relative paths.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from orchestrator.ingestion.models import SkippedFile, SourceFile
from orchestrator.ingestion.secrets import is_secret_path

DEFAULT_MAX_FILE_BYTES = 1_000_000

_FILE_TYPES = {
    ".py": "py",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".md": "md",
}


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a glob (see module docstring) to a compiled regex."""
    i, out = 0, []
    while i < len(pattern):
        c = pattern[i]
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("/**", i) and i + 3 == len(pattern):
            out.append("/.*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif c == "*":
            out.append("[^/]*")
            i += 1
        elif c == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out))


def _compile(globs: list[str]) -> list[re.Pattern[str]]:
    return [glob_to_regex(g.strip().replace("\\", "/")) for g in globs if g.strip()]


def discover_files(
    repo_path: str | Path,
    include_globs: list[str],
    exclude_globs: list[str],
) -> tuple[list[str], list[SkippedFile]]:
    """Return (sorted repo-relative POSIX paths to load, files skipped on purpose).

    Directories matching an exclude glob are pruned without descending into them.
    Symlinks are not followed. Secret-by-nature files (.env, *.pem, ...) are
    always skipped, even if an include glob matches them.
    """
    root = Path(repo_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Repository path is not a directory: {root}")
    include, exclude = _compile(include_globs), _compile(exclude_globs)

    found: list[str] = []
    skipped: list[SkippedFile] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        dirnames[:] = sorted(
            d
            for d in dirnames
            if not any(p.fullmatch(f"{rel_dir}/{d}/".lstrip("/")) for p in exclude)
        )
        for name in sorted(filenames):
            rel = f"{rel_dir}/{name}".lstrip("/")
            if any(p.fullmatch(rel) for p in exclude):
                continue
            if not any(p.fullmatch(rel) for p in include):
                continue
            if is_secret_path(rel):
                skipped.append(SkippedFile(rel, "secret-like file name"))
                continue
            found.append(rel)
    return sorted(found), skipped


def load_files(
    repo_path: str | Path,
    paths: list[str],
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> tuple[list[SourceFile], list[SkippedFile]]:
    """Read and normalise ``paths``. Unreadable, binary, oversize or empty
    files are skipped (and reported), never raised on."""
    root = Path(repo_path).resolve()
    sources: list[SourceFile] = []
    skipped: list[SkippedFile] = []
    for rel in paths:
        full = root / rel
        try:
            size = full.stat().st_size
            if size > max_file_bytes:
                skipped.append(SkippedFile(rel, f"larger than {max_file_bytes} bytes"))
                continue
            raw = full.read_bytes()
        except OSError as exc:
            skipped.append(SkippedFile(rel, f"unreadable: {exc.__class__.__name__}"))
            continue
        if b"\x00" in raw[:8192]:
            skipped.append(SkippedFile(rel, "binary file"))
            continue
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            skipped.append(SkippedFile(rel, "not valid UTF-8"))
            continue
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not text.strip():
            skipped.append(SkippedFile(rel, "empty file"))
            continue
        sources.append(
            SourceFile(
                path=rel,
                text=text,
                file_type=_FILE_TYPES.get(full.suffix.lower(), "other"),
                file_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            )
        )
    return sources, skipped


def load_repository(
    repo_path: str | Path,
    include_globs: list[str],
    exclude_globs: list[str],
    *,
    max_file_bytes: int = DEFAULT_MAX_FILE_BYTES,
) -> tuple[list[SourceFile], list[SkippedFile]]:
    """discover_files() + load_files() in one call."""
    paths, skipped = discover_files(repo_path, include_globs, exclude_globs)
    sources, load_skipped = load_files(repo_path, paths, max_file_bytes=max_file_bytes)
    return sources, [*skipped, *load_skipped]
