"""Keep secrets out of the index.

Repository text is sent to the embeddings gateway and stored in Chroma, so two
layers apply before anything is embedded:

1. ``is_secret_path`` -- files that are secrets by nature are never loaded,
   whatever the include globs say.
2. ``redact_secrets`` -- secret-looking values inside otherwise fine files are
   replaced with a marker (the key and the surrounding structure stay).

This is a safety net, not a scanner: keep real credentials out of the repo.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import PurePosixPath

_SECRET_FILE_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    "*.tfstate",
    "*.tfstate.*",
    "*.tfvars",
    "*.kdbx",
    "credentials",
)


def is_secret_path(path: str) -> bool:
    name = PurePosixPath(path).name
    return any(fnmatch.fnmatch(name, pat) for pat in _SECRET_FILE_PATTERNS)


_PRIVATE_KEY = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL
)
_AWS_KEY_ID = re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
# key = value, where the key names a secret and the value looks like a literal
# (long, no call syntax). Keys that merely NAME or LOCATE a secret
# (..._NAME, ..._ARN, ..._ID, ...) are left alone: they are what a workflow needs.
_KEY_VALUE = re.compile(
    r"""(?ix)
    (?P<key>[A-Za-z0-9_.-]*(?:secret|password|passwd|token|api[_-]?key|access[_-]?key|private[_-]?key)
            [A-Za-z0-9_.-]*)
    (?P<sep>\s*[:=]\s*)
    (?P<quote>["']?)
    (?P<value>[A-Za-z0-9/+=_\-.]{12,})
    (?=[\s"',;)}\]]|$)
    """
)
_HARMLESS_KEY_SUFFIX = re.compile(r"(?i)(name|arn|id|ref|path|prefix|label|type)$")
_MARKER = "<REDACTED>"


def redact_secrets(text: str) -> tuple[str, int]:
    """Return (redacted_text, number_of_redactions)."""
    count = 0

    def sub_block(marker: str):
        def _sub(_m: re.Match) -> str:
            nonlocal count
            count += 1
            return marker

        return _sub

    text = _PRIVATE_KEY.sub(sub_block("<REDACTED_PRIVATE_KEY>"), text)
    text = _AWS_KEY_ID.sub(sub_block("<REDACTED_AWS_KEY_ID>"), text)

    def sub_kv(m: re.Match) -> str:
        nonlocal count
        key, value = m.group("key"), m.group("value")
        if _HARMLESS_KEY_SUFFIX.search(key) or value.startswith(("TODO_", "$", "<", "{")):
            return m.group(0)
        count += 1
        return f"{key}{m.group('sep')}{m.group('quote')}{_MARKER}"

    return _KEY_VALUE.sub(sub_kv, text), count
