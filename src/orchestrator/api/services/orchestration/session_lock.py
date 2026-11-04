"""Per-session locks so parallel chats never share state.

Different session_ids run fully in parallel (no global lock).
The same session_id is serialised so two concurrent /message calls on one
chat cannot interleave SQLite writes or graph state.

UI multi-chat: Chat A and Chat B each have their own session_id → parallel.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

_guard = threading.Lock()
_locks: dict[str, threading.RLock] = defaultdict(threading.RLock)


def _lock_for(session_id: str) -> threading.RLock:
    with _guard:
        return _locks[session_id]


@contextmanager
def session_lock(session_id: str) -> Iterator[None]:
    """Hold the exclusive lock for this session_id only."""
    lock = _lock_for(session_id or "_unknown")
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
