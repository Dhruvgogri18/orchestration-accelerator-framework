"""Per-session locks so parallel chats never corrupt each other's state.

Multiple UI chats map to different session_ids and may hit the API at the
same time. SQLite WAL already allows concurrent readers/writers across
sessions; this module serialises *within* one session so two in-flight
/message calls for the same session_id cannot interleave plan writes.

Cross-session parallelism is unrestricted.
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
    """Hold the exclusive lock for this session_id for the duration of the block."""
    lock = _lock_for(session_id or "_unknown")
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
