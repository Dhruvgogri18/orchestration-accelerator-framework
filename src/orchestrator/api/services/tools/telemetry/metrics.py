"""Structured telemetry for the tools layer.

Integrates with the existing llm_trace module so tool calls appear in the
same per-session transcript that engineering already uses for LLM debugging.
Also keeps an in-process aggregate for operational dashboards / health.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# In-process aggregates (process lifetime; reset on restart)
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_counters: dict[str, int] = defaultdict(int)
_latencies_ms: dict[str, list[float]] = defaultdict(list)
_errors: dict[str, int] = defaultdict(int)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ToolCallContext:
    telemetry_id: str
    tool: str
    started_at: float
    session_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@contextmanager
def trace_tool_call(
    tool: str,
    *,
    session_id: str | None = None,
    input_summary: dict[str, Any] | None = None,
) -> Iterator[ToolCallContext]:
    """Context manager that records start/end, latency, and status.

    Usage:
        with trace_tool_call("list_resources", session_id=sid) as ctx:
            ... do work ...
            ctx.meta["cache_hit"] = True
            ctx.meta["result_count"] = 12
    """
    from orchestrator.api.services.llm_trace import get_session_id, log_event

    sid = session_id or get_session_id()
    telemetry_id = uuid.uuid4().hex[:12]
    ctx = ToolCallContext(
        telemetry_id=telemetry_id,
        tool=tool,
        started_at=time.perf_counter(),
        session_id=sid,
        meta={},
    )

    log_event(
        "tool_start",
        tool=tool,
        telemetry_id=telemetry_id,
        input=input_summary or {},
    )

    status = "ok"
    error_code: str | None = None
    try:
        yield ctx
    except Exception as exc:
        status = "error"
        error_code = type(exc).__name__
        ctx.meta["error"] = str(exc)
        ctx.meta["error_code"] = error_code
        raise
    finally:
        elapsed_ms = (time.perf_counter() - ctx.started_at) * 1000.0
        ctx.meta.setdefault("latency_ms", round(elapsed_ms, 2))
        ctx.meta.setdefault("status", status)

        # Aggregate metrics
        with _lock:
            _counters[f"{tool}.{status}"] += 1
            _counters[f"{tool}.total"] += 1
            _latencies_ms[tool].append(elapsed_ms)
            # Cap memory
            if len(_latencies_ms[tool]) > 500:
                _latencies_ms[tool] = _latencies_ms[tool][-300:]
            if error_code:
                _errors[f"{tool}.{error_code}"] += 1

        log_event(
            "tool_end",
            tool=tool,
            telemetry_id=telemetry_id,
            status=status,
            latency_ms=round(elapsed_ms, 2),
            **{k: v for k, v in ctx.meta.items() if k not in ("status", "latency_ms")},
        )


def record_cache_hit(tool: str) -> None:
    with _lock:
        _counters[f"{tool}.cache_hit"] += 1


def get_tool_metrics() -> dict[str, Any]:
    """Snapshot of in-process tool metrics (for /health or admin endpoints)."""
    with _lock:
        lat: dict[str, Any] = {}
        for tool, samples in _latencies_ms.items():
            if not samples:
                continue
            s = sorted(samples)
            lat[tool] = {
                "count": len(s),
                "p50_ms": round(s[len(s) // 2], 2),
                "p95_ms": round(s[int(len(s) * 0.95)] if len(s) > 1 else s[0], 2),
                "max_ms": round(s[-1], 2),
            }
        return {
            "counters": dict(_counters),
            "latency": lat,
            "errors": dict(_errors),
            "captured_at": _now_iso(),
        }


def reset_metrics() -> None:
    """Test helper."""
    with _lock:
        _counters.clear()
        _latencies_ms.clear()
        _errors.clear()
