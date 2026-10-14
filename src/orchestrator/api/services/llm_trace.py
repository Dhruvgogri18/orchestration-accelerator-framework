"""LLM process logger — captures behind-the-scenes model behaviour.

Writes one plain-text, human-readable log file **per session** (instead of
a single shared file everyone's requests get interleaved into), so a
developer debugging one user's conversation can just open that session's
file and read it top to bottom like a transcript:

  - which pipeline stage ran (extraction, scope, decompose, wire, enrich…)
  - the messages sent to the model (system + history + user)
  - the raw model completion
  - duration, model name, temperature/seed
  - optional stage events (self-consistency sample, validation error, etc.)

Layout on disk (under OAF_LLM_LOG_DIR, default "logs"):

  logs/
    sessions/
      2f9a7e21-....log     # one file per session_id — full readable transcript
      _unassigned.log      # anything logged before a session_id is known
    index.log               # one line per event across ALL sessions, e.g.:
                             #   14:23:01.965 stage_end   flow_structure_analysis  ok   842ms  session=2f9a7e21
                             # -- grep this first to find *which* session file
                             # to open when you don't already know the id.

Enable / configure via env:

  OAF_LLM_LOG=1                         # default on; set 0 to disable
  OAF_LLM_LOG_DIR=./logs                # base directory for log files
  OAF_LLM_LOG_PROMPTS=1                 # log full message content (default 1)
  OAF_LLM_LOG_MAX_CONTENT=50000         # truncate very long fields

Usage from other modules (unchanged from before — this is a drop-in
replacement, no call sites elsewhere needed to change):

  from orchestrator.api.services.llm_trace import (
      llm_stage, log_event, start_request, get_request_id,
  )

  start_request(session_id=..., user_message=...)
  with llm_stage("extraction"):
      ...
  log_event("self_consistency_sample", sample=1, signature=...)

GOTCHA -- background threads: session_id/request_id/stage are stored in
contextvars.ContextVar, which a plain `threading.Thread` or a bare
`concurrent.futures.ThreadPoolExecutor.submit(fn)` does NOT inherit --
events logged from such a thread silently land in sessions/_unassigned.log
instead of the session's own file, because the thread sees an empty
context. If you run LLM calls concurrently (e.g. to cut latency), submit
with `contextvars.copy_context().run(fn)` instead of `fn` directly --
see pipeline_analysis_service.py's `_run_concurrently()` for the pattern.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field as _dc_field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

# ---- token accounting ------------------------------------------------
#
# tiktoken is optional. When present we get a real (if model-approximate)
# token count; when absent we fall back to a chars/4 heuristic, which is
# the commonly-cited rule of thumb for English text and is good enough
# for relative cost tracking (spotting an expensive stage/prompt change)
# even though it won't match a provider's billed count exactly.
try:
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via _TIKTOKEN_AVAILABLE=False path
    tiktoken = None  # type: ignore[assignment]
    _TIKTOKEN_AVAILABLE = False

# Universal fallback encoding -- close enough for GPT/Claude-family
# tokenizers for the purpose of relative cost tracking. We don't know
# the actual model from this layer (gateway_client owns that), so we
# don't try to pick a model-specific encoding.
_FALLBACK_ENCODING_NAME = "cl100k_base"


@lru_cache(maxsize=1)
def _encoding():
    if not _TIKTOKEN_AVAILABLE:
        return None
    try:
        return tiktoken.get_encoding(_FALLBACK_ENCODING_NAME)
    except Exception:
        return None


def count_tokens(text: str | None) -> int:
    """Best-effort token count for one string. Uses tiktoken's cl100k_base
    encoding when available, else a chars/4 heuristic. Never raises.
    """
    if not text:
        return 0
    enc = _encoding()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _token_estimate_method() -> str:
    return "tiktoken:" + _FALLBACK_ENCODING_NAME if _encoding() is not None else "heuristic:chars/4"


def _count_message_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages or []:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, str):
            total += count_tokens(content)
    return total


# ---- cost estimation ---------------------------------------------------
#
# Best-effort USD estimate, OFF by default. Set OAF_LLM_PRICING_MODEL to
# one of the keys below (or add your own) to get a cost figure alongside
# token counts. Prices are per 1M tokens (input, output) and go stale --
# treat this as a rough, relative signal for comparing stages/sessions,
# not a billing reconciliation tool.
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "claude-3-7-sonnet": (3.00, 15.00),
}


def pricing_model_hint() -> str | None:
    """Which MODEL_PRICING_PER_1M row to cost against, if any. This is a
    label for cost estimation only -- it does not select which model
    actually serves the request (that's entirely gateway_client's call).
    """
    return os.environ.get("OAF_LLM_PRICING_MODEL") or None


def estimate_cost_usd(
    prompt_tokens: int, completion_tokens: int, *, model: str | None = None
) -> float | None:
    """Returns None (not an unknown-but-zero cost) when we have no
    pricing row for `model`, so callers/log readers can tell "cost
    unknown" apart from "cost is genuinely zero".
    """
    if not model:
        return None
    row = MODEL_PRICING_PER_1M.get(model)
    if row is None:
        return None
    in_price, out_price = row
    return round(prompt_tokens * in_price / 1_000_000 + completion_tokens * out_price / 1_000_000, 6)

# ---- config ---------------------------------------------------------------

def _enabled() -> bool:
    return os.environ.get("OAF_LLM_LOG", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _log_prompts() -> bool:
    return os.environ.get("OAF_LLM_LOG_PROMPTS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _max_content() -> int:
    try:
        return max(1000, int(os.environ.get("OAF_LLM_LOG_MAX_CONTENT", "50000")))
    except ValueError:
        return 50000


def _log_root() -> Path:
    root = Path(os.environ.get("OAF_LLM_LOG_DIR", "logs"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sessions_dir() -> Path:
    d = _log_root() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_log_path() -> Path:
    return _log_root() / "index.log"


_SAFE_ID_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")


def _safe_session_filename(session_id: str | None) -> str:
    """session_id -> filesystem-safe file stem. Falls back to
    '_unassigned' for None/empty so pre-session logging still lands
    somewhere findable instead of raising.
    """
    if not session_id:
        return "_unassigned"
    cleaned = "".join(c if c in _SAFE_ID_CHARS else "_" for c in str(session_id)).strip("._")
    return cleaned or "_unassigned"


def session_log_path(session_id: str | None) -> Path:
    """Path to a given session's readable log file. Exposed so an admin
    endpoint or CLI can locate/serve it later without duplicating the
    naming rule.
    """
    return _sessions_dir() / f"{_safe_session_filename(session_id)}.log"


# ---- context (per-request / per-stage) ------------------------------------

_request_id: ContextVar[str | None] = ContextVar("llm_request_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("llm_session_id", default=None)
_stage: ContextVar[str | None] = ContextVar("llm_stage", default=None)
_user_message: ContextVar[str | None] = ContextVar("llm_user_message", default=None)

# One lock for all log writes. Volume here is low (a handful of writes per
# user turn, not per-token), so serializing them is cheap and guarantees
# lines never interleave garbled across concurrent sessions/threads --
# unlike a per-path lock scheme, it also can't deadlock or leak locks for
# session ids we've never seen before.
_write_lock = threading.Lock()

# ---- live stage status (for polling) ---------------------------------------
#
# Separate from the file logger above: this is an in-memory, thread-safe
# "what is happening right now for session X" store so a GET endpoint can
# answer instantly without tailing a log file. It only ever holds the
# latest snapshot per session (not history), so it stays tiny.
#
# NOTE: in-memory means single-process only. Fine for one uvicorn worker;
# if you scale to multiple workers/replicas, back this with something
# shared (Redis, the existing sqlite session store, etc.) instead.

_STAGE_LABELS: dict[str, str] = {
    "pipeline_analysis": "Analyzing your request…",
    "environment_inference": "Inferring the target environment…",
    "action_clause_enumeration": "Reading through what you described…",
    "component_identification": "Identifying the workflow's components…",
    "flow_structure_analysis": "Working out the control flow…",
    "parameter_extraction": "Extracting parameters…",
    "integration_analysis": "Checking integrations…",
    "enrichment": "Filling in the remaining details…",
    "apply_placeholders": "Applying your values to the workflow…",
    "asl_generation": "Generating the Step Functions definition…",
    "asl_validation": "Validating the generated workflow…",
}

_status_lock = threading.Lock()
_status_store: dict[str, dict[str, Any]] = {}

# ---- live token/cost usage (for polling + end-of-request summary) --------
#
# Same shape/limits as _status_store above: in-memory, single-process,
# latest-cumulative-total per session (not per-call history -- the full
# per-call detail already lives in the session's .log file via
# log_event("llm_call"/"llm_response", ...)). Keyed by session_id;
# "_unassigned" holds pre-session-id calls same as the file logger does.
#
# NOTE: never evicted. A long-lived server tracking many sessions should
# call clear_token_usage(session_id) once a session is truly done (e.g.
# from the same place that would otherwise let its status/log grow
# unboundedly), or back this with shared storage across workers.

_token_usage_lock = threading.Lock()
_token_usage_store: dict[str, dict[str, Any]] = {}


def _usage_key(session_id: str | None) -> str:
    return session_id or "_unassigned"


def _empty_usage() -> dict[str, Any]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "cost_is_estimate": pricing_model_hint() is not None,
        "stages": {},
    }


def _record_usage(
    session_id: str | None,
    stage: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float | None,
) -> dict[str, Any]:
    key = _usage_key(session_id)
    stage_key = stage or "_unstaged"
    with _token_usage_lock:
        totals = _token_usage_store.setdefault(key, _empty_usage())
        totals["calls"] += 1
        totals["prompt_tokens"] += prompt_tokens
        totals["completion_tokens"] += completion_tokens
        totals["total_tokens"] += prompt_tokens + completion_tokens
        if cost_usd is not None:
            totals["cost_usd"] += cost_usd

        stage_totals = totals["stages"].setdefault(
            stage_key,
            {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0},
        )
        stage_totals["calls"] += 1
        stage_totals["prompt_tokens"] += prompt_tokens
        stage_totals["completion_tokens"] += completion_tokens
        stage_totals["total_tokens"] += prompt_tokens + completion_tokens
        if cost_usd is not None:
            stage_totals["cost_usd"] += cost_usd

        return json.loads(json.dumps(totals))  # cheap deep copy for a safe-to-return snapshot


def get_token_usage(session_id: str | None) -> dict[str, Any]:
    """Cumulative prompt/completion/total token counts (and estimated
    cost, if OAF_LLM_PRICING_MODEL is set) for a session so far, broken
    down by stage. Returns a zeroed structure for an unknown session
    rather than None, so callers can render it directly.
    """
    with _token_usage_lock:
        totals = _token_usage_store.get(_usage_key(session_id))
        return json.loads(json.dumps(totals)) if totals else _empty_usage()


def clear_token_usage(session_id: str | None) -> None:
    """Drop a session's accumulated usage. Call once a session is truly
    finished if you're tracking many long-lived sessions in one process.
    """
    with _token_usage_lock:
        _token_usage_store.pop(_usage_key(session_id), None)


def _stage_label(stage: str | None) -> str:
    if not stage:
        return "Working…"
    return _STAGE_LABELS.get(stage, stage.replace("_", " ").capitalize() + "…")


def set_status(session_id: str | None, stage: str | None, *, status: str = "running") -> None:
    """Publish the current stage for a session so /status can read it.

    Called automatically by llm_stage()/end_request() -- most callers never
    need to call this directly.
    """
    if not session_id:
        return
    with _status_lock:
        if status == "done":
            _status_store.pop(session_id, None)
            return
        _status_store[session_id] = {
            "stage": stage,
            "label": _stage_label(stage),
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }


def get_status(session_id: str) -> dict[str, Any] | None:
    """Latest known stage for a session, or None if it's idle/unknown.

    Safe to call from a different request/thread than the one doing the
    work -- that's the whole point (poll this while /message is in flight
    on another connection).
    """
    with _status_lock:
        snap = _status_store.get(session_id)
        return dict(snap) if snap else None


def get_request_id() -> str | None:
    return _request_id.get()


def get_session_id() -> str | None:
    return _session_id.get()


def get_stage() -> str | None:
    return _stage.get()


def start_request(
    *,
    session_id: str | None = None,
    user_message: str | None = None,
    request_id: str | None = None,
) -> str:
    """Mark the start of a user-facing turn. Returns request_id."""
    rid = request_id or str(uuid.uuid4())
    _request_id.set(rid)
    if session_id is not None:
        _session_id.set(session_id)
    if user_message is not None:
        _user_message.set(user_message)
    log_event(
        "request_start",
        session_id=session_id,
        user_message=_truncate(user_message) if user_message else None,
    )
    set_status(session_id, None, status="running")
    return rid


def end_request(*, status: str | None = None, error: str | None = None) -> None:
    usage = get_token_usage(_session_id.get())
    log_event(
        "request_end",
        status=status,
        error=_truncate(error) if error else None,
        total_prompt_tokens=usage["prompt_tokens"],
        total_completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        total_cost_usd=usage["cost_usd"] if usage["cost_is_estimate"] else None,
    )
    set_status(_session_id.get(), None, status="done")


@contextmanager
def llm_stage(name: str, **meta: Any) -> Iterator[None]:
    """Context manager tagging nested LLM calls with a pipeline stage name."""
    token = _stage.set(name)
    t0 = time.perf_counter()
    log_event("stage_start", stage=name, **_safe_meta(meta))
    set_status(_session_id.get(), name, status="running")
    try:
        yield
        log_event(
            "stage_end",
            stage=name,
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            status="ok",
            **_safe_meta(meta),
        )
    except Exception as exc:
        log_event(
            "stage_end",
            stage=name,
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
            status="error",
            error=f"{type(exc).__name__}: {_truncate(str(exc))}",
            **_safe_meta(meta),
        )
        raise
    finally:
        _stage.reset(token)


# ---- writing ----------------------------------------------------------

def log_event(event: str, **fields: Any) -> None:
    """Record one event to its session's readable log file (+ one summary
    line in the cross-session index).
    """
    if not _enabled():
        return
    record = {
        "ts": datetime.now(timezone.utc),
        "event": event,
        "request_id": _request_id.get(),
        "session_id": fields.pop("session_id", None) or _session_id.get(),
        "stage": fields.pop("stage", None) or _stage.get(),
        **{k: v for k, v in fields.items() if v is not None},
    }
    block = _format_event_block(record)
    index_line = _format_index_line(record)

    with _write_lock:
        path = session_log_path(record["session_id"])
        with path.open("a", encoding="utf-8") as f:
            f.write(block)
        with _index_log_path().open("a", encoding="utf-8") as f:
            f.write(index_line + "\n")


@dataclass
class LlmCallHandle:
    """Returned by log_llm_call(), passed into log_llm_response(). Carries
    the perf_counter start time plus the already-computed prompt token
    count so log_llm_response() doesn't need to re-walk `messages` (and
    so a caller can't accidentally recompute it against the wrong
    message list after appending repair-loop turns).
    """

    t0: float
    prompt_tokens: int
    model: str | None = None
    session_id: str | None = _dc_field(default=None)
    stage: str | None = _dc_field(default=None)


def log_llm_call(
    *,
    messages: list[dict],
    model: str | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    extra: dict | None = None,
) -> LlmCallHandle:
    """Log an outbound LLM request and start its token/cost accounting.

    Returns an LlmCallHandle -- pass it straight into log_llm_response()
    when the completion (or error) comes back.
    """
    t0 = time.perf_counter()
    prompt_tokens = _count_message_tokens(messages or [])
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "message_count": len(messages or []),
        "roles": [m.get("role") for m in (messages or [])],
        "prompt_tokens": prompt_tokens,
        "token_estimate_method": _token_estimate_method(),
    }
    if _log_prompts():
        payload["messages"] = _sanitize_messages(messages or [])
    else:
        payload["messages_omitted"] = True
    if extra:
        payload["extra"] = _safe_meta(extra)
    log_event("llm_call", **payload)
    return LlmCallHandle(
        t0=t0,
        prompt_tokens=prompt_tokens,
        model=model,
        session_id=_session_id.get(),
        stage=_stage.get(),
    )


def log_llm_response(
    *,
    t0: float | LlmCallHandle,
    content: str | None,
    model: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Log the model completion (or error) for the matching llm_call, and
    fold its token/cost usage into the session's running total.

    `t0` accepts either the LlmCallHandle returned by log_llm_call()
    (preferred -- carries the prompt token count along) or a bare
    perf_counter float for callers that only care about timing; in the
    latter case prompt_tokens is reported as 0 since it's unknown here.

    Returns the per-call usage dict that was folded into the session
    total: {prompt_tokens, completion_tokens, total_tokens, cost_usd}.
    """
    if isinstance(t0, LlmCallHandle):
        handle = t0
        start = handle.t0
        prompt_tokens = handle.prompt_tokens
        model = model or handle.model
        session_id = handle.session_id
        stage = handle.stage
    else:
        start = t0
        prompt_tokens = 0
        session_id = _session_id.get()
        stage = _stage.get()

    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    completion_tokens = count_tokens(content) if content is not None else 0
    cost_usd = estimate_cost_usd(prompt_tokens, completion_tokens, model=model or pricing_model_hint())

    payload: dict[str, Any] = {
        "model": model,
        "duration_ms": duration_ms,
        "response_chars": len(content or ""),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost_usd,
    }
    if error:
        payload["error"] = _truncate(error)
    if _log_prompts() and content is not None:
        payload["response"] = _truncate(content)
    log_event("llm_response", **payload)

    if not error:
        # Only successful calls count toward spend -- a failed attempt's
        # prompt tokens were still billed by the provider in reality, but
        # this store is meant to answer "what did the WORK cost", and a
        # retried-after-error call already re-logs its own prompt cost.
        _record_usage(session_id, stage, prompt_tokens, completion_tokens, cost_usd)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost_usd,
    }


# ---- readable formatting ------------------------------------------------
#
# Every event becomes a block like this in the session's .log file:
#
#   ──────────────────────────────────────────────────────────────────────
#   14:23:01.965  STAGE END  flow_structure_analysis   [ok, 842.3ms]
#   ──────────────────────────────────────────────────────────────────────
#     request_id : 3f9a2b1c-...
#     ...
#
# with long text fields (prompts/responses) rendered as an indented,
# line-wrapped block underneath their key rather than crammed into one
# JSON line -- open the file and read it like a transcript, not a data
# dump you need `jq` to parse.

_RULE = "-" * 78

# Friendlier headers for the handful of well-known event types. Anything
# else (custom log_event calls like "self_consistency_sample") still gets
# a readable block via the generic fallback -- no call site anywhere else
# needs to know this formatting exists.
_EVENT_HEADERS: dict[str, str] = {
    "request_start": "REQUEST START",
    "request_end": "REQUEST END",
    "stage_start": "STAGE START",
    "stage_end": "STAGE END",
    "llm_call": "LLM CALL",
    "llm_response": "LLM RESPONSE",
}

# Fields already shown in the header line or as top-level metadata --
# don't repeat them in the key/value body.
_HEADER_FIELDS = {"ts", "event", "request_id", "session_id", "stage", "status", "duration_ms"}

# Long/structured fields that get their own indented sub-block instead of
# a single "key: value" line.
_BLOCK_FIELDS = {"messages", "response", "user_message", "signature", "extra"}


def _format_event_block(record: dict[str, Any]) -> str:
    ts: datetime = record["ts"]
    time_str = ts.strftime("%H:%M:%S.%f")[:-3]
    event = record["event"]
    header_label = _EVENT_HEADERS.get(event, event.replace("_", " ").upper())

    if event in ("stage_start", "stage_end") and record.get("stage"):
        title = f"{header_label}  {record['stage']}"
    elif record.get("stage"):
        title = f"{header_label}  ({record['stage']})"
    else:
        title = header_label

    status_bits = []
    if record.get("status"):
        status_bits.append(str(record["status"]))
    if record.get("duration_ms") is not None:
        status_bits.append(f"{record['duration_ms']}ms")
    suffix = f"  [{', '.join(status_bits)}]" if status_bits else ""

    lines = [
        _RULE,
        f"{time_str}  {title}{suffix}",
        _RULE,
    ]

    if record.get("request_id"):
        lines.append(f"  request_id : {record['request_id']}")
    if record.get("session_id"):
        lines.append(f"  session_id : {record['session_id']}")

    # Everything else, in insertion order, skipping what's already shown.
    for key, value in record.items():
        if key in _HEADER_FIELDS:
            continue
        if value is None:
            continue
        if key in _BLOCK_FIELDS:
            lines.append(f"  {key}:")
            lines.append(_indent_block(_render_block_value(key, value), "    "))
        else:
            lines.append(f"  {key:<14}: {_render_scalar(value)}")

    lines.append("")  # blank line separator between events
    return "\n".join(lines) + "\n"


def _render_block_value(key: str, value: Any) -> str:
    if key == "messages" and isinstance(value, list):
        parts = []
        for m in value:
            role = m.get("role", "?") if isinstance(m, dict) else "?"
            content = m.get("content", "") if isinstance(m, dict) else str(m)
            parts.append(f"[{role}]\n{_indent_block(str(content), '  ')}")
        return "\n\n".join(parts)
    if isinstance(value, str):
        # Model completions are usually JSON -- pretty-print when they
        # parse, fall back to the raw text otherwise.
        try:
            parsed = json.loads(value)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return value
    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _indent_block(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines()) or prefix


def _render_scalar(value: Any) -> str:
    if isinstance(value, (dict, list)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            return str(value)
    return str(value)


def _format_index_line(record: dict[str, Any]) -> str:
    ts: datetime = record["ts"]
    time_str = ts.strftime("%H:%M:%S.%f")[:-3]
    bits = [
        time_str,
        f"{record['event']:<18}",
        f"stage={record.get('stage') or '-':<32}",
    ]
    if record.get("status"):
        bits.append(f"status={record['status']}")
    if record.get("duration_ms") is not None:
        bits.append(f"{record['duration_ms']}ms")
    bits.append(f"session={record.get('session_id') or '-'}")
    if record["event"] == "request_start" and record.get("user_message"):
        um = str(record["user_message"])[:80].replace("\n", " ")
        bits.append(f'user="{um}"')
    return "  ".join(bits)


# ---- internals ------------------------------------------------------------

def _truncate(value: str | None, limit: int | None = None) -> str | None:
    if value is None:
        return None
    lim = limit if limit is not None else _max_content()
    if len(value) <= lim:
        return value
    return value[: lim - 20] + f"…[truncated {len(value) - lim + 20} chars]"


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if isinstance(content, str):
            content = _truncate(content)
        out.append({"role": role, "content": content})
    return out


def _safe_meta(meta: dict) -> dict:
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = _truncate(v) if isinstance(v, str) else v
        else:
            try:
                out[k] = json.loads(json.dumps(v, default=str))
            except Exception:
                out[k] = _truncate(str(v))
    return out