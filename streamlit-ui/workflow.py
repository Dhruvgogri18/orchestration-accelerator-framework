"""
UI orchestration client.

USE_MOCK_API=true  -> local mock (dev without API)
USE_MOCK_API=false -> real FastAPI session endpoints
"""

from __future__ import annotations

import json
from typing import Any

import requests

from settings import get_settings

_settings = get_settings()

USE_MOCK = _settings.use_mock_api
API_BASE_URL = _settings.orchestration_api_url.rstrip("/")
API_TIMEOUT = float(_settings.orchestration_api_timeout)


class OrchestrationApiError(Exception):
    """Raised when the orchestration API call fails."""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _request(method: str, path: str, payload: dict | None = None) -> dict:
    url = f"{API_BASE_URL}{path}"
    try:
        response = requests.request(
            method=method,
            url=url,
            json=payload,
            timeout=API_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
    except requests.RequestException as exc:
        raise OrchestrationApiError(f"Could not reach API at {url}: {exc}") from exc

    try:
        data = response.json() if response.content else {}
    except ValueError:
        data = {"detail": response.text}

    if not response.ok:
        detail = data.get("detail") if isinstance(data, dict) else data
        raise OrchestrationApiError(
            f"API {method} {path} failed ({response.status_code}): {detail}"
        )
    return data if isinstance(data, dict) else {"data": data}


def _empty_artifact(session_id: str = "", stage: str = "failed", text: str = "") -> dict:
    return {
        "session_id": session_id,
        "run_id": session_id,  # alias for older UI helpers
        "stage": stage,
        "assistant_text": text,
        "clarifying_questions": [],
        "summary": None,
        "workflow_spec": None,
        "plan": None,
        "placeholders": [],
        "graph_dot": None,
        "asl": None,
        "asl_json": None,
        "validations": {
            "json_ok": None,
            "structure_ok": None,
            "aws_validate_ok": None,
            "errors": [],
            "aws_details": None,
        },
        "spec_validation": None,
        "deployment": None,
        "remaining_placeholders": [],
        "api_raw": None,
    }


def _normalize_message_response(session_id: str, data: dict) -> dict:
    """Map MessageResponse -> UI artifact."""
    status = data.get("status")
    art = _empty_artifact(session_id)
    art["api_raw"] = data

    if status == "needs_clarification":
        q = data.get("question") or ""
        art["stage"] = "clarifying"
        art["clarifying_questions"] = [q] if q else []
        art["assistant_text"] = q or "I need a bit more detail."
        return art

    if status == "out_of_scope":
        msg = data.get("message") or (
            "This assistant is only designed to help you build AWS Step "
            "Functions workflows from a natural-language description. "
            "It cannot answer general or unrelated questions. Please describe "
            "a workflow you want to orchestrate."
        )
        art["stage"] = "out_of_scope"
        art["assistant_text"] = msg
        return art

    if status == "ready":
        plan = data.get("plan")
        art["stage"] = "plan_review"
        art["plan"] = plan
        art["workflow_spec"] = plan  # render as JSON in UI
        art["summary"] = data.get("summary_markdown")
        art["graph_dot"] = data.get("diagram_dot")
        art["placeholders"] = list(data.get("placeholders") or [])
        name = (plan or {}).get("name") if isinstance(plan, dict) else None
        art["assistant_text"] = (
            data.get("summary_markdown")
            or (f"Plan ready: **{name}**. Review and approve, or request changes.")
        )
        return art

    art["stage"] = "failed"
    art["assistant_text"] = f"Unexpected API status: {status!r}"
    return art


def _normalize_approve_response(session_id: str, data: dict) -> dict:
    """Map ApproveResponse -> UI artifact."""
    art = _empty_artifact(session_id)
    art["api_raw"] = data
    art["stage"] = "spec_approved"
    plan = data.get("resolved_plan")
    art["plan"] = plan
    # Prefer the enriched workflow_spec when Stage 3.5 completed;
    # fall back to the resolved plan so the UI still has something to show.
    art["workflow_spec"] = data.get("workflow_spec") or plan
    remaining = list(data.get("remaining_placeholders") or [])
    art["placeholders"] = remaining
    art["remaining_placeholders"] = remaining
    art["enrichment_error"] = data.get("enrichment_error")
    art["enrichment_question"] = data.get("enrichment_question")

    parts = [
        "Plan **approved**. Resolved specification is locked on the server.",
        "",
    ]
    if remaining:
        parts.append("Remaining placeholders (allowed until later pre-flight):")
        parts.extend(f"- `{p}`" for p in remaining)
    else:
        parts.append("All known placeholders were resolved (or none were present).")

    if data.get("enrichment_question"):
        parts.append("")
        parts.append(
            "Enrichment needs a clarification before the full workflow spec "
            "can be produced:"
        )
        parts.append(data["enrichment_question"])
    elif data.get("enrichment_error"):
        parts.append("")
        parts.append(f"Enrichment failed: {data['enrichment_error']}")
    elif data.get("workflow_spec"):
        parts.append("")
        parts.append("Workflow specification is ready (enrichment complete).")
    else:
        parts.append("")
        parts.append("Enrichment did not return a workflow_spec.")

    parts.append("")
    parts.append("ASL generation and deploy are not available from the API yet.")
    art["assistant_text"] = "\n".join(parts)
    return art


# ---------------------------------------------------------------------------
# Real API
# ---------------------------------------------------------------------------

def create_session() -> str:
    data = _request("POST", "/session")
    session_id = data.get("session_id")
    if not session_id:
        raise OrchestrationApiError(f"POST /session missing session_id: {data}")
    return session_id


def api_send_message(session_id: str, message: str) -> dict:
    data = _request(
        "POST",
        f"/session/{session_id}/message",
        {"message": message},
    )
    return _normalize_message_response(session_id, data)


def api_reject_plan(session_id: str, feedback: str) -> dict:
    data = _request(
        "POST",
        f"/session/{session_id}/reject",
        {"feedback": feedback},
    )
    return _normalize_message_response(session_id, data)


def api_approve_plan(session_id: str, overrides: dict[str, str] | None = None) -> dict:
    data = _request(
        "POST",
        f"/session/{session_id}/approve",
        {"overrides": overrides or {}},
    )
    return _normalize_approve_response(session_id, data)


def api_answer_enrichment(session_id: str, answer: str) -> dict:
    data = _request(
        "POST",
        f"/session/{session_id}/enrichment/answer",
        {"answer": answer},
    )
    return _normalize_approve_response(session_id, data)


def api_get_session(session_id: str) -> dict:
    return _request("GET", f"/session/{session_id}")


# ---------------------------------------------------------------------------
# Public functions used by app.py
# ---------------------------------------------------------------------------

def ensure_session(session_id: str | None) -> str:
    """Return existing session_id or create a new backend session."""
    if USE_MOCK:
        # Mock does not need a pre-created session; send_message creates run_id.
        return session_id or ""
    if session_id:
        return session_id
    return create_session()


def send_message(
    message: str,
    run_id: str | None = None,
    region: str = "ap-south-1",
    env: str = "dev",
) -> dict:
    """
    run_id is the backend session_id when not mocking.
    region/env reserved for future API fields.
    """
    _ = region, env
    if USE_MOCK:
        return _mock_send_message(message, run_id=run_id or None)

    if not run_id:
        run_id = create_session()
    return api_send_message(run_id, message)


def reject_plan(
    run_id: str,
    feedback: str,
    region: str = "ap-south-1",
    env: str = "dev",
) -> dict:
    _ = region, env
    if USE_MOCK:
        # Treat reject as another user message in mock.
        return _mock_send_message(feedback, run_id=run_id)
    return api_reject_plan(run_id, feedback)


def approve_plan(run_id: str, overrides: dict[str, str] | None = None) -> dict:
    if USE_MOCK:
        _mock_pause()
        # Mock: jump to a simple "approved" artifact (no ASL API).
        return {
            **_empty_artifact(run_id, "spec_approved", "Mock plan approved."),
            "workflow_spec": {
                "name": "mock-approved",
                "steps": [
                    {"id": "Validate", "type": "Task", "service": "Lambda"},
                    {"id": "Process", "type": "Task", "service": "Glue"},
                    {"id": "Notify", "type": "Task", "service": "SNS"},
                ],
            },
            "remaining_placeholders": [],
            "enrichment_error": None,
            "enrichment_question": None,
        }
    return api_approve_plan(run_id, overrides or {})


def answer_enrichment(run_id: str, answer: str) -> dict:
    """Resume Stage 3.5 enrichment after a clarifying question."""
    if USE_MOCK:
        _mock_pause()
        return {
            **_empty_artifact(run_id, "spec_approved", "Mock enrichment answered."),
            "workflow_spec": {
                "name": "mock-approved",
                "steps": [
                    {"id": "Validate", "type": "Task", "service": "Lambda"},
                    {"id": "Process", "type": "Task", "service": "Glue"},
                    {"id": "Notify", "type": "Task", "service": "SNS"},
                ],
            },
            "remaining_placeholders": [],
            "enrichment_error": None,
            "enrichment_question": None,
        }
    return api_answer_enrichment(run_id, answer)


# Back-compat names (old mock UI). Prefer approve_plan / reject_plan.
def approve_spec(run_id: str, workflow_spec_override=None, region: str = "ap-south-1") -> dict:
    _ = workflow_spec_override, region
    return approve_plan(run_id, overrides={})


def approve_and_deploy(*args, **kwargs) -> dict:
    raise OrchestrationApiError(
        "Deploy is not available yet — no deploy API endpoint."
    )


def revalidate_asl(asl: Any) -> dict:
    raise OrchestrationApiError("ASL validation API is not available yet.")


def build_graphviz_dot(asl_definition: dict) -> str:
    """Kept for optional local ASL graphs later; plan diagrams come from API."""
    try:
        states = asl_definition["States"]
        start = asl_definition["StartAt"]
    except Exception:
        return "digraph G {}"
    lines = [
        "digraph Workflow {",
        'rankdir="TB";',
        f'start -> "{start}";',
    ]
    for name, state in states.items():
        lines.append(f'"{name}" [label="{name}"];')
        if "Next" in state:
            lines.append(f'"{name}" -> "{state["Next"]}";')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Lightweight mock (optional) — enough for UI shell without API
# ---------------------------------------------------------------------------

_MOCK_RUNS: dict[str, dict] = {}

# Artificial delay so the typing indicator is visible in mock mode.
# Set to 0 to disable.
_MOCK_LATENCY_SEC = 3


def _mock_pause() -> None:
    import time

    if _MOCK_LATENCY_SEC > 0:
        time.sleep(_MOCK_LATENCY_SEC)


def _looks_out_of_scope(text: str) -> bool:
    """Lightweight mock heuristic — real API relies on the extraction LLM."""
    lower = (text or "").lower()
    workflow_hints = (
        "workflow",
        "step function",
        "stepfunctions",
        "orchestrat",
        "lambda",
        "s3",
        "sns",
        "sqs",
        "dynamodb",
        "glue",
        "eventbridge",
        "pipeline",
        "state machine",
        "on failure",
        "retry",
    )
    if any(h in lower for h in workflow_hints):
        return False
    off_topic = (
        "distance",
        "kashmir",
        "kanyakumari",
        "weather",
        "recipe",
        "who won",
        "capital of",
        "tell me a joke",
    )
    if any(h in lower for h in off_topic):
        return True
    # Short messages with no AWS/workflow vocabulary → treat as off-topic in mock.
    return len(lower.split()) >= 4 and not any(
        h in lower for h in ("aws", "amazon", "step", "task", "queue", "bucket")
    )


def _mock_send_message(message: str, run_id: str | None = None) -> dict:
    import uuid

    _mock_pause()
    text = (message or "").strip()

    if _looks_out_of_scope(text):
        sid = run_id or str(uuid.uuid4())
        if sid not in _MOCK_RUNS:
            _MOCK_RUNS[sid] = {"session_id": sid, "n": 0}
        msg = (
            "This assistant is only designed to help you build AWS Step Functions "
            "workflows from a natural-language description. It cannot answer "
            "general questions (for example travel distances or unrelated topics). "
            "Please describe a workflow you want to orchestrate — steps, AWS "
            "services, and what should happen on success or failure."
        )
        return {
            **_empty_artifact(sid, "out_of_scope", msg),
            "assistant_text": msg,
        }

    if not run_id or run_id not in _MOCK_RUNS:
        sid = run_id or str(uuid.uuid4())
        _MOCK_RUNS[sid] = {"session_id": sid, "n": 0}
        return {
            **_empty_artifact(sid, "clarifying", "What should happen on failure?"),
            "clarifying_questions": ["What should happen on failure?"],
        }

    run = _MOCK_RUNS[run_id]
    run["n"] += 1
    if run["n"] == 1 and len(text) < 3:
        return {
            **_empty_artifact(run_id, "clarifying", "Please add a bit more detail."),
            "clarifying_questions": ["Please add a bit more detail."],
        }

    plan = {
        "name": "mock-s3-workflow",
        "summary": "Validate S3 object, process, notify.",
        "steps": [
            {"id": "Validate", "service": "Lambda", "description": "Validate input"},
            {"id": "Process", "service": "Glue", "description": "Run job TODO_GLUE_JOB"},
            {"id": "Notify", "service": "SNS", "description": "Notify TODO_SNS_TOPIC"},
        ],
    }
    return {
        **_empty_artifact(run_id, "plan_review", "### mock-s3-workflow\n\nPlan ready."),
        "plan": plan,
        "workflow_spec": plan,
        "summary": "### mock-s3-workflow\n\nValidate → process → notify.",
        "graph_dot": 'digraph plan { rankdir=TB; "Validate" -> "Process" -> "Notify"; }',
        "placeholders": ["TODO_GLUE_JOB", "TODO_SNS_TOPIC"],
        "assistant_text": "### mock-s3-workflow\n\nPlan ready for review.",
    }