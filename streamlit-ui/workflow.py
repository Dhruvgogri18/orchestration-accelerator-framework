"""
UI orchestration client.

USE_MOCK_API=true -> local mock (dev without API)
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


class PreDeployCheckFailed(OrchestrationApiError):
    """Raised specifically for POST /session/{id}/deploy's 422 response --
    the structured {"reasons": [...]} list the API returns when one or
    more pre-deploy checks fail (validation not passing, leftover TODO_
    placeholders, no execution role configured, etc.). Kept distinct
    from the generic OrchestrationApiError so the UI can render these as
    an actionable checklist instead of a single opaque error string.
    """

    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("Pre-deploy checks failed: " + "; ".join(reasons))


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
        if (
            response.status_code == 422
            and isinstance(detail, dict)
            and isinstance(detail.get("reasons"), list)
        ):
            raise PreDeployCheckFailed(detail["reasons"])
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
        "spec_placeholders": [],
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
    # Spec-level TODOs (workflow_spec) — primary list for the fill form.
    # Prefer this over plan remaining + ASL scan when the API provides it.
    spec_ph = list(data.get("spec_placeholders") or [])
    art["spec_placeholders"] = spec_ph
    art["enrichment_error"] = data.get("enrichment_error")
    art["enrichment_question"] = data.get("enrichment_question")

    parts = [
        "Plan **approved**. Resolved specification is locked on the server.",
        "",
    ]
    if remaining:
        parts.append("Remaining plan placeholders (allowed until later pre-flight):")
        parts.extend(f"- `{p}`" for p in remaining)
    else:
        parts.append("All known plan placeholders were resolved (or none were present).")

    if spec_ph:
        parts.append("")
        parts.append(
            f"**{len(spec_ph)} workflow-spec placeholder(s)** must be filled "
            "before ASL can be generated:"
        )
        parts.extend(f"- `{p}`" for p in spec_ph)

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

    # Stage 4 ASL
    art["asl"] = data.get("asl")
    art["asl_json"] = data.get("asl")
    art["graph_dot"] = data.get("asl_graph_dot") or art.get("graph_dot")
    art["validations"] = data.get("asl_validations") or art.get("validations")

    if data.get("asl_error"):
        parts.append("")
        parts.append(f"ASL generation error: {data['asl_error']}")
    elif data.get("asl"):
        parts.append("")
        vals = data.get("asl_validations") or {}
        if vals.get("ok"):
            parts.append("ASL generated and **all validations passed**.")
        else:
            errs = vals.get("errors") or []
            parts.append(f"ASL generated with **{len(errs)} validation issue(s)**.")
            for e in errs[:5]:
                parts.append(f"- [{e.get('group')}/{e.get('code')}] {e.get('message')}")
    else:
        parts.append("")
        parts.append("ASL not generated yet (waiting on enrichment).")

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


def api_fill_spec_placeholders(
    session_id: str, overrides: dict[str, str] | None = None
) -> dict:
    """POST /session/{id}/spec/placeholders — resolve TODO_ tokens in
    the locked workflow_spec, then run Stage 4 once none remain.
    """
    data = _request(
        "POST",
        f"/session/{session_id}/spec/placeholders",
        {"overrides": overrides or {}},
    )
    return _normalize_approve_response(session_id, data)


def api_get_session(session_id: str) -> dict:
    return _request("GET", f"/session/{session_id}")


def api_get_status(session_id: str) -> dict:
    """GET /session/{id}/status — cheap, poll-friendly read of what the
    backend is doing *right now* for this session (e.g. "Working out the
    control flow…"). Safe to call every ~1s while a /message, /reject,
    /approve, or /enrichment/answer call is in flight on another
    connection. Returns {"busy": False} once nothing is running, and
    also (deliberately, not an error) if the API is briefly unreachable
    -- a failed status poll should never interrupt the actual request.
    """
    if USE_MOCK:
        return {"busy": False}
    try:
        return _request("GET", f"/session/{session_id}/status")
    except OrchestrationApiError:
        return {"busy": False}


def api_generate_asl(session_id: str) -> dict:
    """POST /session/{id}/asl/generate — refresh ASL from locked workflow_spec."""
    return _request("POST", f"/session/{session_id}/asl/generate")


def api_validate_asl(session_id: str, asl: dict | None = None) -> dict:
    """POST /session/{id}/asl/validate — run all five validation groups (+ AWS)."""
    return _request(
        "POST",
        f"/session/{session_id}/asl/validate",
        {"asl": asl},
    )


def api_download_asl(session_id: str) -> tuple[bytes, str]:
    """GET /session/{id}/asl/download — fetch ASL JSON bytes from the backend.

    Returns (content_bytes, filename). Raises OrchestrationApiError on failure.
    """
    url = f"{API_BASE_URL}/session/{session_id}/asl/download"
    try:
        response = requests.get(url, timeout=API_TIMEOUT)
    except requests.RequestException as exc:
        raise OrchestrationApiError(f"Could not reach API at {url}: {exc}") from exc

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise OrchestrationApiError(
            f"API GET /session/{session_id}/asl/download failed "
            f"({response.status_code}): {detail}"
        )

    filename = "workflow.asl.json"
    cd = response.headers.get("Content-Disposition") or ""
    # attachment; filename="foo.asl.json"
    if "filename=" in cd:
        part = cd.split("filename=", 1)[1].strip().strip('"').strip("'")
        if part:
            filename = part
    return response.content, filename


def api_deploy_workflow(
    session_id: str,
    state_machine_name: str | None = None,
    role_arn: str | None = None,
    tags: dict[str, str] | None = None,
    force: bool = False,
) -> dict:
    """POST /session/{id}/deploy — Stage 5: create-or-update the session's
    locked ASL as a real Step Functions state machine.

    Raises PreDeployCheckFailed (with `.reasons`) if the API's pre-deploy
    checks block the deploy (422) -- validation not passing, leftover
    TODO_ placeholders anywhere in the ASL, or no execution role
    configured. Raises the generic OrchestrationApiError for everything
    else (unknown session, no ASL yet, or AWS itself rejecting the call).
    """
    payload: dict = {"force": force}
    if state_machine_name:
        payload["state_machine_name"] = state_machine_name
    if role_arn:
        payload["role_arn"] = role_arn
    if tags:
        payload["tags"] = tags
    data = _request("POST", f"/session/{session_id}/deploy", payload)
    return {
        "state_machine_arn": data.get("state_machine_arn"),
        "action": data.get("action"),
        "state_machine_name": data.get("state_machine_name"),
        "region": data.get("region"),
        "role_arn": data.get("role_arn"),
    }


def deploy_workflow(
    run_id: str,
    state_machine_name: str | None = None,
    role_arn: str | None = None,
    tags: dict[str, str] | None = None,
    force: bool = False,
) -> dict:
    """Stage 5 entry point used by app.py. Mirrors approve_plan /
    answer_enrichment's USE_MOCK split.
    """
    if USE_MOCK:
        _mock_pause()
        return _mock_deploy_result(run_id, state_machine_name)
    return api_deploy_workflow(
        run_id,
        state_machine_name=state_machine_name,
        role_arn=role_arn,
        tags=tags,
        force=force,
    )


def _apply_asl_payload_to_artifact(art: dict, data: dict) -> dict:
    """Merge generate/validate API payload into a UI artifact."""
    if not isinstance(data, dict):
        return art
    if data.get("asl") is not None:
        art["asl"] = data["asl"]
        art["asl_json"] = data["asl"]
    if data.get("graph_dot"):
        art["graph_dot"] = data["graph_dot"]
    if data.get("validations"):
        art["validations"] = data["validations"]
    return art


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


def _mock_approved_artifact(run_id: str, text: str) -> dict:
    """Mock Stage 3.5+4 response with a minimal ASL + validations."""
    asl = {
        "Comment": "mock-approved: Validate → Process → Notify",
        "StartAt": "Validate",
        "States": {
            "Validate": {
                "Type": "Task",
                "Resource": "arn:aws:states:::lambda:invoke",
                "Next": "Process",
            },
            "Process": {
                "Type": "Task",
                "Resource": "arn:aws:states:::glue:startJobRun.sync",
                "Next": "Notify",
            },
            "Notify": {
                "Type": "Task",
                "Resource": "arn:aws:states:::sns:publish",
                "End": True,
            },
        },
    }
    return {
        **_empty_artifact(run_id, "spec_approved", text),
        "workflow_spec": {
            "name": "mock-approved",
            "summary": "Validate, process, notify",
            "start_at": "Validate",
            "steps": [
                {
                    "id": "Validate",
                    "type": "Task",
                    "service": "Lambda",
                    "resource": "arn:aws:states:::lambda:invoke",
                    "next_step": "Process",
                },
                {
                    "id": "Process",
                    "type": "Task",
                    "service": "Glue",
                    "resource": "arn:aws:states:::glue:startJobRun.sync",
                    "next_step": "Notify",
                },
                {
                    "id": "Notify",
                    "type": "Task",
                    "service": "SNS",
                    "resource": "arn:aws:states:::sns:publish",
                    "end": True,
                },
            ],
        },
        "asl": asl,
        "asl_json": asl,
        "graph_dot": build_graphviz_dot(asl),
        "validations": {
            "ok": True,
            "json_ok": True,
            "structure_ok": True,
            "reachability_ok": True,
            "spec_consistency_ok": True,
            "aws_constraints_ok": True,
            "extras_ok": True,
            "aws_validate_ok": None,
            "errors": [],
            "aws_details": {
                "attempted": False,
                "skipped": True,
                "reason": "mock mode",
            },
        },
        "remaining_placeholders": [],
        "enrichment_error": None,
        "enrichment_question": None,
        "asl_error": None,
    }


def approve_plan(run_id: str, overrides: dict[str, str] | None = None) -> dict:
    if USE_MOCK:
        _mock_pause()
        return _mock_approved_artifact(run_id, "Mock plan approved.")
    return api_approve_plan(run_id, overrides or {})


def answer_enrichment(run_id: str, answer: str) -> dict:
    """Resume Stage 3.5 enrichment after a clarifying question."""
    if USE_MOCK:
        _mock_pause()
        return _mock_approved_artifact(run_id, "Mock enrichment answered.")
    return api_answer_enrichment(run_id, answer)


def fill_spec_placeholders(run_id: str, overrides: dict[str, str] | None = None) -> dict:
    """Fill workflow_spec TODO_ tokens then regenerate ASL when clear."""
    if USE_MOCK:
        _mock_pause()
        art = _mock_approved_artifact(run_id, "Mock placeholders applied.")
        # Simulate resolution: drop remaining list when overrides provided.
        if overrides:
            art["spec_placeholders"] = []
            art["remaining_placeholders"] = []
            art["placeholders"] = []
            art["assistant_text"] = (
                "Placeholders applied (mock). ASL regenerated with resolved values."
            )
        return art
    return api_fill_spec_placeholders(run_id, overrides or {})


def generate_asl(run_id: str) -> dict:
    """Re-run Stage 4 from the locked workflow_spec."""
    if USE_MOCK:
        art = _mock_approved_artifact(run_id, "Mock ASL regenerated.")
        return {
            "asl": art["asl"],
            "validations": art["validations"],
            "graph_dot": art["graph_dot"],
        }
    return api_generate_asl(run_id)


# Back-compat names (old mock UI). Prefer approve_plan / reject_plan.
def approve_spec(run_id: str, workflow_spec_override=None, region: str = "ap-south-1") -> dict:
    _ = workflow_spec_override, region
    return approve_plan(run_id, overrides={})


def approve_and_deploy(run_id: str, *args, **kwargs) -> dict:
    """Old name kept for back-compat with any earlier callers. Prefer
    deploy_workflow() -- same thing, clearer name now that the deploy
    API endpoint actually exists.
    """
    _ = args
    return deploy_workflow(run_id, **kwargs)


def revalidate_asl(asl: Any, session_id: str | None = None) -> dict:
    """POST /session/{id}/asl/validate — runs all five validation groups (+ AWS)."""
    if USE_MOCK:
        return {
            "asl": asl,
            "validations": {
                "ok": True,
                "json_ok": True,
                "structure_ok": True,
                "reachability_ok": True,
                "spec_consistency_ok": True,
                "aws_constraints_ok": True,
                "extras_ok": True,
                "aws_validate_ok": None,
                "errors": [],
                "aws_details": {"attempted": False, "skipped": True, "reason": "mock"},
            },
            "graph_dot": build_graphviz_dot(asl) if isinstance(asl, dict) else None,
        }
    if not session_id:
        raise OrchestrationApiError("session_id required for ASL validation")
    return api_validate_asl(session_id, asl if isinstance(asl, dict) else None)


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
_MOCK_DEPLOYS: dict[str, dict] = {}

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


def _mock_deploy_result(run_id: str, state_machine_name: str | None = None) -> dict:
    """Second call for the same run_id reports 'updated' instead of
    'created', matching the real API's create-or-update behavior."""
    name = state_machine_name or f"mock-workflow-{(run_id or 'run')[:8]}"
    existing = _MOCK_DEPLOYS.get(run_id)
    action = "updated" if existing else "created"
    result = {
        "state_machine_arn": f"arn:aws:states:ap-south-1:123456789012:stateMachine:{name}",
        "action": action,
        "state_machine_name": name,
        "region": "ap-south-1",
        "role_arn": "arn:aws:iam::123456789012:role/mock-execution-role",
    }
    _MOCK_DEPLOYS[run_id] = result
    return result


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
