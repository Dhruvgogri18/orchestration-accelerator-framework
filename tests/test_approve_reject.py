"""Tests for the placeholder/override logic in plan_render.py and the
new /approve, /reject endpoints in main.py.
"""
import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from orchestrator.api.main import app
from orchestrator.extraction.models import ExtractionResult, Plan
from orchestrator.extraction.plan_render import apply_overrides, find_placeholders

client = TestClient(app)

GLUE_PLAN_DICT = {
    "name": "s3-glue-job-status-branch",
    "summary": "Read a file from S3, run a Glue job, branch on status.",
    "steps": [
        {"id": "ReadFile", "service": "Amazon S3",
         "description": "Read the input file from S3 using TODO_S3_BUCKET and TODO_S3_OBJECT_KEY."},
        {"id": "RunGlueJob", "service": "AWS Glue",
         "description": "Start and monitor the Glue job TODO_GLUE_JOB_NAME."},
        {"id": "CheckJobStatus", "service": "AWS Step Functions",
         "description": "Branch based on the completed Glue job status."},
        {"id": "NotifyFailure", "service": "TODO_NOTIFICATION_SERVICE",
         "description": "Send a failure notification using TODO_NOTIFICATION_TARGET."},
        {"id": "Finish", "service": "AWS Step Functions",
         "description": "End the workflow successfully."},
    ],
}


# ---- find_placeholders / apply_overrides unit tests -----------------

def test_find_placeholders_returns_all_unique_tokens_in_order():
    plan = Plan.model_validate(GLUE_PLAN_DICT)
    placeholders = find_placeholders(plan)
    assert placeholders == [
        "TODO_S3_BUCKET",
        "TODO_S3_OBJECT_KEY",
        "TODO_GLUE_JOB_NAME",
        "TODO_NOTIFICATION_SERVICE",
        "TODO_NOTIFICATION_TARGET",
    ]


def test_find_placeholders_empty_when_none_present():
    plan = Plan.model_validate({
        "name": "clean", "summary": "no placeholders here",
        "steps": [{"id": "A", "service": "S3", "description": "reads a file"}],
    })
    assert find_placeholders(plan) == []


def test_apply_overrides_substitutes_given_values():
    plan = Plan.model_validate(GLUE_PLAN_DICT)
    resolved, remaining = apply_overrides(plan, {
        "TODO_S3_BUCKET": "my-uploads-bucket",
        "TODO_GLUE_JOB_NAME": "nightly-etl-job",
    })
    read_step = next(s for s in resolved.steps if s.id == "ReadFile")
    glue_step = next(s for s in resolved.steps if s.id == "RunGlueJob")
    assert "my-uploads-bucket" in read_step.description
    assert "TODO_S3_BUCKET" not in read_step.description
    assert "nightly-etl-job" in glue_step.description


def test_apply_overrides_leaves_unprovided_placeholders_untouched():
    plan = Plan.model_validate(GLUE_PLAN_DICT)
    resolved, remaining = apply_overrides(plan, {"TODO_S3_BUCKET": "my-uploads-bucket"})
    assert "TODO_GLUE_JOB_NAME" in remaining
    assert "TODO_S3_BUCKET" not in remaining


def test_apply_overrides_can_resolve_service_field_too():
    plan = Plan.model_validate(GLUE_PLAN_DICT)
    resolved, remaining = apply_overrides(plan, {"TODO_NOTIFICATION_SERVICE": "SNS"})
    notify_step = next(s for s in resolved.steps if s.id == "NotifyFailure")
    assert notify_step.service == "SNS"


def test_apply_overrides_with_all_placeholders_leaves_none_remaining():
    plan = Plan.model_validate(GLUE_PLAN_DICT)
    _, remaining = apply_overrides(plan, {
        "TODO_S3_BUCKET": "my-uploads-bucket",
        "TODO_S3_OBJECT_KEY": "incoming/data.csv",
        "TODO_GLUE_JOB_NAME": "nightly-etl-job",
        "TODO_NOTIFICATION_SERVICE": "SNS",
        "TODO_NOTIFICATION_TARGET": "arn:aws:sns:us-east-1:123456789012:alerts-topic",
    })
    assert remaining == []


# ---- /message now includes rendering + placeholders on ready --------

def test_message_ready_response_includes_render_and_placeholders():
    session_id = client.post("/session").json()["session_id"]
    ready_json = json.dumps({"status": "ready", "plan": GLUE_PLAN_DICT})

    with patch("orchestrator.api.main.extract_turn") as mock_extract:
        mock_extract.return_value = ExtractionResult.model_validate(json.loads(ready_json))
        resp = client.post(f"/session/{session_id}/message", json={"message": "..."})

    body = resp.json()
    assert body["status"] == "ready"
    assert "s3-glue-job-status-branch" in body["summary_markdown"]
    assert "digraph plan" in body["diagram_dot"]
    assert "TODO_GLUE_JOB_NAME" in body["placeholders"]
    assert "TODO_GLUE_JOB_NAME" in body["summary_markdown"]  # callout is in the summary too


# ---- /approve ---------------------------------------------------------

def _get_session_to_ready(session_id: str):
    ready_json = json.dumps({"status": "ready", "plan": GLUE_PLAN_DICT})
    with patch("orchestrator.api.main.extract_turn") as mock_extract:
        mock_extract.return_value = ExtractionResult.model_validate(json.loads(ready_json))
        client.post(f"/session/{session_id}/message", json={"message": "..."})


def test_approve_without_ready_plan_returns_409():
    session_id = client.post("/session").json()["session_id"]
    resp = client.post(f"/session/{session_id}/approve", json={"overrides": {}})
    assert resp.status_code == 409


def test_approve_unknown_session_returns_404():
    resp = client.post("/session/does-not-exist/approve", json={"overrides": {}})
    assert resp.status_code == 404


def test_approve_with_full_overrides_resolves_everything():
    session_id = client.post("/session").json()["session_id"]
    _get_session_to_ready(session_id)

    resp = client.post(f"/session/{session_id}/approve", json={
        "overrides": {
            "TODO_S3_BUCKET": "my-uploads-bucket",
            "TODO_S3_OBJECT_KEY": "incoming/data.csv",
            "TODO_GLUE_JOB_NAME": "nightly-etl-job",
            "TODO_NOTIFICATION_SERVICE": "SNS",
            "TODO_NOTIFICATION_TARGET": "arn:aws:sns:us-east-1:123456789012:alerts-topic",
        }
    })
    body = resp.json()
    assert resp.status_code == 200
    assert body["status"] == "approved"
    assert body["remaining_placeholders"] == []
    assert "my-uploads-bucket" in body["resolved_plan"]["steps"][0]["description"]


def test_approve_with_partial_overrides_reports_remaining():
    session_id = client.post("/session").json()["session_id"]
    _get_session_to_ready(session_id)

    resp = client.post(f"/session/{session_id}/approve", json={
        "overrides": {"TODO_S3_BUCKET": "my-uploads-bucket"}
    })
    body = resp.json()
    assert resp.status_code == 200
    assert "TODO_GLUE_JOB_NAME" in body["remaining_placeholders"]
    assert "TODO_S3_BUCKET" not in body["remaining_placeholders"]


# ---- /reject ------------------------------------------------------

def test_reject_without_ready_plan_returns_409():
    session_id = client.post("/session").json()["session_id"]
    resp = client.post(f"/session/{session_id}/reject", json={"feedback": "change it"})
    assert resp.status_code == 409


def test_reject_unknown_session_returns_404():
    resp = client.post("/session/does-not-exist/reject", json={"feedback": "change it"})
    assert resp.status_code == 404


def test_reject_reopens_conversation_and_clears_latest_plan():
    session_id = client.post("/session").json()["session_id"]
    _get_session_to_ready(session_id)

    clarify_json = json.dumps({
        "status": "needs_clarification",
        "clarifying_question": "What notification service would you like instead?",
    })
    with patch("orchestrator.api.main.extract_turn") as mock_extract:
        mock_extract.return_value = ExtractionResult.model_validate(json.loads(clarify_json))
        resp = client.post(
            f"/session/{session_id}/reject",
            json={"feedback": "Use email instead of the notification service you picked."},
        )

    assert resp.status_code == 200
    assert resp.json()["status"] == "needs_clarification"

    # Plan should no longer be approvable until extraction says ready again.
    approve_resp = client.post(f"/session/{session_id}/approve", json={"overrides": {}})
    assert approve_resp.status_code == 409


def test_reject_feedback_is_recorded_in_history():
    session_id = client.post("/session").json()["session_id"]
    _get_session_to_ready(session_id)

    with patch("orchestrator.api.main.extract_turn") as mock_extract:
        mock_extract.return_value = ExtractionResult.model_validate(
            json.loads(json.dumps({"status": "ready", "plan": GLUE_PLAN_DICT}))
        )
        client.post(
            f"/session/{session_id}/reject",
            json={"feedback": "please use email notifications"},
        )
        sent_history = mock_extract.call_args[0][0]
        contents = [m["content"] for m in sent_history]

    assert any("please use email notifications" in c for c in contents)