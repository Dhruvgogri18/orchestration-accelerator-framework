"""Tests for the extraction API. The LLM gateway call is mocked
throughout -- these tests prove our code handles good/bad LLM output
correctly, not that the LLM itself behaves (that needs separate
manual/integration testing against the real gateway).
"""
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from orchestrator.api.main import app
from orchestrator.extraction.extractor import ExtractionError, extract_turn

client = TestClient(app)


READY_RESPONSE = json.dumps({
    "status": "ready",
    "plan": {
        "name": "process-uploaded-file",
        "summary": "Validates an uploaded file and notifies on failure.",
        "steps": [
            {"id": "ValidateFile", "service": "Lambda", "description": "Validates the file"},
            {"id": "NotifyFailure", "service": "SNS", "description": "Notifies on failure"},
        ],
    },
})

CLARIFY_RESPONSE = json.dumps({
    "status": "needs_clarification",
    "clarifying_question": "What should happen if validation succeeds?",
})


# ---- extractor.py unit tests -------------------------------------

def test_extract_turn_ready():
    with patch("orchestrator.extraction.extractor.chat_completion", return_value=READY_RESPONSE):
        result = extract_turn([{"role": "user", "content": "read a file and validate it"}])
    assert result.status == "ready"
    assert result.plan.name == "process-uploaded-file"
    assert len(result.plan.steps) == 2


def test_extract_turn_needs_clarification():
    with patch("orchestrator.extraction.extractor.chat_completion", return_value=CLARIFY_RESPONSE):
        result = extract_turn([{"role": "user", "content": "process my file somehow"}])
    assert result.status == "needs_clarification"
    assert result.clarifying_question


def test_extract_turn_strips_markdown_fences():
    fenced = f"```json\n{READY_RESPONSE}\n```"
    with patch("orchestrator.extraction.extractor.chat_completion", return_value=fenced):
        result = extract_turn([{"role": "user", "content": "..."}])
    assert result.status == "ready"


def test_extract_turn_rejects_malformed_json():
    with patch("orchestrator.extraction.extractor.chat_completion", return_value="not json at all"):
        with pytest.raises(ExtractionError):
            extract_turn([{"role": "user", "content": "..."}])


def test_extract_turn_rejects_ready_with_no_plan():
    """Guards the exact hallucination-adjacent bug the Pydantic
    validator in models.py exists for: LLM says 'ready' but forgets
    to actually include a plan."""
    bad = json.dumps({"status": "ready"})
    with patch("orchestrator.extraction.extractor.chat_completion", return_value=bad):
        with pytest.raises(ExtractionError):
            extract_turn([{"role": "user", "content": "..."}])


# ---- API integration tests -----------------------------------------

def test_full_flow_ready():
    session_resp = client.post("/session")
    assert session_resp.status_code == 200
    session_id = session_resp.json()["session_id"]

    with patch("orchestrator.api.routers.session_router.extract_turn") as mock_extract:
        from orchestrator.extraction.models import ExtractionResult
        mock_extract.return_value = ExtractionResult.model_validate(json.loads(READY_RESPONSE))

        msg_resp = client.post(
            f"/session/{session_id}/message",
            json={"message": "read a file and validate it"},
        )

    assert msg_resp.status_code == 200
    body = msg_resp.json()
    assert body["status"] == "ready"
    assert body["plan"]["name"] == "process-uploaded-file"


def test_full_flow_needs_clarification():
    session_id = client.post("/session").json()["session_id"]

    with patch("orchestrator.api.routers.session_router.extract_turn") as mock_extract:
        from orchestrator.extraction.models import ExtractionResult
        mock_extract.return_value = ExtractionResult.model_validate(json.loads(CLARIFY_RESPONSE))

        msg_resp = client.post(
            f"/session/{session_id}/message",
            json={"message": "process my file somehow"},
        )

    body = msg_resp.json()
    assert body["status"] == "needs_clarification"
    assert body["question"]


def test_message_unknown_session_returns_404():
    resp = client.post("/session/does-not-exist/message", json={"message": "hi"})
    assert resp.status_code == 404


def test_extraction_error_returns_502_not_500_traceback():
    session_id = client.post("/session").json()["session_id"]
    with patch("orchestrator.api.routers.session_router.extract_turn", side_effect=ExtractionError("boom")):
        resp = client.post(f"/session/{session_id}/message", json={"message": "hi"})
    assert resp.status_code == 502


def test_conversation_history_accumulates():
    """Confirms turn 2 actually includes turn 1's content -- this is
    the concrete guard against the LLM 'forgetting' earlier answers.
    """
    session_id = client.post("/session").json()["session_id"]

    with patch("orchestrator.api.routers.session_router.extract_turn") as mock_extract:
        from orchestrator.extraction.models import ExtractionResult
        mock_extract.return_value = ExtractionResult.model_validate(json.loads(CLARIFY_RESPONSE))
        client.post(f"/session/{session_id}/message", json={"message": "first message"})

    with patch("orchestrator.api.routers.session_router.extract_turn") as mock_extract:
        from orchestrator.extraction.models import ExtractionResult
        mock_extract.return_value = ExtractionResult.model_validate(json.loads(READY_RESPONSE))
        client.post(f"/session/{session_id}/message", json={"message": "second message"})

        sent_history = mock_extract.call_args[0][0]
        contents = [m["content"] for m in sent_history]

    assert "first message" in contents
    assert "second message" in contents