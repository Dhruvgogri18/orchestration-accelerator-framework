"""HTTP request/response schemas for session routes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from orchestrator.api.schemas.common import Plan


class StartSessionResponse(BaseModel):
    session_id: str


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1)


class MessageResponse(BaseModel):
    status: Literal["ready", "needs_clarification"]
    plan: Plan | None = None
    question: str | None = None
    # Populated only when status == "ready" -- see plan_render_service.
    # Sent automatically so the client never has to make a second call
    # just to get a human-friendly view of the plan.
    summary_markdown: str | None = None
    diagram_dot: str | None = None
    placeholders: list[str] | None = None


class ApproveRequest(BaseModel):
    # e.g. {"TODO_S3_BUCKET": "my-uploads-bucket", "TODO_GLUE_JOB_NAME": "nightly-etl-job"}
    # Any placeholder not included here is left unresolved in the
    # approved plan -- see apply_overrides() in plan_render_service for why
    # that's allowed rather than blocked.
    overrides: dict[str, str] = Field(default_factory=dict)


class ApproveResponse(BaseModel):
    status: Literal["approved"]
    resolved_plan: Plan
    remaining_placeholders: list[str]
    workflow_spec: dict | None = None
    enrichment_error: str | None = None
    # Set when enrichment paused to ask the user something -- answer
    # via POST /session/{id}/enrichment/answer.
    enrichment_question: str | None = None


class EnrichmentAnswerRequest(BaseModel):
    answer: str = Field(..., min_length=1)


class RejectRequest(BaseModel):
    feedback: str = Field(..., min_length=1, description="What should change about the plan.")


class SessionStateResponse(BaseModel):
    """Full read-only view of a session -- everything a client needs
    to render its UI after a page load/refresh, in one call.
    """

    session_id: str
    history: list[dict]
    latest_plan: Plan | None = None
    locked_spec: dict | None = None