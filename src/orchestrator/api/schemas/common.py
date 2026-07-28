"""Domain / LLM shapes shared across services and repositories."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PlanStep(BaseModel):
    """One step in the lightweight plan (Stage 2), NOT the full
    workflow_spec.json step shape (that's Story 2.3's job, downstream
    of this). Intentionally thin -- name, service, what it does.
    """

    id: str = Field(..., description="Short unique identifier, e.g. 'ValidateFile'.")
    service: str = Field(..., description="AWS service this step uses, e.g. 'Lambda', 'SNS'.")
    description: str = Field(..., description="One sentence: what this step does.")


class Plan(BaseModel):
    name: str = Field(..., description="Proposed workflow name.")
    summary: str = Field(..., description="One or two sentence description of the whole workflow.")
    steps: list[PlanStep] = Field(..., min_length=1)


class ExtractionResult(BaseModel):
    """The exact shape the LLM must return, every single turn.

    Enforced two ways: the system prompt instructs the model to
    produce this shape, AND this model validates the parsed JSON
    before anything downstream trusts it. Prompt instructions alone
    are not enough -- models drift, so the real guarantee is here.
    """

    status: Literal["ready", "needs_clarification"]
    plan: Plan | None = Field(default=None, validate_default=True)
    clarifying_question: str | None = Field(default=None, validate_default=True)

    @field_validator("plan")
    @classmethod
    def plan_required_when_ready(cls, v, info):
        if info.data.get("status") == "ready" and v is None:
            raise ValueError("plan is required when status is 'ready'")
        return v

    @field_validator("clarifying_question")
    @classmethod
    def question_required_when_clarifying(cls, v, info):
        if info.data.get("status") == "needs_clarification" and not v:
            raise ValueError(
                "clarifying_question is required when status is 'needs_clarification'"
            )
        return v