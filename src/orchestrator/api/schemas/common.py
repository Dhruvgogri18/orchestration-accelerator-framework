"""Domain / LLM shapes shared across services and repositories."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PlanBranch(BaseModel):
    """One exclusive outcome of a choice/decision step."""

    condition: str = Field(
        ...,
        description="Short label for this arm, e.g. 'approved', 'rejected', 'timeout'.",
    )
    next: str = Field(
        ...,
        description="Step id to run when this condition is true.",
    )


class PlanStep(BaseModel):
    """One step in the lightweight plan (Stage 2), NOT the full
    workflow_spec.json step shape (that's Story 2.3's job, downstream
    of this).

    Control-flow fields (next / on_failure / on_timeout / branches) are
    optional for backward compatibility with older stored plans, but new
    extractions should fill them so the diagram matches the written steps.
    """

    id: str = Field(..., description="Short unique identifier, e.g. 'ValidateFile'.")
    service: str = Field(
        ..., description="AWS service this step uses, e.g. 'Lambda', 'SNS'."
    )
    description: str = Field(..., description="One sentence: what this step does.")
    # Success successor (linear happy path). Omit on terminal steps.
    next: str | None = Field(
        default=None,
        description="Step id to run on success when there is a single next step.",
    )
    # Catch-style compensation / error handlers.
    on_failure: list[str] = Field(
        default_factory=list,
        description="Step ids to run when this step fails.",
    )
    on_timeout: list[str] = Field(
        default_factory=list,
        description="Step ids to run when this wait/callback step times out.",
    )
    # Exclusive choice (approve vs reject, etc.). When set, prefer this
    # over a single `next` for drawing decision arms.
    branches: list[PlanBranch] = Field(
        default_factory=list,
        description="Exclusive conditional arms from a decision step.",
    )


class Plan(BaseModel):
    name: str = Field(..., description="Proposed workflow name.")
    summary: str = Field(
        ..., description="One or two sentence description of the whole workflow."
    )
    steps: list[PlanStep] = Field(..., min_length=1)


class ExtractionResult(BaseModel):
    """The exact shape the LLM must return, every single turn.

    Enforced two ways: the system prompt instructs the model to
    produce this shape, AND this model validates the parsed JSON
    before anything downstream trusts it. Prompt instructions alone
    are not enough -- models drift, so the real guarantee is here.
    """

    status: Literal["ready", "needs_clarification", "out_of_scope"]
    plan: Plan | None = Field(default=None, validate_default=True)
    clarifying_question: str | None = Field(default=None, validate_default=True)
    # Required when status is out_of_scope -- polite refusal explaining
    # this product only designs AWS Step Functions workflows.
    message: str | None = Field(default=None, validate_default=True)

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

    @field_validator("message")
    @classmethod
    def message_required_when_out_of_scope(cls, v, info):
        if info.data.get("status") == "out_of_scope" and not v:
            raise ValueError(
                "message is required when status is 'out_of_scope'"
            )
        return v
