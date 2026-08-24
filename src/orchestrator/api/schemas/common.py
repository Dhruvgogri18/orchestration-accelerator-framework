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
    # Populated only on the ready path when produced via
    # pipeline_analysis_service.analyze_pipeline() -- the full,
    # paper-aligned Stage 1 artifact (see schemas/pipeline_analysis.py:
    # PipelineAnalysis.model_dump()). Typed as dict rather than
    # PipelineAnalysis here to avoid a schemas/common <-> schemas/
    # pipeline_analysis import cycle (pipeline_analysis.py imports
    # PlanBranch from this module). None for the legacy single-call
    # extract_turn()/extract_turn_v2() paths, which never populate it.
    analysis: dict | None = Field(default=None)

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


# ---------------------------------------------------------------------------
# Split extraction pipeline (Stage 1-2 determinism pass #1).
#
# Three narrow shapes, one per split extraction stage (see
# extraction_service.py: classify_scope / decompose_steps /
# wire_control_flow). Each stage's LLM call must match exactly ONE of
# these -- same "prompt instructs it, Pydantic enforces it" pattern as
# ExtractionResult above. extract_turn_v2() chains all three and
# returns a normal ExtractionResult, so callers don't need to know the
# pipeline is split internally.
# ---------------------------------------------------------------------------


class ScopeResult(BaseModel):
    """Stage A: is the latest message in-domain at all? Nothing else."""

    status: Literal["in_scope", "out_of_scope"]
    message: str | None = Field(default=None, validate_default=True)

    @field_validator("message")
    @classmethod
    def message_required_when_out_of_scope(cls, v, info):
        if info.data.get("status") == "out_of_scope" and not v:
            raise ValueError("message is required when status is 'out_of_scope'")
        return v


class DraftStep(BaseModel):
    """Stage B output: id/service/description only -- no wiring yet."""

    id: str = Field(..., description="Short unique identifier, e.g. 'ValidateFile'.")
    service: str
    description: str


class StepDecompositionResult(BaseModel):
    """Stage B: break the request into steps, or ask ONE structural question."""

    status: Literal["ready", "needs_clarification"]
    steps: list[DraftStep] | None = Field(default=None, validate_default=True)
    clarifying_question: str | None = Field(default=None, validate_default=True)

    @field_validator("steps")
    @classmethod
    def steps_required_when_ready(cls, v, info):
        if info.data.get("status") == "ready" and not v:
            raise ValueError("steps is required when status is 'ready'")
        return v

    @field_validator("clarifying_question")
    @classmethod
    def question_required_when_clarifying(cls, v, info):
        if info.data.get("status") == "needs_clarification" and not v:
            raise ValueError(
                "clarifying_question is required when status is 'needs_clarification'"
            )
        return v


# Stage C (wiring) reuses Plan directly -- it must echo the same step ids
# from Stage B plus add next/on_failure/on_timeout/branches. No new model
# needed; extraction_service._check_ids_preserved() enforces the "same
# ids, same order, nothing invented/dropped" contract Plan alone can't.