"""Schemas for Stage 1: Pipeline Analysis, aligned to Section 5.1 of the
Prompt2DAG paper (Alidu, Ciavotta, De Paoli, 2025 -- arXiv:2509.13487).

The paper decomposes Pipeline Analysis into five prompt-chained
sub-stages, each building on the previous stage's structured output:

  1. Environment inference     -- infer the intended execution platform.
  2. Component identification  -- extract the pipeline's building blocks
                                   using a standardized taxonomy.
  3. Flow structure analysis   -- assemble components into an execution
                                   graph (entry points, edges, parallel
                                   sections).
  4. Parameter extraction      -- global / component-scoped / environment
                                   parameters, each with type, default,
                                   requirement status, constraints.
  5. Integration analysis      -- external systems each component talks
                                   to, plus the pipeline's primary data
                                   sources and sinks.

The five outputs compose into one versioned JSON artifact
(`PipelineAnalysis`), mirroring the paper's "analysis metadata, pipeline
summary, component definitions, flow structure with parallel patterns,
scoped parameters, and integration specifications" (Section 5.1).

This taxonomy (ComponentType) is intentionally separate from
`plan.matching.roles.ROLES`: that module is a deterministic, structural
tagger used to build a template-matching signature (Stage 2/3
boundary); this one is an LLM-driven classification produced *during*
analysis, using the paper's own vocabulary, and is meant to be read by
a human reviewing the analysis artifact -- the two do not need to
agree term-for-term.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from orchestrator.api.schemas.common import ComponentType, PlanBranch

PIPELINE_ANALYSIS_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Sub-stage 1.1: Action clause enumeration (NOT in the original paper --
# added specifically to defend component identification against
# under-extraction on compound requests).
# ---------------------------------------------------------------------------


class ActionClauseResult(BaseModel):
    """Output of a narrow, single-purpose pass over the raw request:
    list every distinct action clause (verb + object), nothing else --
    no AWS knowledge, no grouping into components, no service names.

    This runs as its OWN call, strictly BEFORE component identification,
    on purpose: if the same call that decides the final component list
    were also asked to self-report a checklist, an under-extracting
    model would tend to under-report the checklist too, since it's the
    same generation optimizing toward the same (too-compressed) picture
    of the request -- the checklist would just be gamed into agreeing
    with whatever the model already decided. Doing it as an isolated
    earlier pass, whose only job is enumeration, and asking it to
    account for none of the downstream structuring or grouping,
    prevents that: it has nothing to reconcile against yet. This mirrors
    the paper's own "narrower objectives improve accuracy" reasoning for
    prompt chaining generally, applied one level earlier than the paper
    itself splits things.
    """

    clauses: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "One entry per distinct action clause in the request, in "
            "the order they appear. Two clauses joined by 'and' under a "
            "parallel/concurrent verb are always two entries."
        ),
    )


# ---------------------------------------------------------------------------
# Sub-stage 1: Environment inference (paper Section 5.1, Appendix C.1)
# ---------------------------------------------------------------------------


class EnvironmentInference(BaseModel):
    """Target platform is fixed for this product (AWS Step Functions --
    there is only one downstream code generator, asl_service.py), but
    execution characteristics are still inferred from cues in the
    description, which is what actually varies pipeline to pipeline and
    what the paper's environment-inference stage is for.
    """

    target_platform: Literal["aws_step_functions"] = "aws_step_functions"
    state_machine_type: Literal["STANDARD", "EXPRESS"] = Field(
        ...,
        description=(
            "STANDARD for long-running, human-in-the-loop, exactly-once, or "
            "auditable workflows. EXPRESS for high-volume, short-duration "
            "(<5 min), at-least-once workloads."
        ),
    )
    execution_mode: Literal["synchronous", "asynchronous"] = Field(
        ...,
        description=(
            "synchronous: the caller waits for the workflow result (e.g. an "
            "API request). asynchronous: fire-and-forget (e.g. an event, "
            "schedule, or upload trigger)."
        ),
    )
    rationale: str = Field(
        ...,
        description=(
            "One sentence citing the specific cue(s) in the description that "
            "justify state_machine_type and execution_mode."
        ),
    )


# ---------------------------------------------------------------------------
# Sub-stage 2: Component identification (paper Section 5.1, Appendix C.2)
# ---------------------------------------------------------------------------
# ComponentType itself now lives in schemas/common.py (imported above) so
# that PlanStep can carry it as a first-class field without a circular
# import -- see that module for the full rationale.


class ComponentDefinition(BaseModel):
    id: str = Field(..., description="Short unique identifier, e.g. 'ValidateFile'.")
    service: str = Field(
        ...,
        description=(
            "AWS service this step invokes, e.g. 'Lambda', 'SNS'. 'None' for "
            "pure control/terminal components that call no AWS API."
        ),
    )
    description: str = Field(..., description="One sentence: what this component does.")
    component_type: ComponentType = Field(
        ...,
        description=(
            "Taxonomy classification. Data-centric: DataLoader (ingestion), "
            "Transformer (manipulation), Reconciliator (standardization/entity "
            "linking), Enricher (augmentation from another source). Output: "
            "Exporter, QualityCheck. Control-flow: Splitter, Merger, "
            "Orchestrator (Parallel/Map container), DecisionGate (Choice), "
            "Terminal (Succeed/Fail). Other: Notifier, ExternalCall, Other."
        ),
    )


class ComponentIdentificationResult(BaseModel):
    """Sub-stage 2 output. Same ready/needs_clarification contract as the
    rest of the extraction pipeline -- this is the ONE point in the
    five-stage chain where the structure of the workflow can still be
    genuinely undecidable (later stages only add detail to an already-
    fixed component list, so they never need to ask a structural
    question).

    Deliberately does NOT self-report a completeness checklist here --
    that's supplied as INPUT from the independent ActionClauseResult
    pass (sub-stage 1.1) instead, specifically so the completeness check
    isn't the same generation grading its own homework. See
    ActionClauseResult's docstring for why.
    """

    status: Literal["ready", "needs_clarification"]
    components: list[ComponentDefinition] | None = Field(default=None, validate_default=True)
    clarifying_question: str | None = Field(default=None, validate_default=True)

    @field_validator("components")
    @classmethod
    def components_required_when_ready(cls, v, info):
        if info.data.get("status") == "ready" and not v:
            raise ValueError("components is required when status is 'ready'")
        return v

    @field_validator("clarifying_question")
    @classmethod
    def question_required_when_clarifying(cls, v, info):
        if info.data.get("status") == "needs_clarification" and not v:
            raise ValueError(
                "clarifying_question is required when status is 'needs_clarification'"
            )
        return v


# ---------------------------------------------------------------------------
# Sub-stage 3: Flow structure analysis (paper Section 5.1, Appendix C.3)
# ---------------------------------------------------------------------------


class FlowStep(BaseModel):
    """Same control-flow shape as PlanStep (schemas/common.py) plus the
    component_type carried over unchanged from sub-stage 2, so the flow
    artifact is self-contained without a second lookup against the
    component list.
    """

    id: str
    service: str
    description: str
    component_type: ComponentType
    next: str | None = None
    on_failure: list[str] = Field(default_factory=list)
    on_timeout: list[str] = Field(default_factory=list)
    branches: list[PlanBranch] = Field(default_factory=list)


class FlowStructure(BaseModel):
    pipeline_name: str = Field(..., description="Proposed short kebab-case workflow name.")
    pipeline_summary: str = Field(
        ..., description="One or two sentence description of the whole workflow."
    )
    entry_points: list[str] = Field(
        ...,
        description="Ids of components with no incoming edge -- where execution enters the graph.",
    )
    parallel_sections: list[list[str]] = Field(
        default_factory=list,
        description=(
            "Groups of component ids that execute concurrently inside the same "
            "fan-out/Parallel construct, one list per construct."
        ),
    )
    steps: list[FlowStep] = Field(..., min_length=1)


# ---------------------------------------------------------------------------
# Sub-stage 4: Parameter extraction (paper Section 5.1, Appendix C.4)
# ---------------------------------------------------------------------------
# Optional merged sub-stage 2+3: components + flow in one LLM call
# (latency optimization — see MERGE_COMPONENT_AND_FLOW)
# ---------------------------------------------------------------------------


class CombinedComponentsAndFlow(BaseModel):
    """Single-call alternative to ComponentIdentificationResult then
    FlowStructure. Same ready/needs_clarification gate as component
    identification; when ready, both components and flow are required.
    """

    status: Literal["ready", "needs_clarification"]
    components: list[ComponentDefinition] | None = Field(default=None, validate_default=True)
    clarifying_question: str | None = Field(default=None, validate_default=True)
    # Populated only when status == "ready"
    pipeline_name: str | None = None
    pipeline_summary: str | None = None
    entry_points: list[str] | None = None
    parallel_sections: list[list[str]] | None = None
    steps: list[FlowStep] | None = None

    @field_validator("components")
    @classmethod
    def components_required_when_ready(cls, v, info):
        if info.data.get("status") == "ready" and not v:
            raise ValueError("components is required when status is 'ready'")
        return v

    @field_validator("clarifying_question")
    @classmethod
    def question_required_when_clarifying(cls, v, info):
        if info.data.get("status") == "needs_clarification" and not v:
            raise ValueError(
                "clarifying_question is required when status is 'needs_clarification'"
            )
        return v

    @field_validator("steps")
    @classmethod
    def steps_required_when_ready(cls, v, info):
        if info.data.get("status") == "ready" and not v:
            raise ValueError("steps is required when status is 'ready'")
        return v


# ---------------------------------------------------------------------------

ParameterScope = Literal["global", "component", "environment"]
ParameterDataType = Literal["string", "number", "boolean", "list", "object"]


class ParameterSpec(BaseModel):
    name: str = Field(..., description="e.g. 'TODO_SNS_TOPIC_ARN', 'RETRY_COUNT'.")
    scope: ParameterScope = Field(
        ...,
        description=(
            "global: affects the entire pipeline (e.g. a shared data directory). "
            "component: configures one specific component. environment: a "
            "runtime environment variable (e.g. AWS_REGION)."
        ),
    )
    component_id: str | None = Field(
        default=None,
        description="Set when scope == 'component': the id of the component this parameter configures.",
    )
    data_type: ParameterDataType
    # Was `str | None`. The model is (correctly) told to make `default` agree
    # with `data_type` -- a "number" parameter naturally gets a JSON number
    # default (e.g. ARCHIVE_LOOKBACK_HOURS -> 24), a "boolean" one gets a JSON
    # bool. A str-only type rejected every one of those, forcing a guaranteed
    # extra round trip through the repair loop (resending the *entire*
    # parameter list) just to re-quote numbers as strings -- seen in practice
    # taking longer than the original call itself. Widening the type to match
    # what data_type actually implies removes that failure mode entirely.
    default: str | int | float | bool | None = Field(
        default=None, description="Default value if stated or clearly implied."
    )
    required: bool
    constraints: str | None = Field(
        default=None,
        description="Validation rule mentioned or implied, e.g. 'must be > 0', 'one of: pass, fail'.",
    )
    description: str

    @field_validator("component_id")
    @classmethod
    def component_id_required_when_scoped(cls, v, info):
        if info.data.get("scope") == "component" and not v:
            raise ValueError("component_id is required when scope is 'component'")
        return v


class ParameterExtractionResult(BaseModel):
    parameters: list[ParameterSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sub-stage 5: Integration analysis (paper Section 5.1, Appendix C.5)
# ---------------------------------------------------------------------------

IntegrationKind = Literal["api", "database", "file_system", "message_queue", "cloud_service"]
IntegrationDirection = Literal["inbound", "outbound", "bidirectional"]


class IntegrationPoint(BaseModel):
    name: str = Field(..., description="e.g. 'orders DynamoDB table', 'payments SNS topic'.")
    kind: IntegrationKind
    service: str = Field(..., description="AWS service or external system name.")
    connection_details: str = Field(
        ..., description="Endpoint, table/queue/topic name, or a TODO_ placeholder if not stated."
    )
    auth_requirement: str | None = Field(
        default=None, description="Auth mechanism if mentioned or implied, e.g. 'IAM role', 'API token'."
    )
    direction: IntegrationDirection
    components: list[str] = Field(..., description="Ids of components that talk to this integration.")


class IntegrationAnalysisResult(BaseModel):
    integrations: list[IntegrationPoint] = Field(default_factory=list)
    primary_sources: list[str] = Field(
        default_factory=list, description="Component ids that are the pipeline's original data sources."
    )
    primary_sinks: list[str] = Field(
        default_factory=list, description="Component ids that are the pipeline's ultimate data sinks."
    )


# ---------------------------------------------------------------------------
# Composite artifact -- the paper's versioned JSON artifact.
# ---------------------------------------------------------------------------


class PipelineAnalysis(BaseModel):
    """Stage 1 output in full. This is what gets hashed/cached, shown to
    the user for transparency, and handed to Stage 2 (Structured
    Workflow Generation / enrichment_service.py) -- though today only
    its derived Plan (see pipeline_analysis_service._flow_to_plan) is
    actually consumed downstream; parameters/integrations/environment
    are carried for audit and future Stage 2 use, per the paper's own
    "beyond these primary outputs, the system generates auxiliary
    artifacts for debugging, validation, and reproducibility."
    """

    schema_version: str = PIPELINE_ANALYSIS_SCHEMA_VERSION
    pipeline_name: str
    pipeline_summary: str
    environment: EnvironmentInference
    components: list[ComponentDefinition]
    flow: FlowStructure
    parameters: list[ParameterSpec]
    integrations: IntegrationAnalysisResult