"""Pydantic models for tool inputs, outputs, and internal resource records.

Every tool returns a stable, versioned JSON shape so downstream stages
(pipeline analysis, enrichment, ASL generation) can rely on the contract
without re-parsing free-form text.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared resource model (normalised across all AWS services)
# ---------------------------------------------------------------------------


class ResourceSummary(BaseModel):
    """Normalised view of any AWS resource the tools can discover."""

    service: str = Field(description="AWS service id, e.g. 'lambda', 's3', 'dynamodb'")
    resource_type: str = Field(description="e.g. 'function', 'bucket', 'table', 'queue'")
    id: str = Field(description="Primary identifier used by the service (name or id)")
    name: str = Field(description="Human-readable name (often same as id)")
    arn: str | None = Field(default=None, description="Canonical ARN when available")
    region: str = Field(description="AWS region")
    account_id: str | None = Field(default=None)
    tags: dict[str, str] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Service-specific lightweight attributes (runtime, status, etc.)",
    )
    last_modified: str | None = Field(
        default=None,
        description="ISO-8601 timestamp when known",
    )


class ResourceDetail(ResourceSummary):
    """Full detail returned by get_resource."""

    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Raw or lightly-normalised configuration payload",
    )
    code_location: dict[str, Any] | None = Field(
        default=None,
        description="For compute resources: S3 location, image URI, etc.",
    )
    related: list[ResourceSummary] = Field(
        default_factory=list,
        description="Linked resources (e.g. Lambda → SQS event source)",
    )


# ---------------------------------------------------------------------------
# Tool input models
# ---------------------------------------------------------------------------


class ListResourcesInput(BaseModel):
    service: str = Field(
        description="AWS service identifier (lambda, s3, dynamodb, sqs, sns, states, glue, ecs, events, ...)"
    )
    resource_type: str | None = Field(
        default=None,
        description="Optional subtype filter (function, bucket, table, queue, topic, stateMachine, job, ...)",
    )
    region: str | None = Field(default=None, description="Override default region")
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional filters. Common keys: name_contains, name_prefix, tags "
            "(object of tag-key → value), status, runtime."
        ),
    )
    max_results: int = Field(default=50, ge=1, le=200)


class GetResourceInput(BaseModel):
    service: str
    identifier: str = Field(
        description="Name, ARN, or partial identifier of the resource"
    )
    region: str | None = None
    include: list[str] = Field(
        default_factory=lambda: ["config", "tags"],
        description="What to include: config, tags, code_location, policy, related",
    )


class SearchResourcesInput(BaseModel):
    query: str = Field(description="Free-text or keyword search across resource names/tags")
    services: list[str] | None = Field(
        default=None,
        description="Limit search to these services; omit for all allow-listed services",
    )
    tags: dict[str, str] = Field(default_factory=dict)
    region: str | None = None
    max_results: int = Field(default=20, ge=1, le=100)


class ResolveIdentifierInput(BaseModel):
    service: str
    hint: str = Field(
        description="Human name, partial name, ARN fragment, or alias"
    )
    region: str | None = None
    context: str | None = Field(
        default=None,
        description="Optional workflow context to improve ranking (e.g. 'payment pipeline')",
    )


class ValidateResourceReferenceInput(BaseModel):
    service: str
    identifier: str
    region: str | None = None
    expected_type: str | None = Field(
        default=None,
        description="If set, also verify the resource is of this type",
    )


class DescribeServiceInput(BaseModel):
    service: str = Field(description="Service id to describe")


class InvokeAwsOperationInput(BaseModel):
    service: str
    operation: str = Field(
        description="boto3 operation name (must be allow-listed: list*, describe*, get*)"
    )
    parameters: dict[str, Any] = Field(default_factory=dict)
    region: str | None = None


class ListAwsServicesInput(BaseModel):
    region: str | None = None
    category: str | None = Field(
        default=None,
        description="Optional filter: compute | storage | messaging | orchestration | analytics | integration",
    )


# ---------------------------------------------------------------------------
# Deterministic / zero-LLM tool inputs (no caching; stable shapes)
# ---------------------------------------------------------------------------


class NormalizePromptInput(BaseModel):
    text: str = Field(description="Raw user workflow description")


class ExtractEntitiesInput(BaseModel):
    text: str = Field(description="Text to scan for resource names and ARNs")


class ClassifyWorkflowPatternInput(BaseModel):
    text: str = Field(
        description="User text (preferably normalized) to classify into a closed pattern set"
    )


class AssignCanonicalStepIdsInput(BaseModel):
    plan: dict[str, Any] = Field(
        description="Plan object with a 'steps' array (id, service, description, next, branches, ...)"
    )


class SuggestPlaceholderNamesInput(BaseModel):
    entities: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Entities from extract_entities (kind/value/service_hint)",
    )
    roles: list[str] = Field(default_factory=list)
    services: list[str] = Field(default_factory=list)


class GetServiceIntegrationContractInput(BaseModel):
    service: str | None = Field(
        default=None,
        description="AWS service id (lambda, s3, dynamodb, ...). Omit to list all contracts.",
    )


class GroundResourcesInput(BaseModel):
    text: str = Field(description="Workflow description to ground against live AWS")
    services: list[str] | None = Field(
        default=None,
        description="Optional service filter; when omitted, inferred from text",
    )
    region: str | None = None
    session_id: str | None = None
    max_per_service: int = Field(default=15, ge=1, le=50)


class ValidatePlanAgainstAwsInput(BaseModel):
    plan: dict[str, Any] = Field(description="Plan with steps to validate")
    region: str | None = None
    session_id: str | None = None


# ---------------------------------------------------------------------------
# Tool output envelopes
# ---------------------------------------------------------------------------



class SearchRepoContextInput(BaseModel):
    query: str = Field(description="Free-text query over the indexed repository")
    repo_id: str | None = Field(default=None, description="Optional repo scope")
    top_k: int = Field(default=8, ge=1, le=30)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)


class ResolveRepoResourceInput(BaseModel):
    hint: str = Field(description="Job name, Lambda name, or path fragment")
    kind: str | None = Field(
        default=None,
        description="Optional kind filter: lambda, glue, pipeline, sns, sqs, s3, dynamodb",
    )
    repo_id: str | None = None


class ListRepoArtifactsInput(BaseModel):
    kind: str | None = Field(default=None, description="Optional kind filter")
    repo_id: str | None = None
    max_results: int = Field(default=40, ge=1, le=100)


class ValidatePlanAgainstRepoInput(BaseModel):
    plan: dict[str, Any] = Field(
        description="Workflow plan with steps to ground against the repo index"
    )
    repo_id: str | None = Field(default=None, description="Optional repo scope")
    require_all: bool = Field(
        default=False,
        description="If true, unmatched tokens make the result not ok",
    )


class ToolStatus(str, Enum):
    OK = "ok"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    FORBIDDEN = "forbidden"
    ERROR = "error"
    CACHE_HIT = "cache_hit"


class ToolResult(BaseModel):
    """Uniform envelope returned by every tool execution."""

    tool: str
    status: ToolStatus
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    error_code: str | None = None
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="latency_ms, cache_hit, region, request_id, etc.",
    )
    telemetry_id: str | None = None

    def to_llm_content(self) -> str:
        """Compact JSON string suitable for injection into an LLM message."""
        payload = {
            "status": self.status.value,
            "data": self.data,
        }
        if self.error:
            payload["error"] = self.error
        return __import__("json").dumps(payload, default=str, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Service catalogue (static knowledge used by describe_service / list_services)
# ---------------------------------------------------------------------------


class ServiceInfo(BaseModel):
    id: str
    name: str
    category: str
    resource_types: list[str]
    common_operations: list[str]
    identifier_fields: list[str] = Field(
        description="Fields commonly used as the primary key (FunctionName, Bucket, TableName, ...)"
    )
    supports_tags: bool = True
    notes: str | None = None
