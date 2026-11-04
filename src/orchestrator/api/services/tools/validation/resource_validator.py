"""Validate that a resource reference is real and usable in a workflow."""

from __future__ import annotations

from typing import Any

from ..aws.allowlist import canonical_service
from ..aws.client import get_default_region
from ..aws.resource_getter import get_resource
from ..aws.resolver import CONFIDENT_MATCH, resolve_identifier


def validate_resource_reference(
    service: str,
    identifier: str,
    *,
    region: str | None = None,
    expected_type: str | None = None,
) -> dict[str, Any]:
    """Check whether identifier resolves to a live resource.

    Policy aligned with resolve_identifier:
      - valid=True only when a confident, unambiguous match exists
      - ambiguous matches → valid=False with alternatives in suggestions
    """
    svc = canonical_service(service)
    region = region or get_default_region()
    identifier = (identifier or "").strip()
    issues: list[str] = []
    suggestions: list[str] = []

    if not identifier:
        return {
            "valid": False,
            "service": svc,
            "identifier": identifier,
            "resolved": None,
            "arn": None,
            "issues": ["empty_identifier"],
            "suggestions": [],
            "confidence": 0.0,
        }

    # Fast path: exact get
    detail = get_resource(svc, identifier, region=region, include=["config", "tags"])
    confidence = 1.0 if detail else 0.0

    if detail is None:
        resolution = resolve_identifier(svc, identifier, region=region)
        if resolution.get("ambiguous"):
            issues.append("ambiguous_matches")
            for alt in resolution.get("alternatives") or []:
                r = alt.get("resource") or {}
                suggestions.append(
                    f"{r.get('name') or r.get('id')} (confidence={alt.get('confidence')})"
                )
            return {
                "valid": False,
                "service": svc,
                "identifier": identifier,
                "resolved": None,
                "arn": None,
                "issues": issues,
                "suggestions": [s for s in suggestions if s],
                "confidence": float(resolution.get("confidence") or 0.0),
            }

        if resolution.get("matched") and resolution.get("resource"):
            best = resolution["resource"]
            detail = get_resource(
                svc, best.get("id") or best.get("name") or identifier, region=region
            )
            confidence = float(resolution.get("confidence") or 0.0)
            if confidence < CONFIDENT_MATCH:
                issues.append("low_confidence_match")
        else:
            issues.append(resolution.get("reason") or "not_found")
            for alt in (resolution.get("alternatives") or [])[:3]:
                r = alt.get("resource") or {}
                suggestions.append(r.get("name") or r.get("id") or "")
            return {
                "valid": False,
                "service": svc,
                "identifier": identifier,
                "resolved": None,
                "arn": None,
                "issues": issues,
                "suggestions": [s for s in suggestions if s],
                "confidence": float(resolution.get("confidence") or 0.0),
            }

    if expected_type and detail and detail.resource_type.lower() != expected_type.lower():
        issues.append(
            f"type_mismatch: expected '{expected_type}', got '{detail.resource_type}'"
        )

    if detail and svc == "lambda":
        runtime = (detail.summary or {}).get("runtime") or (detail.config or {}).get("Runtime")
        if not runtime and not (detail.config or {}).get("PackageType"):
            issues.append("lambda_missing_runtime_info")

    blocking = [i for i in issues if i not in ("lambda_missing_runtime_info",)]
    valid = len(blocking) == 0 and detail is not None

    return {
        "valid": valid,
        "service": svc,
        "identifier": identifier,
        "resolved": detail.model_dump() if detail else None,
        "arn": detail.arn if detail else None,
        "issues": issues,
        "suggestions": suggestions,
        "confidence": confidence,
    }
