"""Derive the user-facing workflow phase from session artifacts.

Phase is computed, not stored, so it cannot drift from plan / locked_spec.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class WorkflowPhase(str, Enum):
    COLLECTING = "collecting"  # need NL or clarifying question
    PLAN_REVIEW = "plan_review"  # plan ready, await approve/reject
    ENRICHING = "enriching"  # enrichment asked a question
    SPEC_PLACEHOLDERS = "spec_placeholders"  # TODO_ in workflow_spec
    ASL_REVIEW = "asl_review"  # ASL generated; may have validation issues
    DEPLOYED = "deployed"
    FAILED = "failed"


def _locked(session: Any) -> dict:
    locked = getattr(session, "locked_spec", None)
    return locked if isinstance(locked, dict) else {}


def derive_phase(session: Any) -> WorkflowPhase:
    """Single source of phase truth from session.latest_plan + locked_spec."""
    locked = _locked(session)
    deployment = locked.get("deployment") if isinstance(locked.get("deployment"), dict) else {}
    if deployment.get("state_machine_arn"):
        return WorkflowPhase.DEPLOYED

    if locked.get("enrichment_question") or locked.get("enrichment_pending"):
        return WorkflowPhase.ENRICHING

    asl = locked.get("asl")
    has_asl = isinstance(asl, dict) and bool(asl.get("States") or asl)
    spec_placeholders = locked.get("spec_placeholders") or []
    if isinstance(spec_placeholders, list) and spec_placeholders:
        return WorkflowPhase.SPEC_PLACEHOLDERS

    if has_asl:
        return WorkflowPhase.ASL_REVIEW

    if isinstance(locked.get("workflow_spec"), dict) and locked.get("workflow_spec"):
        # Spec locked but ASL not generated yet (placeholders cleared mid-path
        # or generation error). Treat as ASL review surface so user can retry.
        return WorkflowPhase.ASL_REVIEW

    if getattr(session, "latest_plan", None) is not None:
        return WorkflowPhase.PLAN_REVIEW

    if locked.get("enrichment_error") or locked.get("asl_error"):
        return WorkflowPhase.FAILED

    return WorkflowPhase.COLLECTING


def phase_label(phase: WorkflowPhase) -> str:
    return {
        WorkflowPhase.COLLECTING: "Collecting requirements",
        WorkflowPhase.PLAN_REVIEW: "Plan ready for review",
        WorkflowPhase.ENRICHING: "Needs enrichment answer",
        WorkflowPhase.SPEC_PLACEHOLDERS: "Fill workflow placeholders",
        WorkflowPhase.ASL_REVIEW: "ASL ready for review",
        WorkflowPhase.DEPLOYED: "Deployed",
        WorkflowPhase.FAILED: "Needs attention",
    }.get(phase, phase.value)
