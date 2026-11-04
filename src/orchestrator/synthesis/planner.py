"""Repo-grounded, consistency-aware workflow plan synthesis.

Pipeline (enterprise path):
  1. Canonicalize request (paraphrase collapse)
  2. Optional plan cache hit → return identical plan for same intent
  3. Deterministic context (normalize / entities / pattern)
  4. Repo RAG retrieval + live AWS discovery
  5. Pipeline analysis (LLM) → Plan
  6. Deterministic plan matching (template snap when high confidence)
  7. Canonical step ids
  8. Consistency report attached to result
"""

from __future__ import annotations

import logging
from typing import Any

from orchestrator.synthesis.models import PlanResult, SynthesisContext

logger = logging.getLogger(__name__)


def _build_context(
    user_message: str,
    *,
    history: list[dict] | None = None,
    session_id: str | None = None,
    current_plan: dict[str, Any] | None = None,
    use_cache: bool = True,
) -> SynthesisContext:
    from orchestrator.api.services.consistency import (
        canonicalize_request,
        request_fingerprint,
    )

    text = (user_message or "").strip()
    ctx = SynthesisContext(
        user_message=text,
        canonical_request=canonicalize_request(text),
        request_fingerprint=request_fingerprint(text),
        history=list(history or []),
        current_plan=current_plan,
        session_id=session_id,
    )
    if not ctx.history and text:
        ctx.history = [{"role": "user", "content": text}]

    try:
        from orchestrator.api.services.tools.context import run_discovery_safe

        ctx.discovery = run_discovery_safe(text, session_id=session_id) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("discovery failed: %s", exc)
        ctx.discovery = {}

    try:
        from orchestrator.api.services.tools.repo import build_repo_context_bundle

        ctx.repo_context = build_repo_context_bundle(text) or {}
        ctx.repo_context_markdown = ctx.repo_context.get("context") or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning("repo context failed: %s", exc)
        ctx.repo_context = {}

    if ctx.repo_context.get("hits") or ctx.repo_context.get("resolved"):
        ctx.discovery = dict(ctx.discovery)
        ctx.discovery["repo_hits"] = ctx.repo_context.get("hits") or []
        ctx.discovery["repo_resolved"] = ctx.repo_context.get("resolved") or []
        ctx.discovery["repo_context"] = ctx.repo_context_markdown
        ctx.discovery["repo_enabled"] = bool(ctx.repo_context.get("enabled"))

    return ctx


def _inject_repo_into_history(
    history: list[dict],
    repo_context: dict[str, Any],
) -> list[dict]:
    if not repo_context:
        return history

    lines = [
        "REPOSITORY CONTEXT (retrieved from project index — prefer these real "
        "job names, entry points, and scripts over invented ones. If a named "
        "resource is not listed, use a TODO_ placeholder.)",
    ]
    resolved = repo_context.get("resolved") or []
    if resolved:
        lines.append("## Resolved resources")
        for r in resolved[:12]:
            art = r.get("artifact") or {}
            lines.append(
                f"- {r.get('token')}: name={art.get('resource_name')} "
                f"type={art.get('resource_type') or art.get('kind')} "
                f"entry={art.get('entry_point') or '-'} path={art.get('path') or '-'}"
            )
    hits = repo_context.get("hits") or []
    if hits and not resolved:
        lines.append("## Top retrieved artifacts")
        for h in hits[:8]:
            lines.append(
                f"- {h.get('resource_name') or h.get('id')}: "
                f"type={h.get('resource_type') or h.get('kind')} "
                f"path={h.get('path') or '-'}"
            )
    md = repo_context.get("context") or ""
    if md:
        lines.append("## Retrieved excerpts")
        lines.append(md[:3500])

    block = "\n".join(lines)
    if len(block) < 80:
        return history
    return [{"role": "system", "content": block}, *history]


def _apply_matching(
    plan_dict: dict[str, Any], ctx: SynthesisContext
) -> tuple[dict[str, Any], str | None, str | None]:
    try:
        from orchestrator.api.schemas.common import Plan
        from orchestrator.api.services.plan.matching.matcher import match

        plan = Plan.model_validate(plan_dict)
        result = match(
            plan,
            session_id=ctx.session_id,
            user_message=ctx.user_message,
        )
        if result.plan is not None and result.tier in ("high", "medium"):
            return result.plan.model_dump(), result.tier, result.template_name
        return plan_dict, result.tier, result.template_name
    except Exception as exc:  # noqa: BLE001
        logger.warning("plan matching skipped: %s", exc)
        return plan_dict, None, None


def _canonical_step_ids(plan_dict: dict[str, Any]) -> dict[str, Any]:
    try:
        from orchestrator.api.services.tools.deterministic import (
            handle_assign_canonical_step_ids,
        )

        out = handle_assign_canonical_step_ids({"plan": plan_dict})
        if isinstance(out.get("plan"), dict):
            return out["plan"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("canonical step ids skipped: %s", exc)
    return plan_dict


def generate_plan(
    user_message: str,
    *,
    history: list[dict] | None = None,
    session_id: str | None = None,
    current_plan: dict[str, Any] | None = None,
    use_cache: bool = True,
    self_consistency: bool = False,
) -> PlanResult:
    """Main entry: natural language → structured workflow plan."""
    from orchestrator.api.services.consistency import (
        build_consistency_report,
        plan_structural_hash,
    )

    text = (user_message or "").strip()
    if not text and not (history or []):
        return PlanResult(
            status="error",
            error="empty_request",
            message="Provide a workflow description.",
        )

    # Plan/prompt caching disabled — every request is synthesised fresh.
    _ = use_cache

    ctx = _build_context(
        text or _last_user_content(history),
        history=history,
        session_id=session_id,
        current_plan=current_plan,
        use_cache=use_cache,
    )

    grounded_history = _inject_repo_into_history(ctx.history, ctx.repo_context)

    try:
        from orchestrator.api.services.pipeline_analysis.service import analyze_pipeline

        extraction = analyze_pipeline(grounded_history, discovery=ctx.discovery)
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        if "Extraction" in name:
            return PlanResult(
                status="error",
                error=str(exc),
                message="Pipeline analysis failed after repair attempts.",
            )
        logger.exception("analyze_pipeline failed")
        return PlanResult(status="error", error=f"{name}: {exc}")

    status = getattr(extraction, "status", None) or (
        extraction.get("status") if isinstance(extraction, dict) else None
    )
    if status == "out_of_scope":
        return PlanResult(
            status="out_of_scope",
            message=getattr(extraction, "message", None)
            or (extraction.get("message") if isinstance(extraction, dict) else None),
        )
    if status == "needs_clarification":
        return PlanResult(
            status="needs_clarification",
            clarifying_question=getattr(extraction, "clarifying_question", None)
            or (
                extraction.get("clarifying_question")
                if isinstance(extraction, dict)
                else None
            ),
        )

    plan_obj = getattr(extraction, "plan", None)
    if plan_obj is not None and hasattr(plan_obj, "model_dump"):
        plan_dict = plan_obj.model_dump()
    elif isinstance(plan_obj, dict):
        plan_dict = plan_obj
    elif isinstance(extraction, dict) and extraction.get("plan"):
        plan_dict = extraction["plan"]
    else:
        return PlanResult(
            status="error",
            error="no_plan",
            message="Analysis returned ready but no plan.",
        )

    analysis = getattr(extraction, "analysis", None)
    if analysis is None and isinstance(extraction, dict):
        analysis = extraction.get("analysis")

    plan_dict, tier, template_name = _apply_matching(plan_dict, ctx)
    plan_dict = _canonical_step_ids(plan_dict)

    alternate = None
    self_ok = None
    if self_consistency:
        try:
            from orchestrator.api.services.pipeline_analysis.service import (
                analyze_pipeline as _ap,
            )

            second = _ap(grounded_history, discovery=ctx.discovery)
            alt_plan = getattr(second, "plan", None)
            if alt_plan is not None and hasattr(alt_plan, "model_dump"):
                alternate = alt_plan.model_dump()
            elif isinstance(alt_plan, dict):
                alternate = alt_plan
            if alternate:
                alternate = _canonical_step_ids(alternate)
                self_ok = plan_structural_hash(plan_dict) == plan_structural_hash(
                    alternate
                )
                if not self_ok and tier not in ("high", "medium"):
                    return PlanResult(
                        status="needs_clarification",
                        clarifying_question=(
                            "I got two different workflow structures for the same "
                            "request. Please confirm the main steps in order "
                            "(e.g. validate → charge → ship), or name the jobs "
                            "from your repository that should be included."
                        ),
                        consistency=build_consistency_report(
                            ctx.user_message,
                            plan_dict,
                            discovery=ctx.discovery,
                            match_tier=tier,
                            matched_template=template_name,
                            alternate_plan=alternate,
                        ).to_dict(),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("self-consistency pass failed: %s", exc)

    report = build_consistency_report(
        ctx.user_message,
        plan_dict,
        discovery=ctx.discovery,
        match_tier=tier,
        matched_template=template_name,
        alternate_plan=alternate,
    )
    if self_ok is not None:
        report.self_consistent = self_ok

    return PlanResult(
        status="ready",
        plan=plan_dict,
        analysis=analysis if isinstance(analysis, dict) else None,
        match_tier=tier,
        matched_template=template_name,
        consistency=report.to_dict(),
        from_cache=False,
    )


def _last_user_content(history: list[dict] | None) -> str:
    for m in reversed(history or []):
        if isinstance(m, dict) and m.get("role") == "user":
            return str(m.get("content") or "").strip()
    return ""
