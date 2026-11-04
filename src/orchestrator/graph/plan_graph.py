"""Enterprise LangGraph: natural language → Step Functions workflow plan.

Pipeline (every node is traced with OpenTelemetry):

  START
    → retrieve          (RAG over indexed repository — dual dense + structured)
    → discover_tools    (deterministic + AWS + repo tools; no hardcoded domains)
    → synthesize_plan   (LLM pipeline analysis grounded in retrieved context)
    → ground_plan       (match step tokens to real repo artifacts when available)
    → present           (summary + diagram + placeholders for the user)
    → END

No prompt/plan caching. No domain-specific hardcoded workflows. Works for any
NL description of an orchestration the model can express as steps/services.

Public API
----------
  nl_to_plan(user_message) -> dict   # preferred one-shot entry
  run_plan_graph(state) -> state
  build_plan_graph() -> compiled graph
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional, TypedDict

logger = logging.getLogger(__name__)


class PlanGraphState(TypedDict, total=False):
    """State flowing through the plan-generation graph."""

    # Input
    user_message: str
    history: list[dict]
    session_id: str

    # Retrieval
    repo_context: dict[str, Any]
    repo_context_markdown: str
    retrieval_hits: list[dict[str, Any]]

    # Tools / discovery
    discovery: dict[str, Any]
    tool_trace: list[dict[str, Any]]

    # Synthesis
    status: str  # ready | needs_clarification | out_of_scope | error
    plan: Optional[dict[str, Any]]
    analysis: Optional[dict[str, Any]]
    clarifying_question: Optional[str]
    message: Optional[str]
    match_tier: Optional[str]
    matched_template: Optional[str]

    # Grounding
    grounding: Optional[dict[str, Any]]

    # Presentation
    summary_markdown: Optional[str]
    diagram_dot: Optional[str]
    placeholders: list[str]

    # Control / observability
    error: Optional[str]
    stages: list[str]
    warnings: list[str]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def _append_stage(state: PlanGraphState, name: str) -> list[str]:
    stages = list(state.get("stages") or [])
    stages.append(name)
    return stages


def retrieve_node(state: PlanGraphState) -> dict[str, Any]:
    """RAG: dual dense + structured retrieval over the project index."""
    from orchestrator.observability.otel import span, trace_retrieval

    text = (state.get("user_message") or "").strip()
    with span("graph.retrieve", oaf_node="retrieve") as s:
        with trace_retrieval(query=text):
            repo_context: dict[str, Any] = {
                "enabled": False,
                "hits": [],
                "context": "",
                "resolved": [],
            }
            try:
                from orchestrator.api.services.tools.repo import build_repo_context_bundle

                repo_context = build_repo_context_bundle(text) or repo_context
            except Exception as exc:  # noqa: BLE001
                logger.warning("retrieve_node: %s", exc)
                try:
                    s.set_attribute("error", str(exc)[:200])
                except Exception:  # noqa: BLE001
                    pass

            hits = list(repo_context.get("hits") or [])
            try:
                s.set_attribute("retrieval.hit_count", len(hits))
                s.set_attribute(
                    "retrieval.enabled", bool(repo_context.get("enabled"))
                )
            except Exception:  # noqa: BLE001
                pass

            return {
                "repo_context": repo_context,
                "repo_context_markdown": repo_context.get("context") or "",
                "retrieval_hits": hits,
                "stages": _append_stage(state, "retrieve"),
            }


def discover_tools_node(state: PlanGraphState) -> dict[str, Any]:
    """Run the advanced tool layer (deterministic + AWS + repo) for this request.

    Tools are selected dynamically from the user text — no per-domain hardcoding.
    Failures are soft: empty discovery still allows plan synthesis.
    """
    from orchestrator.observability.otel import span, trace_tool

    text = (state.get("user_message") or "").strip()
    session_id = state.get("session_id")
    tool_trace: list[dict[str, Any]] = []
    discovery: dict[str, Any] = {}

    with span("graph.discover_tools", oaf_node="discover_tools"):
        # 1) Safe multi-tool discovery bundle (AWS + deterministic)
        try:
            with trace_tool("run_discovery_safe"):
                from orchestrator.api.services.tools.context import run_discovery_safe

                discovery = run_discovery_safe(text, session_id=session_id) or {}
                tool_trace.append(
                    {
                        "tool": "run_discovery_safe",
                        "ok": True,
                        "keys": sorted(discovery.keys())[:20],
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("discover_tools discovery: %s", exc)
            tool_trace.append(
                {"tool": "run_discovery_safe", "ok": False, "error": str(exc)[:200]}
            )
            discovery = {}

        # 2) Deterministic context (normalize / entities / pattern) — pure code
        try:
            with trace_tool("run_deterministic_context"):
                from orchestrator.api.services.tools.context import (
                    run_deterministic_context,
                )

                det = run_deterministic_context(text) or {}
                if det:
                    discovery = {**discovery, **{f"det_{k}": v for k, v in det.items()}}
                    # Prefer non-prefixed merge for common keys
                    for k in ("pattern", "services", "name_tokens", "entities"):
                        if k in det and k not in discovery:
                            discovery[k] = det[k]
                tool_trace.append({"tool": "run_deterministic_context", "ok": True})
        except Exception as exc:  # noqa: BLE001
            logger.warning("discover_tools deterministic: %s", exc)
            tool_trace.append(
                {
                    "tool": "run_deterministic_context",
                    "ok": False,
                    "error": str(exc)[:200],
                }
            )

        # 3) Fold RAG hits into discovery so synthesis sees one context bag
        repo = state.get("repo_context") or {}
        if repo.get("hits") or repo.get("resolved"):
            discovery = dict(discovery)
            discovery["repo_hits"] = repo.get("hits") or []
            discovery["repo_resolved"] = repo.get("resolved") or []
            discovery["repo_context"] = repo.get("context") or ""
            discovery["repo_enabled"] = bool(repo.get("enabled"))

        return {
            "discovery": discovery,
            "tool_trace": tool_trace,
            "stages": _append_stage(state, "discover_tools"),
        }


def _inject_context_into_history(
    history: list[dict],
    repo_context: dict[str, Any],
    discovery: dict[str, Any],
) -> list[dict]:
    """Prepend a system block with retrieved repo + tool findings (no hardcoding)."""
    blocks: list[str] = []

    resolved = (repo_context or {}).get("resolved") or []
    hits = (repo_context or {}).get("hits") or []
    md = (repo_context or {}).get("context") or ""

    if resolved or hits or md:
        lines = [
            "REPOSITORY CONTEXT (from project index). Prefer these real resource "
            "names, entry points, and scripts. Do not invent names that conflict "
            "with them. If a needed resource is absent, use a TODO_ placeholder.",
        ]
        if resolved:
            lines.append("## Resolved resources")
            for r in resolved[:15]:
                art = r.get("artifact") or {}
                lines.append(
                    f"- token={r.get('token')}: name={art.get('resource_name')} "
                    f"type={art.get('resource_type') or art.get('kind')} "
                    f"entry={art.get('entry_point') or '-'} path={art.get('path') or '-'}"
                )
        if hits and not resolved:
            lines.append("## Top retrieved artifacts")
            for h in hits[:10]:
                lines.append(
                    f"- {h.get('resource_name') or h.get('id')}: "
                    f"type={h.get('resource_type') or h.get('kind')} "
                    f"path={h.get('path') or '-'}"
                )
        if md:
            lines.append("## Retrieved excerpts")
            lines.append(md[:4000])
        blocks.append("\n".join(lines))

    # Tool discovery summary (services inferred, pattern, name tokens)
    if discovery:
        dlines = ["TOOL DISCOVERY (live AWS + deterministic analysis):"]
        for key in (
            "pattern",
            "services",
            "services_inferred",
            "name_tokens",
            "entities",
        ):
            val = discovery.get(key)
            if val:
                dlines.append(f"- {key}: {val}")
        # Resolved AWS bindings if present
        for key in ("bindings", "resolved", "resources"):
            val = discovery.get(key)
            if val:
                dlines.append(f"- {key}: {str(val)[:1500]}")
        if len(dlines) > 1:
            blocks.append("\n".join(dlines))

    if not blocks:
        return history

    system = {"role": "system", "content": "\n\n".join(blocks)}
    return [system, *history]


def synthesize_plan_node(state: PlanGraphState) -> dict[str, Any]:
    """LLM plan synthesis grounded in RAG + tool discovery. No response cache."""
    from orchestrator.observability.otel import span

    text = (state.get("user_message") or "").strip()
    history = list(state.get("history") or [])
    if not history and text:
        history = [{"role": "user", "content": text}]

    grounded = _inject_context_into_history(
        history,
        state.get("repo_context") or {},
        state.get("discovery") or {},
    )
    discovery = state.get("discovery") or {}
    warnings = list(state.get("warnings") or [])

    with span("graph.synthesize_plan", oaf_node="synthesize_plan") as s:
        try:
            from orchestrator.api.services.pipeline_analysis.service import (
                analyze_pipeline,
            )

            extraction = analyze_pipeline(grounded, discovery=discovery)
        except TypeError:
            # Older signature without discovery kwarg
            try:
                from orchestrator.api.services.pipeline_analysis.service import (
                    analyze_pipeline,
                )

                extraction = analyze_pipeline(grounded)
            except Exception as exc:  # noqa: BLE001
                logger.exception("synthesize_plan failed")
                return {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "stages": _append_stage(state, "synthesize_plan"),
                    "warnings": warnings,
                }
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            if "Extraction" in name:
                return {
                    "status": "error",
                    "error": str(exc),
                    "message": "Pipeline analysis failed after repair attempts.",
                    "stages": _append_stage(state, "synthesize_plan"),
                    "warnings": warnings,
                }
            logger.exception("synthesize_plan failed")
            return {
                "status": "error",
                "error": f"{name}: {exc}",
                "stages": _append_stage(state, "synthesize_plan"),
                "warnings": warnings,
            }

        status = getattr(extraction, "status", None) or (
            extraction.get("status") if isinstance(extraction, dict) else None
        )

        if status == "out_of_scope":
            msg = getattr(extraction, "message", None) or (
                extraction.get("message") if isinstance(extraction, dict) else None
            )
            return {
                "status": "out_of_scope",
                "message": msg,
                "stages": _append_stage(state, "synthesize_plan"),
                "warnings": warnings,
            }

        if status == "needs_clarification":
            q = getattr(extraction, "clarifying_question", None) or (
                extraction.get("clarifying_question")
                if isinstance(extraction, dict)
                else None
            )
            return {
                "status": "needs_clarification",
                "clarifying_question": q,
                "stages": _append_stage(state, "synthesize_plan"),
                "warnings": warnings,
            }

        plan_obj = getattr(extraction, "plan", None)
        if plan_obj is not None and hasattr(plan_obj, "model_dump"):
            plan_dict = plan_obj.model_dump()
        elif isinstance(plan_obj, dict):
            plan_dict = plan_obj
        elif isinstance(extraction, dict) and extraction.get("plan"):
            plan_dict = extraction["plan"]
        else:
            return {
                "status": "error",
                "error": "no_plan",
                "message": "Analysis returned ready but no plan.",
                "stages": _append_stage(state, "synthesize_plan"),
                "warnings": warnings,
            }

        analysis = getattr(extraction, "analysis", None)
        if analysis is None and isinstance(extraction, dict):
            analysis = extraction.get("analysis")

        # Optional template match (structural reuse — not a hardcode of domains)
        match_tier = None
        template_name = None
        try:
            from orchestrator.api.schemas.common import Plan
            from orchestrator.api.services.plan.matching.matcher import match

            plan_model = Plan.model_validate(plan_dict)
            result = match(
                plan_model,
                session_id=state.get("session_id"),
                user_message=text,
            )
            if result.plan is not None and result.tier in ("high", "medium"):
                plan_dict = result.plan.model_dump()
            match_tier = result.tier
            template_name = result.template_name
        except Exception as exc:  # noqa: BLE001
            logger.info("plan matching skipped: %s", exc)
            warnings.append(f"matching_skipped: {type(exc).__name__}")

        # Canonical step ids (deterministic, domain-agnostic)
        try:
            from orchestrator.api.services.tools.deterministic import (
                handle_assign_canonical_step_ids,
            )

            out = handle_assign_canonical_step_ids({"plan": plan_dict})
            if isinstance(out.get("plan"), dict):
                plan_dict = out["plan"]
        except Exception as exc:  # noqa: BLE001
            logger.info("canonical ids skipped: %s", exc)

        try:
            s.set_attribute("plan.step_count", len(plan_dict.get("steps") or []))
            if match_tier:
                s.set_attribute("plan.match_tier", match_tier)
        except Exception:  # noqa: BLE001
            pass

        return {
            "status": "ready",
            "plan": plan_dict,
            "analysis": analysis if isinstance(analysis, dict) else None,
            "match_tier": match_tier,
            "matched_template": template_name,
            "stages": _append_stage(state, "synthesize_plan"),
            "warnings": warnings,
        }


def ground_plan_node(state: PlanGraphState) -> dict[str, Any]:
    """Ground plan steps against repo index when retrieval is enabled."""
    from orchestrator.observability.otel import span

    if state.get("status") != "ready" or not state.get("plan"):
        return {"stages": _append_stage(state, "ground_plan_skipped")}

    with span("graph.ground_plan", oaf_node="ground_plan"):
        try:
            from orchestrator.synthesis.grounding import (
                apply_grounding_to_plan,
                ground_plan_against_repo,
            )

            report = ground_plan_against_repo(state["plan"], require_all=False)
            plan = state["plan"]
            if report.enabled and report.hits:
                plan = apply_grounding_to_plan(plan, report)
            warnings = list(state.get("warnings") or [])
            warnings.extend(report.notes)
            return {
                "plan": plan,
                "grounding": report.to_dict(),
                "warnings": warnings,
                "stages": _append_stage(state, "ground_plan"),
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("ground_plan: %s", exc)
            return {
                "warnings": list(state.get("warnings") or [])
                + [f"grounding_skipped: {type(exc).__name__}: {exc}"],
                "stages": _append_stage(state, "ground_plan"),
            }


def present_node(state: PlanGraphState) -> dict[str, Any]:
    """Build human-readable summary / diagram / placeholder list for the plan."""
    from orchestrator.observability.otel import span

    if state.get("status") != "ready" or not state.get("plan"):
        return {"stages": _append_stage(state, "present_skipped")}

    with span("graph.present", oaf_node="present"):
        summary = None
        diagram = None
        placeholders: list[str] = []
        try:
            from orchestrator.api.schemas.common import Plan
            from orchestrator.api.services.plan.render import (
                find_placeholders,
                render_plan_diagram,
                render_plan_summary,
            )

            plan = Plan.model_validate(state["plan"])
            summary = render_plan_summary(plan)
            diagram = render_plan_diagram(plan)
            placeholders = list(find_placeholders(plan) or [])
        except Exception as exc:  # noqa: BLE001
            logger.info("present render fallback: %s", exc)
            # Minimal fallback without schema dependency
            steps = (state.get("plan") or {}).get("steps") or []
            lines = [f"**{(state.get('plan') or {}).get('name') or 'Workflow'}**", ""]
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    continue
                lines.append(
                    f"{i + 1}. `{step.get('id')}` — "
                    f"{step.get('service') or 'Task'}: "
                    f"{step.get('description') or ''}"
                )
            summary = "\n".join(lines)

        return {
            "summary_markdown": summary,
            "diagram_dot": diagram,
            "placeholders": placeholders,
            "stages": _append_stage(state, "present"),
        }


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_plan_graph():
    """Compile the NL → plan LangGraph (no checkpointer — pure request/response)."""
    from langgraph.graph import END, START, StateGraph

    g: StateGraph = StateGraph(PlanGraphState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("discover_tools", discover_tools_node)
    g.add_node("synthesize_plan", synthesize_plan_node)
    g.add_node("ground_plan", ground_plan_node)
    g.add_node("present", present_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "discover_tools")
    g.add_edge("discover_tools", "synthesize_plan")
    g.add_edge("synthesize_plan", "ground_plan")
    g.add_edge("ground_plan", "present")
    g.add_edge("present", END)

    return g.compile()


_compiled_plan_graph = None


def get_plan_graph():
    global _compiled_plan_graph
    if _compiled_plan_graph is None:
        _compiled_plan_graph = build_plan_graph()
    return _compiled_plan_graph


def run_plan_graph(
    user_message: str,
    *,
    history: list[dict] | None = None,
    session_id: str | None = None,
) -> PlanGraphState:
    """Execute the full plan graph once and return final state."""
    from orchestrator.observability.otel import configure_otel, span

    configure_otel()
    text = (user_message or "").strip()
    sid = session_id or str(uuid.uuid4())
    hist = list(history or [])
    if not hist and text:
        hist = [{"role": "user", "content": text}]

    initial: PlanGraphState = {
        "user_message": text,
        "history": hist,
        "session_id": sid,
        "stages": [],
        "warnings": [],
        "placeholders": [],
    }

    with span(
        "graph.nl_to_plan",
        oaf_entry="run_plan_graph",
        session_id=sid,
        user_message_len=len(text),
    ):
        graph = get_plan_graph()
        return graph.invoke(initial)


def nl_to_plan(
    user_message: str,
    *,
    history: list[dict] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Primary enterprise entry: any natural-language request → plan package.

    Returns a plain dict suitable for API / CLI / UI:

      {
        "status": "ready" | "needs_clarification" | "out_of_scope" | "error",
        "plan": {...} | null,
        "summary_markdown": "...",
        "diagram_dot": "...",
        "placeholders": [...],
        "grounding": {...},
        "clarifying_question": "...",
        "message": "...",
        "error": "...",
        "stages": [...],
        "warnings": [...],
        "session_id": "...",
        "retrieval_hits": [...],
        "tool_trace": [...],
      }

    No caching. No domain hardcoding. Accuracy comes from RAG + tools + LLM
    grounded on retrieved context.
    """
    state = run_plan_graph(
        user_message, history=history, session_id=session_id
    )
    return {
        "status": state.get("status") or "error",
        "plan": state.get("plan"),
        "analysis": state.get("analysis"),
        "summary_markdown": state.get("summary_markdown"),
        "diagram_dot": state.get("diagram_dot"),
        "placeholders": list(state.get("placeholders") or []),
        "grounding": state.get("grounding"),
        "clarifying_question": state.get("clarifying_question"),
        "message": state.get("message"),
        "error": state.get("error"),
        "match_tier": state.get("match_tier"),
        "matched_template": state.get("matched_template"),
        "stages": list(state.get("stages") or []),
        "warnings": list(state.get("warnings") or []),
        "session_id": state.get("session_id"),
        "retrieval_hits": list(state.get("retrieval_hits") or []),
        "tool_trace": list(state.get("tool_trace") or []),
        "repo_enabled": bool((state.get("repo_context") or {}).get("enabled")),
    }
