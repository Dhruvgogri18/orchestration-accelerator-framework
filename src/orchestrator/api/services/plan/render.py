"""Renders a Stage 2 `Plan` as summary + Graphviz diagram for Stage 3 review.

Diagram topology: drawn ONLY from structured control-flow fields
(next / on_failure / on_timeout / branches). This is how the diagram
is kept in sync with the written steps.

#5 determinism pass: the old "legacy heuristic fallback" -- regex
guessing a linear happy path + dashed failure branches from step ids/
descriptions when those fields were empty -- has been REMOVED. Both
extraction paths (extract_turn's EXTRACTION_SYSTEM_PROMPT rule 8, and
the split pipeline's WIRING_SYSTEM_PROMPT) are instructed to always
populate control flow on a ready plan, so an empty-flow plan reaching
this module now means an extraction defect, not an expected legacy
shape -- it should surface as an error to retry, not get silently
guessed at. See PlanRenderError below and session_router.py's handling
of it (mapped to HTTP 502, same treatment as ExtractionError).

Once Stage 3.5 enrichment exists, prefer rendering from workflow_spec.
"""

from __future__ import annotations

import re
from typing import Any

from orchestrator.api.schemas.common import Plan, PlanStep

_PLACEHOLDER_PATTERN = re.compile(r"\bTODO_[A-Z0-9_]+\b")


class PlanRenderError(Exception):
    """Raised when a plan reaches rendering with no structured control
    flow at all -- extraction should have populated next/on_failure/
    on_timeout/branches on every ready plan; an empty set here means
    the extraction stage didn't do its job, not that a heuristic
    should guess one. Callers should treat this as a retryable
    extraction defect (see session_router.py mapping it to 502).
    """


def find_placeholders(plan: Plan) -> list[str]:
    seen: list[str] = []
    for step in plan.steps:
        for field_value in (step.service, step.description):
            for match in _PLACEHOLDER_PATTERN.findall(field_value or ""):
                if match not in seen:
                    seen.append(match)
    return seen


def apply_overrides(plan: Plan, overrides: dict[str, str]) -> tuple[Plan, list[str]]:
    new_steps = []
    for step in plan.steps:
        service = step.service
        description = step.description
        for token, value in overrides.items():
            service = service.replace(token, value)
            description = description.replace(token, value)
        new_steps.append(
            PlanStep(
                id=step.id,
                service=service,
                description=description,
                next=step.next,
                on_failure=list(step.on_failure or []),
                on_timeout=list(step.on_timeout or []),
                branches=list(step.branches or []),
            )
        )

    resolved_plan = Plan(name=plan.name, summary=plan.summary, steps=new_steps)
    remaining = find_placeholders(resolved_plan)
    return resolved_plan, remaining


def apply_overrides_to_json(node: Any, overrides: dict[str, str]) -> Any:
    """Same substitution as apply_overrides() above, generalized to any
    JSON-shaped value instead of just a Plan's steps.

    Needed once placeholders start showing up in workflow_spec (Stage
    3.5's enrichment output) rather than only the thin Plan (Stage 2):
    enrichment's ARN whitelist deliberately lets TODO_ tokens through,
    and it can place them in any field it writes -- a Parameters value,
    a resource string, anywhere -- not just the service/description
    pair apply_overrides() knows about. This walks the whole structure
    (dicts, lists, and strings, recursively) and substitutes every
    token in every string it finds, so it works the same regardless of
    where enrichment put the placeholder.

    Returns a new structure; the input is not mutated.
    """
    if isinstance(node, str):
        result = node
        for token, value in overrides.items():
            result = result.replace(token, value)
        return result
    if isinstance(node, dict):
        return {k: apply_overrides_to_json(v, overrides) for k, v in node.items()}
    if isinstance(node, list):
        return [apply_overrides_to_json(v, overrides) for v in node]
    return node


def _is_placeholder_service(service: str | None) -> bool:
    """True when the plan step does not call a real AWS service."""
    s = (service or "").strip().lower()
    return s in ("", "none", "n/a", "na", "null", "control", "terminal")


def _node_label(step: PlanStep) -> str:
    """Graphviz node label: omit (service) for pure terminal placeholders."""
    if _is_placeholder_service(step.service):
        return step.id
    return f"{step.id}\\n({step.service})"


def _summary_step_line(step: PlanStep) -> str:
    if _is_placeholder_service(step.service):
        return f"**{step.id}** — {step.description}"
    return f"**{step.id}** ({step.service}) — {step.description}"


def render_plan_summary(plan: Plan) -> str:
    lines = [f"### {plan.name}", "", plan.summary, "", "**Steps:**"]
    for i, step in enumerate(plan.steps, start=1):
        lines.append(f"{i}. {_summary_step_line(step)}")
        flow_bits: list[str] = []
        if step.next:
            flow_bits.append(f"next → `{step.next}`")
        if step.on_failure:
            flow_bits.append(
                "on_failure → " + ", ".join(f"`{x}`" for x in step.on_failure)
            )
        if step.on_timeout:
            flow_bits.append(
                "on_timeout → " + ", ".join(f"`{x}`" for x in step.on_timeout)
            )
        if step.branches:
            arms = ", ".join(
                f"{b.condition} → `{b.next}`" for b in step.branches
            )
            flow_bits.append(f"branches: {arms}")
        if flow_bits:
            lines.append("   - " + "; ".join(flow_bits))

    placeholders = find_placeholders(plan)
    if placeholders:
        lines += [
            "",
            "**Before you approve this plan, these values still need to be filled in:**",
        ]
        for token in placeholders:
            lines.append(f"- `{token}`")

    return "\n".join(lines)


def _dot_escape(text: str) -> str:
    return text.replace('"', '\\"')


def _plan_has_structured_flow(plan: Plan) -> bool:
    for step in plan.steps:
        if step.next or step.on_failure or step.on_timeout or step.branches:
            return True
    return False


def _is_parallel_style_branches(step: PlanStep) -> bool:
    """True when branches look like Parallel arms, not exclusive Choice arms.

    Extraction often encodes Parallel as a synthetic step with branch
    conditions like 'parallel arm: orders'. Choice arms use labels like
    'approved' / 'rejected'. Detecting this lets the diagram draw a clear
    join edge (next) after the arms complete.
    """
    if not step.branches:
        return False
    blob = " ".join(
        f"{b.condition or ''} {step.description or ''} {step.id or ''}"
        for b in step.branches
    ).lower()
    return any(
        k in blob
        for k in (
            "parallel",
            "concurrent",
            "simultaneously",
            "at the same time",
            "parallel arm",
            "arm:",
        )
    )


def _step_style(step: PlanStep, is_branch: bool) -> tuple[str, str, str]:
    if is_branch or step.branches:
        return "#FEF3C7", "#B45309", "diamond"
    blob = f"{step.service or ''} {step.description or ''}".lower()
    if any(k in blob for k in ("fail", "notif", "release", "compensat", "rollback", "refund", "alert", "timeout")):
        return "#FDE2E1", "#C53030", "box"
    if any(k in blob for k in ("succe", "finish", "complete", "disburse")):
        return "#D4F4DD", "#2F855A", "box"
    return "#E0E7FF", "#4338CA", "box"


def _render_structured_diagram(plan: Plan) -> str:
    """Draw edges exclusively from next / on_failure / on_timeout / branches.

    Important: a step may have BOTH branches and next.
    - Choice: branches are exclusive arms; next is usually absent (default
      is modeled as a branch or left unset).
    - Parallel (synthetic): branches are concurrent arms; next is the
      join successor that runs after ALL arms succeed. That next edge
      must always be drawn or the diagram hides when the post-parallel
      step runs.
    """
    lines = [
        "digraph plan {",
        "  rankdir=TB;",
        '  node [style="rounded,filled", fontname="Helvetica"];',
    ]

    known_ids = {s.id for s in plan.steps}

    for step in plan.steps:
        is_branch = bool(step.branches)
        label = _node_label(step)
        fillcolor, strokecolor, shape = _step_style(step, is_branch=is_branch)
        lines.append(
            f'  "{step.id}" [label="{_dot_escape(label)}", '
            f'fillcolor="{fillcolor}", color="{strokecolor}", shape={shape}];'
        )

    drawn: set[tuple[str, str, str]] = set()

    def add_edge(src: str, dst: str, style: str, color: str, label: str | None = None):
        if src not in known_ids or dst not in known_ids:
            return
        key = (src, dst, style)
        if key in drawn:
            return
        drawn.add(key)
        if label:
            lines.append(
                f'  "{src}" -> "{dst}" [style={style}, color="{color}", '
                f'label="{_dot_escape(label)}"];'
            )
        else:
            lines.append(
                f'  "{src}" -> "{dst}" [style={style}, color="{color}"];'
            )

    for step in plan.steps:
        parallel_style = _is_parallel_style_branches(step)

        if step.branches:
            for branch in step.branches:
                if parallel_style:
                    add_edge(
                        step.id,
                        branch.next,
                        "solid",
                        "#B45309",
                        branch.condition or "parallel arm",
                    )
                else:
                    add_edge(
                        step.id,
                        branch.next,
                        "dashed",
                        "#B45309",
                        branch.condition,
                    )

        if step.next:
            if parallel_style:
                add_edge(
                    step.id,
                    step.next,
                    "solid",
                    "#2F855A",
                    "after both complete",
                )
            elif step.branches:
                add_edge(step.id, step.next, "solid", "#666666", "next")
            else:
                add_edge(step.id, step.next, "solid", "#666666")

        for target in step.on_failure or []:
            add_edge(step.id, target, "dashed", "#C53030", "on failure")

        for target in step.on_timeout or []:
            add_edge(step.id, target, "dashed", "#B45309", "on timeout")

    lines.append("}")
    return "\n".join(lines)



def _is_map_style_step(step: PlanStep) -> bool:
    """True when the step is a Map / for-each over a collection.

    Structural signal takes priority: a container with exactly ONE
    outgoing branch is a Map (one reusable per-item body run once per
    element), not a Parallel fan-out (that's >=2 named branches -- see
    `_is_parallel_style_branches`). Branch count is unambiguous and
    catches paraphrases ("process every file concurrently") that don't
    happen to contain a literal "for each" / "each file" phrase. Falls
    back to keyword phrasing only for the rare case of a step with no
    branches at all describing iteration in a plain `next` chain --
    and only when the step itself is a control/placeholder step (no
    real AWS service). Without that guard, an ordinary Task step whose
    description merely mentions "each file" in passing (e.g.
    "Load each file's transformed data into OpenSearch") gets
    mislabeled as its own Map container even though it does no
    iteration at all -- it's just one step inside someone else's loop.
    """
    if step.branches:
        return len(step.branches) == 1
    if not _is_placeholder_service(step.service):
        return False
    blob = f"{step.id} {step.description or ''}".lower()
    return any(
        k in blob
        for k in (
            "for each",
            "for every",
            "map over",
            "iterate over",
            "each file",
            "each item",
            "each element",
            "over a list",
            "over the list",
        )
    )


def _mermaid_escape(label: str) -> str:
    return (
        (label or "")
        .replace('"', "'")
        .replace("\n", " ")
        .replace("[", "(")
        .replace("]", ")")
        .replace("{", "(")
        .replace("}", ")")
        .replace("|", "/")
    )


def _render_mermaid_diagram(plan: Plan) -> str:
    """Mermaid flowchart with Parallel subgraphs and Map styling.

    Edges come only from structured control-flow fields (same as DOT).
    Parallel steps: subgraph of arms; join via `next`.
    Map steps: hexagon node labeled Map.
    Choice steps: diamond + labeled arms.
    """
    lines: list[str] = ["flowchart TD"]
    known = {s.id for s in plan.steps}

    for step in plan.steps:
        label = step.id
        if not _is_placeholder_service(step.service):
            label = f"{step.id} ({step.service})"
        label = _mermaid_escape(label)
        if _is_map_style_step(step):
            lines.append(f'  {step.id}{{{{"{label} [Map]"}}}}')
        elif _is_parallel_style_branches(step):
            lines.append(f'  {step.id}["{label} [Parallel]"]')
        elif step.branches:
            lines.append(f'  {step.id}{{"{label}"}}')
        else:
            blob = f"{step.service or ''} {step.description or ''}".lower()
            if any(k in blob for k in ("sns", "notif", "alert")):
                lines.append(f'  {step.id}(["{label}"])')
            else:
                lines.append(f'  {step.id}["{label}"]')

    drawn: set[tuple[str, str, str]] = set()

    def edge(src: str, dst: str, kind: str, elabel: str | None = None) -> None:
        if src not in known or dst not in known:
            return
        key = (src, dst, kind)
        if key in drawn:
            return
        drawn.add(key)
        if kind == "fail":
            lines.append(
                f"  {src} -.->|{_mermaid_escape(elabel or 'on failure')}| {dst}"
            )
        elif kind == "timeout":
            lines.append(
                f"  {src} -.->|{_mermaid_escape(elabel or 'on timeout')}| {dst}"
            )
        elif elabel:
            lines.append(f"  {src} -->|{_mermaid_escape(elabel)}| {dst}")
        else:
            lines.append(f"  {src} --> {dst}")

    for step in plan.steps:
        if not _is_parallel_style_branches(step):
            continue
        arm_ids = [b.next for b in step.branches if b.next in known]
        if not arm_ids:
            continue
        lines.append(f'  subgraph {step.id}_arms["{step.id} branches"]')
        for aid in arm_ids:
            lines.append(f"    {aid}")
        lines.append("  end")

    for step in plan.steps:
        parallel = _is_parallel_style_branches(step)
        if step.branches:
            for branch in step.branches:
                edge(
                    step.id,
                    branch.next,
                    "ok",
                    branch.condition or ("parallel arm" if parallel else "branch"),
                )
        if step.next:
            edge(
                step.id,
                step.next,
                "ok",
                "after all complete" if parallel else None,
            )
        for t in step.on_failure or []:
            edge(step.id, t, "fail", "on failure")
        for t in step.on_timeout or []:
            edge(step.id, t, "timeout", "on timeout")

    lines.append("")
    return "\n".join(lines)


def render_plan_diagram(plan: Plan) -> str:
    """Returns a Mermaid flowchart for the plan (Parallel/Map aware).

    Drawn only from structured control-flow fields. Graphviz DOT remains
    available via render_plan_diagram_dot() for callers that need it.

    Raises PlanRenderError if the plan has no structured flow at all.
    """
    if not plan.steps:
        return "flowchart TD\n"

    if len(plan.steps) > 1 and not _plan_has_structured_flow(plan):
        raise PlanRenderError(
            f"Plan '{plan.name}' has {len(plan.steps)} steps but no "
            "next/on_failure/on_timeout/branches on any of them -- "
            "extraction should always wire control flow on a ready "
            "plan. This plan cannot be diagrammed; retry extraction "
            "rather than guessing a layout."
        )
    return _render_mermaid_diagram(plan)


def render_plan_diagram_dot(plan: Plan) -> str:
    """Graphviz DOT variant (legacy)."""
    if not plan.steps:
        return "digraph plan {}"
    if len(plan.steps) > 1 and not _plan_has_structured_flow(plan):
        raise PlanRenderError(
            f"Plan '{plan.name}' has {len(plan.steps)} steps but no structured flow."
        )
    return _render_structured_diagram(plan)