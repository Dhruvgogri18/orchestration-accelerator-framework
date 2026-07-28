"""Renders a Stage 2 `Plan` (the thin, pre-enrichment shape) as a
human-friendly summary and a colorful flow diagram -- for Stage 3
review, BEFORE any spec enrichment exists. Also handles finding
TODO_ placeholders in a plan, and substituting user-supplied values
for them at approval time.

Important honesty note: `Plan` has no structured branching data (no
`choices`, no `next_step`) -- that only exists after Stage 3.5
enrichment. So the diagram here uses a heuristic to guess branch
points from step descriptions. It's good enough for an early visual
review, but it is NOT authoritative. Once enrichment exists, re-render
the SAME way but from the full `workflow_spec.json` (which has real
structure) for Stage 6's final review, where accuracy actually matters.
"""
from __future__ import annotations

import re

from orchestrator.extraction.models import Plan, PlanStep

# Heuristic signal words -- if a step's description contains any of
# these, treat it as a branch point: everything after it fans out as
# separate branches rather than continuing as one straight line.
_BRANCH_SIGNAL_PATTERN = re.compile(
    r"\bbranch(es|ing)?\b|\bbased on\b|\bdepending on\b|\bif\b|\bwhether\b",
    re.IGNORECASE,
)

# Heuristic signal words for coloring -- these do NOT affect graph
# structure (only _BRANCH_SIGNAL_PATTERN does), just which color/shape
# a step is drawn with.
_FAILURE_SIGNAL_PATTERN = re.compile(
    r"\bfail(ure|s|ed)?\b|\bnotif", re.IGNORECASE
)
_SUCCESS_TERMINAL_PATTERN = re.compile(
    r"\bsucce(ed|ss|ssfully)\b|\bfinish\b|\bcomplete\b", re.IGNORECASE
)

# Any TODO_SOMETHING_LIKE_THIS token -- these are the placeholders
# extraction deliberately leaves behind instead of inventing values
# (see prompts.py rule 1).
_PLACEHOLDER_PATTERN = re.compile(r"\bTODO_[A-Z0-9_]+\b")


def _is_branch_step(step: PlanStep) -> bool:
    return bool(_BRANCH_SIGNAL_PATTERN.search(step.description))


def _step_style(step: PlanStep, is_branch: bool) -> tuple[str, str, str]:
    """Returns (fillcolor, strokecolor, shape) for a step, based on
    simple keyword heuristics over its service/description. Purely
    cosmetic -- never affects the graph's actual edges.
    """
    if is_branch:
        return "#FEF3C7", "#B45309", "diamond"  # amber
    if _FAILURE_SIGNAL_PATTERN.search(step.service) or _FAILURE_SIGNAL_PATTERN.search(step.description):
        return "#FDE2E1", "#C53030", "box"  # red
    if _SUCCESS_TERMINAL_PATTERN.search(step.service) or _SUCCESS_TERMINAL_PATTERN.search(step.description):
        return "#D4F4DD", "#2F855A", "box"  # green
    return "#E0E7FF", "#4338CA", "box"  # default blue/purple (task)


def find_placeholders(plan: Plan) -> list[str]:
    """Every distinct TODO_ token across all steps' service/description
    fields, in first-seen order. Used both to show the user what still
    needs filling in before approval, and by apply_overrides below.
    """
    seen: list[str] = []
    for step in plan.steps:
        for field_value in (step.service, step.description):
            for match in _PLACEHOLDER_PATTERN.findall(field_value):
                if match not in seen:
                    seen.append(match)
    return seen


def apply_overrides(plan: Plan, overrides: dict[str, str]) -> tuple[Plan, list[str]]:
    """Substitutes user-supplied values for TODO_ placeholders found in
    the plan. Returns (resolved_plan, remaining_unresolved_placeholders)
    -- overrides the user didn't provide are left as-is, on purpose
    (see docs/phase1_architecture.md Stage 3 discussion: unresolved
    placeholders are allowed to persist and get caught later at
    Stage 7 pre-flight checks, rather than blocking approval here).
    """
    new_steps = []
    for step in plan.steps:
        service = step.service
        description = step.description
        for token, value in overrides.items():
            service = service.replace(token, value)
            description = description.replace(token, value)
        new_steps.append(PlanStep(id=step.id, service=service, description=description))

    resolved_plan = Plan(name=plan.name, summary=plan.summary, steps=new_steps)
    remaining = find_placeholders(resolved_plan)
    return resolved_plan, remaining


def render_plan_summary(plan: Plan) -> str:
    """Markdown summary: the plan's own summary line, a readable step
    list, and (if any) an explicit callout of unresolved placeholders
    -- so a reviewer knows exactly what they still need to provide
    before approving. No JSON -- meant to be shown directly to a
    non-technical reviewer at Stage 3.
    """
    lines = [f"### {plan.name}", "", plan.summary, "", "**Steps:**"]
    for i, step in enumerate(plan.steps, start=1):
        lines.append(f"{i}. **{step.id}** ({step.service}) — {step.description}")

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


def render_plan_diagram(plan: Plan) -> str:
    """Returns Graphviz DOT source for the plan's flow, with steps
    colored by role (task / branch / failure-notify / terminal
    success) -- see _step_style. Branch detection/fan-out logic is
    unchanged from the original heuristic.
    """
    if not plan.steps:
        return "digraph plan {}"

    lines = [
        "digraph plan {",
        "  rankdir=TB;",
        '  node [style="rounded,filled", fontname="Helvetica"];',
    ]

    branch_index = next(
        (i for i, s in enumerate(plan.steps) if _is_branch_step(s)), None
    )

    for i, step in enumerate(plan.steps):
        label = f"{step.id}\\n({step.service})"
        fillcolor, strokecolor, shape = _step_style(step, is_branch=(i == branch_index))
        lines.append(
            f'  "{step.id}" [label="{_dot_escape(label)}", '
            f'fillcolor="{fillcolor}", color="{strokecolor}", shape={shape}];'
        )

    if branch_index is None:
        for a, b in zip(plan.steps, plan.steps[1:]):
            lines.append(f'  "{a.id}" -> "{b.id}" [color="#666666"];')
    else:
        for a, b in zip(plan.steps[: branch_index + 1], plan.steps[1 : branch_index + 1]):
            lines.append(f'  "{a.id}" -> "{b.id}" [color="#666666"];')
        branch_step = plan.steps[branch_index]
        for target in plan.steps[branch_index + 1 :]:
            lines.append(f'  "{branch_step.id}" -> "{target.id}" [style=dashed, color="#B45309"];')

    lines.append("}")
    return "\n".join(lines)