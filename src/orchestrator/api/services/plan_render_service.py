"""Renders a Stage 2 `Plan` as summary + Graphviz diagram for Stage 3 review.

Diagram topology rules (in order of preference):

1. **Structured control flow** — if any step has `next`, `on_failure`,
   `on_timeout`, or `branches`, draw ONLY from those fields. This is how
   the diagram is kept in sync with the written steps.

2. **Legacy heuristic fallback** — for older plans that lack those fields,
   infer a linear happy path and dashed failure side-branches from
   step ids / descriptions. Heuristic diagrams are best-effort only.

Once Stage 3.5 enrichment exists, prefer rendering from workflow_spec.
"""

from __future__ import annotations

import re

from orchestrator.api.schemas.common import Plan, PlanStep

# --- Legacy heuristic patterns (fallback only) -------------------------------

_BRANCH_SIGNAL_PATTERN = re.compile(
    r"\bbranch(es|ing)?\b|\bbased on\b|\bdepending on\b|\bwhether\b|\bchoice\b",
    re.IGNORECASE,
)

_FAILURE_HANDLER_ID_PATTERN = re.compile(
    r"("
    r"OnPaymentFailure|OnShippingFailure|OnFailure|AfterFailure|AfterShipping|"
    r"ShippingFailure|PaymentFailure|UnderwriterTimeout|Timeout|"
    r"Fail(ure)?|"
    r"Release(Reserved)?Inventory|ReleaseOrAdjust|ReleaseInventory|"
    r"Compensat|Rollback|"
    r"Refund(Customer)?(Payment)?|RefundPayment|"
    r"AdjustInventory|"
    r"DeadLetter|DLQ|"
    r"NotifyOperations|NotifyOps|AlertOperations|NotifyTeamLead|NotifyApplicant|"
    r"MarkOrderForManual|MarkNeeds|ManualFollowUp|ManualOrderReview|QueueManual|"
    r"Handle\w*Fail"
    r")",
    re.IGNORECASE,
)

_FAILURE_HANDLER_DESC_PATTERN = re.compile(
    r"^\s*on\s+\w.+\bfail(ure|s|ed)?\b|"
    r"^\s*triggered\s+only\s+by\b|"
    r"^\s*compensat|"
    r"^\s*rollback\b",
    re.IGNORECASE,
)

_HAPPY_PATH_ID_GUARD = re.compile(
    r"^(Check|Reserve|Process|Collect|Create|Validate|Start|Receive|Submit|"
    r"Transform|Load|Send|Publish|Invoke|Pull|Run|Wait|Disburse)",
    re.IGNORECASE,
)
_HAPPY_PATH_ID_OVERRIDE = re.compile(
    r"(Fail|Failure|Release|Refund|Compensat|Rollback|Manual|NotifyOps|"
    r"AlertOperations|AdjustInventory|QueueManual|Timeout|SecondaryReview)",
    re.IGNORECASE,
)

_FAILURE_SIGNAL_PATTERN = re.compile(
    r"\bfail(ure|s|ed)?\b|\bnotif|\brelease\b|\bcompensat|\brollback\b|"
    r"\brefund\b|\balert\b|\btimeout\b",
    re.IGNORECASE,
)
_SUCCESS_TERMINAL_PATTERN = re.compile(
    r"\bsucce(ed|ss|ssfully)\b|\bfinish\b|\bcomplete\b|\bdisburse\b",
    re.IGNORECASE,
)
_PLACEHOLDER_PATTERN = re.compile(r"\bTODO_[A-Z0-9_]+\b")


def _is_failure_handler(step: PlanStep) -> bool:
    step_id = step.id or ""
    if _HAPPY_PATH_ID_GUARD.search(step_id) and not _HAPPY_PATH_ID_OVERRIDE.search(
        step_id
    ):
        return False
    if _FAILURE_HANDLER_ID_PATTERN.search(step_id):
        return True
    return bool(_FAILURE_HANDLER_DESC_PATTERN.search(step.description or ""))


def _is_branch_step(step: PlanStep) -> bool:
    if _is_failure_handler(step):
        return False
    if step.branches:
        return True
    return bool(_BRANCH_SIGNAL_PATTERN.search(step.description or ""))


def _step_style(step: PlanStep, is_branch: bool) -> tuple[str, str, str]:
    if is_branch or step.branches:
        return "#FEF3C7", "#B45309", "diamond"
    if _is_failure_handler(step) or _FAILURE_SIGNAL_PATTERN.search(step.service or ""):
        return "#FDE2E1", "#C53030", "box"
    if _SUCCESS_TERMINAL_PATTERN.search(
        step.service or ""
    ) or _SUCCESS_TERMINAL_PATTERN.search(step.description or ""):
        return "#D4F4DD", "#2F855A", "box"
    return "#E0E7FF", "#4338CA", "box"


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


def _render_structured_diagram(plan: Plan) -> str:
    """Draw edges exclusively from next / on_failure / on_timeout / branches.

    Important: a step may have BOTH branches and next.
    - Choice: branches are exclusive arms; next is usually absent (default
      is modeled as a branch or left unset).
    - Parallel (synthetic): branches are concurrent arms; next is the
      join successor that runs after ALL arms succeed. That next edge
      must always be drawn or the diagram hides when the post-parallel
      step (e.g. BuildCustomer360) runs.
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
                # Parallel arms: solid orange; Choice arms: dashed amber
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

        # Always draw next when present — including on Parallel steps where
        # next = "after both arms complete". Previously this was skipped
        # whenever branches existed, which hid the join → BuildCustomer360 edge.
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
                # Choice with an explicit next (unusual) — still show it
                add_edge(step.id, step.next, "solid", "#666666", "next")
            else:
                add_edge(step.id, step.next, "solid", "#666666")

        for target in step.on_failure or []:
            add_edge(step.id, target, "dashed", "#C53030", "on failure")

        for target in step.on_timeout or []:
            add_edge(step.id, target, "dashed", "#B45309", "on timeout")

    lines.append("}")
    return "\n".join(lines)


def _find_happy_by_role(happy_path: list[PlanStep], *role_keywords: str) -> str | None:
    scored: list[tuple[int, int, str]] = []
    for step in happy_path:
        sid = step.id or ""
        lower = sid.lower()
        score = sum(1 for kw in role_keywords if kw in lower)
        if score:
            scored.append((score, len(sid), sid))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    return scored[0][2]


def _parent_for_handler(
    plan_index: int,
    handler: PlanStep,
    happy_path: list[PlanStep],
    plan_steps: list[PlanStep],
) -> str | None:
    if not happy_path:
        return None

    happy_by_id = {s.id: s for s in happy_path}
    text = f"{handler.id} {handler.description}"
    lower = text.lower()

    matches = [
        hid for hid in happy_by_id if re.search(re.escape(hid), text, re.IGNORECASE)
    ]
    if matches:
        matches.sort(key=len, reverse=True)
        return matches[0]

    payment_parent = _find_happy_by_role(
        happy_path, "payment", "pay", "charge", "collect"
    )
    shipping_parent = _find_happy_by_role(happy_path, "ship", "fulfill", "delivery")
    wait_parent = _find_happy_by_role(happy_path, "wait", "underwriter", "approval")

    if any(
        k in lower
        for k in ("timeout", "underwriter", "no response", "nonresponsive")
    ):
        if wait_parent:
            return wait_parent

    if any(
        k in lower
        for k in ("payment", "pay fail", "collectpayment", "processpayment", "onpayment")
    ):
        if payment_parent:
            return payment_parent

    if any(
        k in lower
        for k in (
            "ship",
            "fulfill",
            "createordershipment",
            "createshipment",
            "onshipping",
            "aftershipping",
        )
    ):
        if shipping_parent:
            return shipping_parent

    if re.search(r"release|refund|adjust|alert|notify|manual|queue", lower):
        if re.search(r"release", lower) and payment_parent and not re.search(
            r"ship", lower
        ):
            return payment_parent
        if shipping_parent:
            return shipping_parent
        if payment_parent:
            return payment_parent

    for j in range(plan_index - 1, -1, -1):
        candidate = plan_steps[j]
        if candidate.id in happy_by_id:
            return candidate.id

    return happy_path[-1].id


def _render_heuristic_diagram(plan: Plan) -> str:
    """Best-effort diagram for plans without structured flow fields.

    Linear happy path only — does NOT fan every later step out of a
    "based on" diamond (that caused diagram vs step-text mismatches).
    """
    lines = [
        "digraph plan {",
        "  rankdir=TB;",
        '  node [style="rounded,filled", fontname="Helvetica"];',
    ]

    happy_path: list[PlanStep] = []
    failure_handlers: list[tuple[int, PlanStep]] = []
    for i, step in enumerate(plan.steps):
        if _is_failure_handler(step):
            failure_handlers.append((i, step))
        else:
            happy_path.append(step)

    if not happy_path:
        happy_path = list(plan.steps)
        failure_handlers = []

    for step in plan.steps:
        label = _node_label(step)
        fillcolor, strokecolor, shape = _step_style(step, is_branch=False)
        lines.append(
            f'  "{step.id}" [label="{_dot_escape(label)}", '
            f'fillcolor="{fillcolor}", color="{strokecolor}", shape={shape}];'
        )

    for a, b in zip(happy_path, happy_path[1:]):
        lines.append(f'  "{a.id}" -> "{b.id}" [color="#666666"];')

    for plan_index, handler in failure_handlers:
        parent_id = _parent_for_handler(
            plan_index, handler, happy_path, list(plan.steps)
        )
        if parent_id is not None:
            lines.append(
                f'  "{parent_id}" -> "{handler.id}" '
                f'[style=dashed, color="#C53030", label="on failure"];'
            )

    lines.append("}")
    return "\n".join(lines)


def render_plan_diagram(plan: Plan) -> str:
    """Returns Graphviz DOT for the plan.

    Prefer structured next/on_failure/on_timeout/branches so the picture
    matches the written steps. Fall back to the legacy heuristic only
    when those fields are entirely absent.
    """
    if not plan.steps:
        return "digraph plan {}"

    if _plan_has_structured_flow(plan):
        return _render_structured_diagram(plan)
    return _render_heuristic_diagram(plan)
