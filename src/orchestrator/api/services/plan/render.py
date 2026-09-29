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

# A field value of exactly "$.TODO_X" is a JSONPath *reference* meaning
# "look this up from execution input named TODO_X at runtime" -- it is
# not literal text that merely contains the token. See the note in
# apply_overrides_to_json() below.
_JSONPATH_TODO_REF = re.compile(r"^\$\.(TODO_[A-Z0-9_]+)$")

# A field value of exactly "{% TODO_X %}" is a JSONata-wrapped
# placeholder. No known generator in this codebase currently emits this
# exact shape (unresolved tokens are always prefixed with
# "$states.input." -- see _todo_to_input_expr), but resolving it
# defensively is cheap and correct: once TODO_X has a concrete value,
# the field is a plain literal and the "{% %}" JSONata markers must be
# dropped, not left wrapping raw unquoted text (which is not valid
# JSONata and would fail to parse/deploy).
_JSONATA_TODO_REF = re.compile(r"^\{%\s*(TODO_[A-Z0-9_]+)\s*%\}$")

# Whole-value "{% $states.input.TODO_X %}" (post-translation JSONata form
# of an unresolved placeholder). Same whole-expression replacement rule
# as the two patterns above: once filled, the field becomes a plain
# literal and the wrapper must not remain around the substituted text.
_JSONATA_STATES_INPUT_TODO_REF = re.compile(
    r"^\{%\s*\$states\.input\.(TODO_[A-Z0-9_]+)\s*%\}$"
)

# Bare unresolved TODO token used as a whole value.
_BARE_TODO_TOKEN = re.compile(r"^TODO_[A-Z0-9_]+$")


def _is_path_or_expression_value(value: str) -> bool:
    """True when a parameter value is still a runtime path / expression.

    After placeholder fill, a former ``$.TODO_X`` becomes a plain constant
    (bucket name, S3 URI, table name, …). Callers use this to decide
    whether a leftover JSONPath ``.$`` key suffix should be stripped so
    ASL generation does not re-wrap the constant as invalid JSONata.
    """
    s = (value or "").strip()
    if not s:
        return False
    if s.startswith("$") or s.startswith("{%"):
        return True
    if _BARE_TODO_TOKEN.match(s):
        return True
    return False


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
                component_type=step.component_type,
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
    a resource string, **or a dict key** (e.g. DynamoDB Item attribute
    name ``TODO_ATTRIBUTE_NAME``) -- not just the service/description
    pair apply_overrides() knows about. This walks the whole structure
    (dicts, lists, and strings, recursively) and substitutes every
    token in every string value **and every dict key**, so it works the
    same regardless of where enrichment put the placeholder.

    Returns a new structure; the input is not mutated.
    """
    if isinstance(node, str):
        # A whole-value JSONPath reference like "$.TODO_SOURCE_S3_BUCKET"
        # must resolve to the literal value, not have the token replaced
        # in place. A plain substring replace would turn it into
        # "$.test-bucket" -- still a "$." reference, just now pointing at
        # a field named "test-bucket" that will never exist in the
        # execution input. asl_service.py's JSONPath->JSONata conversion
        # then faithfully renders that bogus reference as
        # "{% $states.input.test-bucket %}": syntactically broken
        # (JSONata parses the hyphen as subtraction) and semantically
        # wrong (it was supposed to be the literal bucket name). Once an
        # override supplies a concrete value for the token, the field is
        # no longer a runtime lookup at all -- resolve the reference to
        # the literal value directly, dropping the "$." wrapper.
        stripped = node.strip()
        ref_match = _JSONPATH_TODO_REF.match(stripped)
        if (
            ref_match
            and ref_match.group(1) in overrides
            and overrides[ref_match.group(1)] is not None
        ):
            return overrides[ref_match.group(1)]

        # Same idea for a bare "{% TODO_X %}" JSONata-wrapped placeholder:
        # substring-replacing the token inside it would leave the "{% %}"
        # markers around raw, unquoted literal text -- invalid JSONata
        # for anything that isn't itself a legal bare expression (a URI
        # like "s3://bucket/key" is not). Drop the wrapper and use the
        # literal value directly.
        jsonata_match = _JSONATA_TODO_REF.match(stripped)
        if (
            jsonata_match
            and jsonata_match.group(1) in overrides
            and overrides[jsonata_match.group(1)] is not None
        ):
            return overrides[jsonata_match.group(1)]

        # Post-translation form: "{% $states.input.TODO_X %}". Whole-
        # expression replacement yields the plain literal so generation
        # does not re-wrap it (and so substring replace cannot produce
        # "{% $states.input.my-input-bucket %}" — still-broken JSONata).
        states_match = _JSONATA_STATES_INPUT_TODO_REF.match(stripped)
        if (
            states_match
            and states_match.group(1) in overrides
            and overrides[states_match.group(1)] is not None
        ):
            return overrides[states_match.group(1)]

        result = node
        for token, value in overrides.items():
            if token and value is not None:
                result = result.replace(token, value)
        return result
    if isinstance(node, dict):
        out: dict = {}
        for k, v in node.items():
            new_k = k
            if isinstance(k, str):
                for token, value in overrides.items():
                    if token and value is not None and token in k:
                        new_k = k.replace(token, value)
            new_v = apply_overrides_to_json(v, overrides)
            # After a path-style TODO under a JSONPath '.$' key is filled
            # with a constant, strip the '.$' suffix. Leaving it would
            # cause ASL generation to treat the constant as a path and
            # emit invalid JSONata like "{% s3://bucket/key %}" or
            # "{% $states.input.my-table %}" for a filled table name.
            if (
                isinstance(new_k, str)
                and new_k.endswith(".$")
                and isinstance(new_v, str)
                and not _is_path_or_expression_value(new_v)
            ):
                new_k = new_k[:-2]
            out[new_k] = new_v
        return out
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


def _md_escape_currency(text: str | None) -> str:
    """Prevent Streamlit/Markdown math mode from eating `$1000`-style amounts.

    Unescaped `$…$` pairs are treated as inline math, which collapses branch
    labels like ``order amount is over $1000 → Charge`` into mangled text
    (e.g. ``orderamountisnotover1000→…``). Escape every ``$`` so currency
    and thresholds render literally in the plan summary.
    """
    if not text:
        return ""
    return str(text).replace("$", "\\$")


def _summary_step_line(step: PlanStep) -> str:
    desc = _md_escape_currency(step.description)
    if _is_placeholder_service(step.service):
        return f"**{step.id}** — {desc}"
    return f"**{step.id}** ({step.service}) — {desc}"


def render_plan_summary(plan: Plan) -> str:
    lines = [
        f"### {_md_escape_currency(plan.name)}",
        "",
        _md_escape_currency(plan.summary),
        "",
        "**Steps:**",
    ]
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
                f"{_md_escape_currency(b.condition)} → `{b.next}`"
                for b in step.branches
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


# ---------------------------------------------------------------------------
# Control-flow kind classification (Choice / Parallel / Map)
#
# Single source of truth used by render (diagram + summary) and by
# canonicalize (branch labeling). Prefer explicit `component_type` from
# Pipeline Analysis; only fall back to text heuristics for older stored
# plans / templates that predate the field.
#
# Design rule for the fallback (when component_type is absent):
#   - Parallel requires POSITIVE concurrency evidence.
#   - Map requires POSITIVE iteration evidence (and a single branch body).
#   - Otherwise any step with branches is Choice (exclusive alternatives).
# Exclusive branches are the default shape of "decide / if / whether /
# threshold / outcome" language. Concurrent branches are the special case
# and must not be inferred from the mere presence of a branches list.
# ---------------------------------------------------------------------------

_PARALLEL_EVIDENCE = (
    "parallel",
    "concurrent",
    "simultaneously",
    "at the same time",
    "in parallel",
    "all at once",
    "in tandem",
    "fan-out",
    "fan out",
)

_MAP_ITERATE_KEYWORDS = (
    "for each",
    "for every",
    "map over",
    "iterate over",
    "each file",
    "each item",
    "each element",
    "each record",
    "each order",
    "each row",
    "per item",
    "per file",
    "per record",
    "per order",
    "over a list",
    "over the list",
    "over the collection",
    "every item",
    "every file",
    "every order",
)

# Signals on the step itself (id / description) that this is a decision
# point rather than a concurrent fan-out or a Map iterator.
_CHOICE_SIGNAL_KEYWORDS = (
    "evaluat",
    "rout",
    "decision",
    "choice",
    "decide",
    "determine",
    "whether",
    "gate",
    "branch on",
    "based on",
    "depending on",
    "if ",
    "otherwise",
    "bypass",
    "skip",
    "threshold",
    "amount",
    "greater",
    "less than",
    "over $",
    "under $",
    "above",
    "below",
    "at least",
    "at most",
    "exceed",
)

# Signals on a branch condition that the arm is an exclusive outcome,
# not a concurrent worker job.
_CHOICE_CONDITION_KEYWORDS = (
    "succeeded",
    "success",
    "approved",
    "rejected",
    "pass",
    "fail",
    "failed",
    "failure",
    "eligible",
    "ineligible",
    "valid",
    "invalid",
    "true",
    "false",
    "match",
    "no match",
    "over",
    "under",
    "or less",
    "or more",
    "or fewer",
    "greater than",
    "less than",
    "at least",
    "at most",
    "above",
    "below",
    "exceed",
    "within",
    "outside",
    "skip",
    "bypass",
    "otherwise",
    "else",
    "default",
    "timeout",
    "timed out",
    "still failing",
    "still failed",
    "not found",
    "found",
    "empty",
    "non-empty",
    "present",
    "absent",
    "yes",
    "no",
)


def _step_text_blob(step: PlanStep) -> str:
    """Id + description + every branch condition, lowercased."""
    parts = [step.id or "", step.description or ""]
    for b in step.branches or []:
        parts.append(b.condition or "")
    return " ".join(parts).lower()


def _has_parallel_evidence(step: PlanStep) -> bool:
    """Positive concurrency signal in id/description/conditions."""
    blob = _step_text_blob(step)
    if any(k in blob for k in _PARALLEL_EVIDENCE):
        return True
    # Explicit "parallel arm" labels (often written by older canonicalize)
    # count as evidence, but bare "arm:" alone does not — that substring
    # appears in non-concurrent wording and caused false Parallel labels.
    for b in step.branches or []:
        cond = (b.condition or "").lower()
        if "parallel arm" in cond or cond.startswith("parallel "):
            return True
    return False


def _has_choice_evidence(step: PlanStep) -> bool:
    """Positive exclusive-decision signal in id/description/conditions."""
    blob = _step_text_blob(step)
    if any(k in blob for k in _CHOICE_SIGNAL_KEYWORDS):
        return True
    for b in step.branches or []:
        cond = (b.condition or "").lower()
        if any(k in cond for k in _CHOICE_CONDITION_KEYWORDS):
            return True
    # Id patterns that almost always mean a Choice node.
    sid = (step.id or "").lower()
    if sid.endswith("gate") or sid.endswith("choice") or "decision" in sid:
        return True
    return False


def _has_map_evidence(step: PlanStep) -> bool:
    """Positive per-item iteration signal in id/description/conditions."""
    blob = _step_text_blob(step)
    return any(k in blob for k in _MAP_ITERATE_KEYWORDS)


def _component_kind(step: PlanStep) -> str | None:
    """Classify a step as map / parallel / choice / task, or None.

    Order of authority:
      1. Explicit `component_type` from Pipeline Analysis (DecisionGate,
         Orchestrator, or any Task-like type). Orchestrator is split into
         map vs parallel by branch cardinality only after the type is known.
      2. Text heuristics for plans that predate component_type. Parallel
         and Map require positive evidence; otherwise branches imply Choice.
    """
    ct = step.component_type
    if ct is not None:
        if ct == "DecisionGate":
            return "choice"
        if ct == "Orchestrator":
            n = len(step.branches or [])
            if n == 1:
                return "map"
            if n >= 2:
                return "parallel"
            return None  # Orchestrator with no branches: extraction defect
        # Any other component_type is not a Map/Parallel/Choice container.
        return "task"

    # --- legacy fallback (no component_type) ---
    if not step.branches:
        # Rare: iteration described only in prose with a plain next chain.
        if _is_placeholder_service(step.service) and _has_map_evidence(step):
            return "map"
        return None

    n = len(step.branches)
    if n == 1 and _has_map_evidence(step) and not _has_choice_evidence(step):
        return "map"
    if n >= 2 and _has_parallel_evidence(step) and not _has_choice_evidence(step):
        return "parallel"
    if n >= 1 and _has_choice_evidence(step):
        return "choice"
    # Default for any branched step without concurrency/iteration evidence:
    # exclusive alternatives (Choice). Do NOT default to Parallel — that is
    # what mislabeled amount/threshold/outcome gates as concurrent arms.
    if n >= 1:
        return "choice"
    return None


def _is_parallel_style_branches(step: PlanStep) -> bool:
    """True when branches are concurrent Parallel arms (not exclusive Choice).

    Uses `_component_kind` so render and canonicalize share one classifier.
    """
    return _component_kind(step) == "parallel"


def _step_style(step: PlanStep, is_branch: bool) -> tuple[str, str, str]:
    """fillcolor, strokecolor, shape — diamond only for exclusive Choice."""
    kind = _component_kind(step)
    if kind == "choice" or (kind is None and step.branches and not _is_parallel_style_branches(step) and not _is_map_style_step(step)):
        return "#FEF3C7", "#B45309", "diamond"
    if kind == "parallel" or kind == "map":
        return "#EDE9FE", "#7C3AED", "box"
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

    Delegates entirely to `_component_kind` so Map vs Choice vs Parallel
    share one classifier (component_type first, then positive-evidence
    text heuristics). See `_component_kind` for the full rules.
    """
    return _component_kind(step) == "map"


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
        # Currency amounts must not look like empty / broken tokens in labels.
        .replace("$", "USD ")
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