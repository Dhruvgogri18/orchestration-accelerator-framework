"""Registered archetype templates + one generic graph builder.

The old approach (plan_canonicalize.py) had one bespoke Python function
per workflow family. This replaces that with DATA: each template is just
a role graph (which roles exist, how many, how they connect). Adding a
new archetype means registering a new template, not writing a new
rebuild function -- `render()` below is the only builder and it's
archetype-agnostic.

A template is defined by giving `register()` a small *prototype* Plan
that's a clean textbook example of the shape. Its signature is computed
once at import time and used as the lookup key -- so the template
definition and its matching key can never drift out of sync.
"""
from __future__ import annotations

from dataclasses import dataclass

from orchestrator.api.schemas.common import Plan, PlanBranch, PlanStep
from .signature import _all_targets, _canonical_ids, plan_signature, tag_plan

ROLE_DEFAULTS: dict[str, tuple[str, str]] = {
    # role -> (default service, default description)
    "INGEST": ("Lambda", "Read the incoming input for this workflow."),
    "VALIDATE": ("Lambda", "Validate the input. On failure, alert operations."),
    "TRANSFORM": ("Glue", "Process the data."),
    "ENRICH": ("Lambda", "Enrich the record with additional data."),
    "QUALITY_CHECK": ("Lambda", "Run a data quality check on the output."),
    "DECISION": ("None", "Evaluate the outcome and route accordingly."),
    "FAN_OUT": ("None", "Run independent jobs in parallel; continue when all succeed."),
    "WORKER": ("Glue", "Run one independent processing job."),
    "AGGREGATE": ("Lambda", "Combine the results of the parallel branches."),
    "PERSIST": ("Lambda", "Load the results into the target database."),
    "MAP_ITERATE": ("None", "Iterate over each item in the collection."),
    "RETRY_WAIT": ("None", "Wait before retrying."),
    "COMPENSATE": ("Lambda", "Roll back the prior step's work."),
    "EXTERNAL_CALL": ("Lambda", "Invoke the external service."),
    "ACTION": ("Lambda", "Carry out the workflow's next action."),
    "NOTIFY_SUCCESS": ("SNS", "Publish a successful completion notification."),
    "NOTIFY_FAILURE": ("SNS", "Publish an operations failure alert."),
    "TERMINAL": ("None", "End the workflow."),
}


@dataclass(frozen=True)
class Template:
    name: str
    signature: tuple  # (role_multiset, edges) -- see signature.plan_signature


_REGISTRY: dict[tuple, Template] = {}


def register(name: str, prototype: Plan) -> Template:
    sig = plan_signature(prototype)
    tpl = Template(name=name, signature=sig)
    _REGISTRY[sig] = tpl
    return tpl


def all_templates() -> list[Template]:
    return list(_REGISTRY.values())


def lookup_exact(sig: tuple) -> Template | None:
    return _REGISTRY.get(sig)


def _overlap_score(a: tuple, b: tuple) -> float:
    a_roles, a_edges = a
    b_roles, b_edges = b
    ra, rb = set(a_roles), set(b_roles)
    ea, eb = set(a_edges), set(b_edges)
    role_j = len(ra & rb) / len(ra | rb) if (ra | rb) else 1.0
    edge_j = len(ea & eb) / len(ea | eb) if (ea | eb) else 1.0
    return 0.4 * role_j + 0.6 * edge_j


def best_match(sig: tuple) -> tuple[Template | None, float]:
    """Nearest registered template by structural overlap, and its score."""
    if not _REGISTRY:
        return None, 0.0
    scored = sorted(
        ((t, _overlap_score(sig, t.signature)) for t in _REGISTRY.values()),
        key=lambda ts: (-ts[1], ts[0].name),
    )
    return scored[0]


def render(
    original: Plan,
    template: Template,
    embed_fn=None,
    llm_classify_fn=None,
) -> Plan:
    """Deterministically render `original` into `template`'s fixed shape.

    Reuses the original step's id/service/description wherever a step of
    the matching role exists (so the LLM's naming survives); synthesizes
    a sensible default only for roles the template needs that the
    original plan didn't clearly produce.

    embed_fn/llm_classify_fn are accepted (and forwarded to tag_plan) so
    this matches how matcher.match() actually calls it: the hooks used to
    score the plan into `sig` must be the SAME hooks used to re-tag it
    here, or an exact-signature match could re-tag differently on render
    and silently drop or misassign a step. (Previously this signature
    only took 2 args while matcher.match() called it with 4 -- every
    "high" tier match raised TypeError before reaching the caller.)
    """
    tags = tag_plan(original, embed_fn, llm_classify_fn)
    canon = _canonical_ids(original, tags)  # raw_id -> "ROLE_n"
    by_canon = {v: k for k, v in canon.items()}  # "ROLE_n" -> raw_id
    orig_by_id = {s.id: s for s in original.steps}

    role_multiset, edges = template.signature
    used_ids: set[str] = set()

    def make_id(canon_id: str) -> str:
        raw = by_canon.get(canon_id)
        if raw and raw not in used_ids:
            used_ids.add(raw)
            return raw
        base = canon_id
        candidate, n = base, 2
        while candidate in used_ids:
            candidate = f"{base}_{n}"
            n += 1
        used_ids.add(candidate)
        return candidate

    final_id = {c: make_id(c) for c in role_multiset}

    # index outgoing edges per canonical source id, by edge type
    out_next: dict[str, str] = {}
    out_fail: dict[str, list[str]] = {}
    out_timeout: dict[str, list[str]] = {}
    out_branch: dict[str, list[tuple[str, str]]] = {}
    for src, etype, dst in edges:
        if etype == "next":
            out_next[src] = dst
        elif etype == "on_failure":
            out_fail.setdefault(src, []).append(dst)
        elif etype == "on_timeout":
            out_timeout.setdefault(src, []).append(dst)
        elif etype.startswith("branch:"):
            out_branch.setdefault(src, []).append((etype.split(":", 1)[1], dst))

    steps: list[PlanStep] = []
    for canon_id in role_multiset:
        role = canon_id.rsplit("_", 1)[0]
        raw = by_canon.get(canon_id)
        default_service, default_desc = ROLE_DEFAULTS.get(role, ("Lambda", f"{role} step."))
        orig = orig_by_id.get(raw) if raw else None

        steps.append(
            PlanStep(
                id=final_id[canon_id],
                service=(orig.service if orig and orig.service else default_service),
                description=(orig.description if orig and orig.description else default_desc),
                next=final_id.get(out_next.get(canon_id, ""), None) if canon_id in out_next else None,
                on_failure=[final_id[t] for t in out_fail.get(canon_id, []) if t in final_id],
                on_timeout=[final_id[t] for t in out_timeout.get(canon_id, []) if t in final_id],
                branches=[
                    PlanBranch(condition=cond, next=final_id[t])
                    for cond, t in out_branch.get(canon_id, [])
                    if t in final_id
                ],
            )
        )

    return Plan(name=original.name, summary=original.summary, steps=steps)


# ---------------------------------------------------------------------------
# Register the starter archetype library. Each prototype is a clean,
# textbook example -- add more by writing one prototype Plan, not a new
# rebuild function.
# ---------------------------------------------------------------------------

register(
    "linear_etl",
    Plan(
        name="linear-etl",
        summary="Ingest, transform, persist.",
        steps=[
            PlanStep(id="Ingest", service="Lambda", description="Read the input.", next="Transform", on_failure=["Alert"]),
            PlanStep(id="Transform", service="Glue", description="Process the data.", next="Persist", on_failure=["Alert"]),
            PlanStep(id="Persist", service="Lambda", description="Load results.", next="Success", on_failure=["Alert"]),
            PlanStep(id="Success", service="SNS", description="Publish success."),
            PlanStep(id="Alert", service="SNS", description="Publish failure alert."),
        ],
    ),
)

register(
    "fan_out_fan_in",
    Plan(
        name="fan-out-fan-in",
        summary="Validate, run two jobs in parallel, quality-check, persist.",
        steps=[
            PlanStep(id="Validate", service="Lambda", description="Validate input.", next="Parallel", on_failure=["Alert"]),
            PlanStep(
                id="Parallel", service="None", description="Run two jobs in parallel.",
                next="Quality", on_failure=["Alert"],
                branches=[
                    PlanBranch(condition="parallel arm: JobOne", next="JobOne"),
                    PlanBranch(condition="parallel arm: JobTwo", next="JobTwo"),
                ],
            ),
            PlanStep(id="JobOne", service="Glue", description="Run processing job one.", on_failure=["Alert"]),
            PlanStep(id="JobTwo", service="Glue", description="Run processing job two.", on_failure=["Alert"]),
            PlanStep(id="Quality", service="Lambda", description="Run data quality check.", next="Persist", on_failure=["Alert"]),
            PlanStep(id="Persist", service="Lambda", description="Load approved results.", next="Success", on_failure=["Alert"]),
            PlanStep(id="Success", service="SNS", description="Publish success."),
            PlanStep(id="Alert", service="SNS", description="Publish failure alert."),
        ],
    ),
)

register(
    "map_over_collection_with_retry_notify",
    Plan(
        name="map-over-collection-with-retry-notify",
        summary=(
            "Discover items in a collection and process each one independently "
            "and concurrently through validation, transformation, and load; "
            "invalid items notify without stopping the others; join into a "
            "final summary once all items finish."
        ),
        steps=[
            PlanStep(
                id="ReadItems", service="S3", description="Discover and read all items from the input location.",
                next="ProcessItemsConcurrently",
            ),
            PlanStep(
                id="ProcessItemsConcurrently", service="None",
                description="Iterate over each discovered item and run every one independently and concurrently.",
                next="Summarize",
                branches=[PlanBranch(condition="each item", next="ValidateItem")],
            ),
            PlanStep(
                id="ValidateItem", service="Lambda", description="Validate one item.",
                next="TransformItem", on_failure=["NotifyInvalidItem"],
            ),
            PlanStep(id="TransformItem", service="Lambda", description="Transform one validated item.", next="PersistItem"),
            PlanStep(id="PersistItem", service="Lambda", description="Load one item's transformed data into the target store."),
            PlanStep(id="NotifyInvalidItem", service="SNS", description="Publish a failure alert for an invalid item."),
            PlanStep(id="Summarize", service="Lambda", description="Combine the outcomes of all items into a final summary."),
        ],
    ),
)

register(
    "choice_branch_approval",
    Plan(
        name="choice-branch-approval",
        summary="Submit, decide, take one of two exclusive paths.",
        steps=[
            PlanStep(id="Submit", service="Lambda", description="Receive the request.", next="Decide"),
            PlanStep(
                id="Decide", service="None", description="Evaluate the request.",
                branches=[
                    PlanBranch(condition="approved", next="Proceed"),
                    PlanBranch(condition="rejected", next="Alert"),
                ],
            ),
            PlanStep(id="Proceed", service="Lambda", description="Carry out the approved action.", next="Success"),
            PlanStep(id="Success", service="SNS", description="Publish success."),
            PlanStep(id="Alert", service="SNS", description="Publish failure alert."),
        ],
    ),
)
