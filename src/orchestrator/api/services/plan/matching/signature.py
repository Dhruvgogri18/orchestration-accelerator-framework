"""Canonical structural signature for a Plan.

Two plans get the SAME signature iff they have the same role graph --
same roles, same edges between them -- regardless of step id spelling,
description wording, step ordering in the source list, or which service
name was used. This is what makes template lookup an exact-key lookup
instead of fuzzy text matching: the fuzziness (paraphrase tolerance) was
already resolved by the role tagger; everything downstream of that is a
plain deterministic graph algorithm.
"""
from __future__ import annotations

import hashlib
import json

from orchestrator.api.schemas.common import Plan
from .roles import EmbedFn, LlmClassifyFn, condition_bucket as _condition_bucket, tag_step


def tag_plan(plan: Plan, embed_fn: EmbedFn = None, llm_classify_fn: LlmClassifyFn = None) -> dict[str, str]:
    """Return {raw_step_id: role_name} for every step, deterministically."""
    incoming: set[str] = set()
    for s in plan.steps:
        if s.next:
            incoming.add(s.next)
        incoming.update(s.on_failure or [])
        incoming.update(s.on_timeout or [])
        for b in s.branches or []:
            incoming.add(b.next)

    tags: dict[str, str] = {}
    for s in plan.steps:
        has_outgoing = bool(s.next or s.on_failure or s.on_timeout or s.branches)
        t = tag_step(
            step_id=s.id,
            service=s.service,
            description=s.description,
            has_branches=bool(s.branches),
            branch_conditions=[b.condition for b in (s.branches or [])],
            has_incoming=s.id in incoming,
            has_outgoing=has_outgoing,
            embed_fn=embed_fn,
            llm_classify_fn=llm_classify_fn,
        )
        tags[s.id] = t.role

    # Second pass: a FAN_OUT container's branch targets are its WORKER
    # arms. This needs the full graph (which the per-step tagger doesn't
    # see), so it's resolved here rather than in tag_step.
    for s in plan.steps:
        if tags.get(s.id) != "FAN_OUT":
            continue
        for b in s.branches or []:
            if tags.get(b.next) not in (None, "NOTIFY_SUCCESS", "NOTIFY_FAILURE", "TERMINAL"):
                tags[b.next] = "WORKER"
    return tags


def _canonical_ids(plan: Plan, tags: dict[str, str]) -> dict[str, str]:
    """Deterministic instance-numbered id per raw step: ROLE_1, ROLE_2...

    Traversal order: topological from root(s), ties broken by (role name,
    original list index) -- a pure function of plan content, so replaying
    the same structure always yields the same canonical ids.
    """
    by_id = {s.id: s for s in plan.steps}
    order_index = {s.id: i for i, s in enumerate(plan.steps)}
    incoming = {s.id: 0 for s in plan.steps}
    for s in plan.steps:
        for tgt in _all_targets(s):
            if tgt in incoming:
                incoming[tgt] += 1

    roots = sorted(
        (sid for sid, deg in incoming.items() if deg == 0),
        key=lambda sid: order_index[sid],
    )
    if not roots:  # cyclic / everything has an incoming edge -- fall back
        roots = sorted(by_id, key=lambda sid: order_index[sid])

    visited: list[str] = []
    seen: set[str] = set()
    frontier = list(roots)
    while frontier:
        frontier.sort(key=lambda sid: (tags.get(sid, ""), order_index[sid]))
        sid = frontier.pop(0)
        if sid in seen or sid not in by_id:
            continue
        seen.add(sid)
        visited.append(sid)
        for tgt in _all_targets(by_id[sid]):
            if tgt in by_id and tgt not in seen:
                frontier.append(tgt)
    for s in plan.steps:  # any node the traversal missed (disconnected)
        if s.id not in seen:
            visited.append(s.id)
            seen.add(s.id)

    counters: dict[str, int] = {}
    canonical: dict[str, str] = {}
    for sid in visited:
        role = tags.get(sid, "UNKNOWN")
        counters[role] = counters.get(role, 0) + 1
        canonical[sid] = f"{role}_{counters[role]}"
    return canonical


def _all_targets(s) -> list[str]:
    out = []
    if s.next:
        out.append(s.next)
    out.extend(s.on_failure or [])
    out.extend(s.on_timeout or [])
    out.extend(b.next for b in (s.branches or []))
    return out


def plan_signature(plan: Plan, embed_fn: EmbedFn = None, llm_classify_fn: LlmClassifyFn = None) -> tuple:
    """The canonical, hashable structural signature of a plan."""
    tags = tag_plan(plan, embed_fn, llm_classify_fn)
    canon = _canonical_ids(plan, tags)

    edges = []
    for s in plan.steps:
        src = canon[s.id]
        if s.next and s.next in canon:
            edges.append((src, "next", canon[s.next]))
        for tgt in s.on_failure or []:
            if tgt in canon:
                edges.append((src, "on_failure", canon[tgt]))
        for tgt in s.on_timeout or []:
            if tgt in canon:
                edges.append((src, "on_timeout", canon[tgt]))
        for b in s.branches or []:
            if b.next in canon:
                edges.append((src, f"branch:{_condition_bucket(b.condition)}", canon[b.next]))

    role_multiset = tuple(sorted(canon.values()))
    return (role_multiset, tuple(sorted(edges)))


def signature_hash(plan: Plan, embed_fn: EmbedFn = None, llm_classify_fn: LlmClassifyFn = None) -> str:
    sig = plan_signature(plan, embed_fn, llm_classify_fn)
    blob = json.dumps(sig, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
