"""Deterministic post-processing for Stage-2 Plan objects.

Runs AFTER extraction (never an LLM call).

Two layers:
  A) Light cleanup — drop pure terminals, merge notify ids, strip bogus
     branches from non-decision steps, Parallel service → None.
  B) Pattern rebuild — when the plan clearly matches the common
     "validate → parallel workers → quality → load → success / fail→ops"
     family, replace free-form topology with one fixed graph so
     paraphrases with the same intent draw the same diagram.

This is intentional structure locking for a detected pattern family,
not prompt-specific hardcoding of resource names.
"""
from __future__ import annotations

import re

from orchestrator.api.schemas.common import Plan, PlanBranch, PlanStep

_FAIL_NOTIFY_RE = re.compile(
    r"(alert|notify).*(ops|operations|failure|fail)|"
    r"(ops|operations).*(alert|notify)|"
    r"notifyops|alertoperations|sendoperationsalert|"
    r"notifydataops|notifyonfailure",
    re.I,
)
_SUCCESS_NOTIFY_RE = re.compile(
    r"(success|completion).*(notif|sns|publish)|"
    r"(notif|notify).*(success|completion)|"
    r"sendsuccessnotification|notifysuccess",
    re.I,
)
_PARALLEL_RE = re.compile(
    r"\b(parallel|concurrently|concurrent|simultaneously|"
    r"at the same time|in parallel)\b",
    re.I,
)
# Stronger signal for Parallel *control* nodes. Do NOT use "both … jobs"
# alone — quality text often says "outputs from both processing jobs".
_PARALLEL_CONTAINER_RE = re.compile(
    r"\b(parallel|concurrently|in parallel|at the same time|"
    r"run both|two independent)\b",
    re.I,
)
_MAP_ITERATE_RE = re.compile(
    r"\b(for each|for every|each item|each file|each record|each element|"
    r"map over|iterate over|per item|per file|per record|independently)\b",
    re.I,
)
_PURE_TERMINAL_RE = re.compile(
    r"^(WorkflowSucceeded|WorkflowFailed|CompleteWorkflow|"
    r"TerminateWorkflow|Succeed|Fail|EndWorkflow|Finish)$",
    re.I,
)
_VALIDATE_RE = re.compile(r"\bvalidat", re.I)
_QUALITY_RE = re.compile(r"\b(quality|data.?quality|\bdq\b)", re.I)
_LOAD_RE = re.compile(r"\b(load|write).*(database|dynamo|rds|table|target)|\bload\b", re.I)
_WORK_ACTION_RE = re.compile(
    r"\b(validat|process|processo|glue|load|quality|enrich|transform|"
    r"invok|reserv|charge|ship|refund|check|read|write|upload|download|"
    r"start.?job|run.?job|query|scan|put.?item|get.?item)\b",
    re.I,
)
_CHOICE_RE = re.compile(r"\b(evaluat\w*|rout\w*|decision\w*|choice\w*)\b", re.I)


def _blob(step: PlanStep) -> str:
    return f"{step.id} {step.service or ''} {step.description or ''}"


def _is_placeholder_service(service: str | None) -> bool:
    s = (service or "").strip().lower()
    return s in ("", "none", "n/a", "na", "null", "control", "terminal", "pass")


def _is_pure_terminal(step: PlanStep) -> bool:
    if step.next or step.on_failure or step.on_timeout or step.branches:
        return False
    if not _is_placeholder_service(step.service):
        return False
    if _PURE_TERMINAL_RE.match(step.id or ""):
        return True
    desc = (step.description or "").lower()
    return any(
        k in desc
        for k in (
            "end the workflow",
            "finish successfully",
            "do nothing further",
            "terminate the workflow",
            "fail the workflow",
            "workflow succeeded",
            "workflow failed",
        )
    )


def _is_failure_notify(step: PlanStep) -> bool:
    if _is_pure_terminal(step):
        return False
    sid = (step.id or "").strip()
    desc = (step.description or "").strip()
    svc = (step.service or "").strip().lower()
    if re.match(
        r"^(AlertOperations|NotifyOps|NotifyDataOps|NotifyOnFailure|"
        r"SendOperationsAlert|SendFailureNotification|AlertOps)$",
        sid,
        re.I,
    ):
        return True
    if svc not in ("sns", "amazon sns", "email", "ses"):
        return False
    if _WORK_ACTION_RE.search(sid) or _WORK_ACTION_RE.search(desc):
        return False
    return bool(_FAIL_NOTIFY_RE.search(f"{sid} {desc}"))


def _is_success_notify(step: PlanStep) -> bool:
    if _is_failure_notify(step) or _is_pure_terminal(step):
        return False
    sid = (step.id or "").strip()
    desc = (step.description or "").strip()
    svc = (step.service or "").strip().lower()
    if re.match(
        r"^(SendSuccessNotification|NotifySuccess|SuccessNotification|"
        r"NotifyOnSuccess|PublishSuccess)$",
        sid,
        re.I,
    ):
        return True
    if svc not in ("sns", "amazon sns", "email", "ses"):
        return False
    if _WORK_ACTION_RE.search(sid) or _WORK_ACTION_RE.search(desc):
        return False
    return bool(_SUCCESS_NOTIFY_RE.search(f"{sid} {desc}"))


def _is_parallel_container(step: PlanStep) -> bool:
    """True for Parallel control nodes — not quality/load steps that mention jobs."""
    blob = _blob(step)
    if _is_failure_notify(step) or _is_success_notify(step):
        return False
    if _QUALITY_RE.search(blob) and not re.search(r"\bparallel\b", blob, re.I):
        return False
    if _CHOICE_RE.search(step.id or "") or (
        step.branches
        and any(
            re.search(r"pass|fail|good|not good|eligible|approved", b.condition or "", re.I)
            for b in step.branches
        )
    ):
        # Choice-style branches (pass/fail), not parallel arms
        if not re.search(r"\bparallel\b", blob, re.I):
            return False
    if not _PARALLEL_CONTAINER_RE.search(blob):
        return False
    # Prefer service None / control, or explicit parallel in id
    if re.search(r"parallel|concurrent", step.id or "", re.I):
        return True
    if _is_placeholder_service(step.service):
        return True
    # Description-only parallel with real service is still a container if
    # it has branches to other work steps
    if step.branches and len(step.branches) >= 2:
        return True
    return bool(re.search(r"\b(parallel|in parallel|concurrently)\b", blob, re.I))


def _is_map_container(step: PlanStep) -> bool:
    """True for a Map/iteration control node: ONE reusable per-item body
    executed once per element of a collection -- not N distinct named
    parallel jobs (that's _is_parallel_container, which requires >=2
    branches).

    NOTE: `_light_cleanup` no longer gates branch-preservation through
    this (or `_is_parallel_container`) -- any step with a non-empty
    `branches` list is now trusted as real structure unconditionally,
    because keyword-gating on that decision is exactly what silently
    deleted correctly-wired branches whenever the description used a
    synonym ("simultaneously", "at the same time") the regex hadn't
    been taught yet. This helper is kept for callers elsewhere in this
    module that still want a best-effort keyword classification (e.g.
    the "does this look like a Map, specifically" question), not for
    deciding whether to keep or discard graph structure.
    """
    if _is_failure_notify(step) or _is_success_notify(step):
        return False
    if not step.branches or len(step.branches) != 1:
        return False
    if _CHOICE_RE.search(step.id or ""):
        return False
    blob = _blob(step)
    return bool(_PARALLEL_RE.search(blob) or _MAP_ITERATE_RE.search(blob))


def _is_validate(step: PlanStep) -> bool:
    return bool(_VALIDATE_RE.search(_blob(step))) and not _is_failure_notify(step)


def _is_quality(step: PlanStep) -> bool:
    if _is_failure_notify(step) or _is_success_notify(step):
        return False
    blob = _blob(step)
    if not _QUALITY_RE.search(blob):
        return False
    # Pure routing Choice with pass/fail arms is not the quality *task*
    if step.branches and any(
        re.search(r"pass|fail|good|not good|approved|rejected", b.condition or "", re.I)
        for b in step.branches
    ):
        return False
    return True


def _is_quality_choice(step: PlanStep) -> bool:
    return bool(step.branches) and bool(_QUALITY_RE.search(_blob(step)) or _CHOICE_RE.search(_blob(step)))


def _is_load(step: PlanStep) -> bool:
    if _is_failure_notify(step) or _is_success_notify(step) or _is_quality(step):
        return False
    return bool(_LOAD_RE.search(_blob(step)))


def _norm_svc(service: str | None, default: str) -> str:
    if _is_placeholder_service(service):
        return default
    return (service or default).strip() or default


def canonicalize_plan(plan: Plan) -> Plan:
    """Light cleanup, then optional fixed-topology rebuild for known patterns."""
    if not plan.steps:
        return plan

    cleaned = _light_cleanup(plan)
    rebuilt = _rebuild_validate_parallel_quality_load(cleaned)
    return rebuilt if rebuilt is not None else cleaned


# ---------------------------------------------------------------------------
# Layer A — light cleanup
# ---------------------------------------------------------------------------


def _light_cleanup(plan: Plan) -> Plan:
    steps = [
        PlanStep(
            id=s.id,
            service=s.service,
            description=s.description,
            next=s.next,
            on_failure=list(s.on_failure or []),
            on_timeout=list(s.on_timeout or []),
            branches=[PlanBranch(condition=b.condition, next=b.next) for b in (s.branches or [])],
        )
        for s in plan.steps
    ]

    drop_ids = {s.id for s in steps if _is_pure_terminal(s)}
    fail_ids = [s.id for s in steps if s.id not in drop_ids and _is_failure_notify(s)]
    success_ids = [s.id for s in steps if s.id not in drop_ids and _is_success_notify(s)]
    fail_canonical = "AlertOperations"
    success_canonical = "SendSuccessNotification"
    id_map: dict[str, str] = {fid: fail_canonical for fid in fail_ids}
    id_map.update({sid: success_canonical for sid in success_ids})
    for did in drop_ids:
        id_map[did] = ""

    def remap(i: str | None) -> str | None:
        if not i:
            return None
        if i in id_map:
            return id_map[i] or None
        return i

    seen_fail = False
    seen_success = False
    rebuilt: list[PlanStep] = []
    for s in steps:
        if s.id in drop_ids:
            continue
        new_id = id_map.get(s.id, s.id)
        if new_id == fail_canonical:
            if seen_fail:
                continue
            seen_fail = True
            rebuilt.append(
                PlanStep(
                    id=fail_canonical,
                    service="SNS",
                    description=s.description
                    or "Publish an operations failure alert and stop the workflow.",
                    next=None,
                    on_failure=[],
                    on_timeout=[],
                    branches=[],
                )
            )
            continue
        if new_id == success_canonical:
            if seen_success:
                continue
            seen_success = True
            rebuilt.append(
                PlanStep(
                    id=success_canonical,
                    service="SNS",
                    description=s.description
                    or "Publish a successful pipeline completion notification.",
                    next=None,
                    on_failure=[],
                    on_timeout=[],
                    branches=[],
                )
            )
            continue

        # A step with a populated `branches` list is, by construction,
        # an orchestration node (Parallel / Map / Choice) -- the flow
        # structure stage's own validator (_check_decision_gates_wired /
        # _check_flow_connected in pipeline_analysis_service.py) already
        # enforces that branches only get added there for real control
        # flow. Trust that structure unconditionally instead of
        # re-deciding "is this really a container?" here via a keyword
        # regex against the description: "concurrently", "in parallel",
        # "simultaneously", "at the same time", "all at once", "in
        # tandem" (and any future synonym nobody has written a rule for
        # yet) all describe the exact same shape, and gating on a fixed
        # word list means correctly-wired branches silently get deleted
        # again every time the wording doesn't happen to match one of
        # them -- which is exactly what happened here. A false NEGATIVE
        # (discarding real structure) is much worse than a false
        # positive (keeping branches nothing downstream cares about),
        # so when in doubt, keep them.
        keep_branches = bool(s.branches)
        service = "None" if keep_branches else s.service
        branches = []
        if keep_branches:
            # Labeling is shape-based too, not keyword-based: exactly
            # one branch is structurally a Map's single per-item body
            # (see the identical reasoning in roles.py's MAP_ITERATE
            # override); two or more branches is either exclusive
            # Choice arms (if the conditions themselves use outcome
            # wording like approved/rejected/pass/fail) or Parallel arms
            # otherwise.
            is_map = len(s.branches or []) == 1
            is_choice = (not is_map) and (
                _CHOICE_RE.search(_blob(s))
                or any(
                    re.search(
                        r"\b(pass|fail|failed|failure|good|not good|eligible|"
                        r"approved|rejected|success|succeeded)\b",
                        b.condition or "",
                        re.I,
                    )
                    for b in (s.branches or [])
                )
            )
            for i, b in enumerate(s.branches or []):
                cond = (b.condition or "").strip()
                if is_map:
                    # Map containers get a single "per item" arm, never
                    # the "parallel arm: X" label -- that phrasing implies
                    # N distinct concurrent jobs, which a Map body isn't.
                    cond = cond or "each item"
                elif is_choice:
                    cond = cond or f"outcome {i + 1}"
                else:
                    cond = cond or f"parallel arm {i + 1}"
                    if "parallel" not in cond.lower():
                        cond = f"parallel arm: {b.next}"
                branches.append(PlanBranch(condition=cond, next=b.next))

        rebuilt.append(
            PlanStep(
                id=s.id,
                service=service,
                description=s.description,
                next=s.next,
                on_failure=list(s.on_failure or []),
                on_timeout=list(s.on_timeout or []),
                branches=branches,
            )
        )

    known = {s.id for s in rebuilt}
    final: list[PlanStep] = []
    for s in rebuilt:
        nxt = remap(s.next)
        if nxt and nxt not in known:
            nxt = None
        fails = []
        for f in s.on_failure or []:
            rf = remap(f)
            if rf and rf in known and rf not in fails:
                fails.append(rf)
        timeouts = []
        for t in s.on_timeout or []:
            rt = remap(t)
            if rt and rt in known and rt not in timeouts:
                timeouts.append(rt)
        branches = []
        for b in s.branches or []:
            bn = remap(b.next)
            if bn and bn in known:
                branches.append(PlanBranch(condition=b.condition, next=bn))
        final.append(
            PlanStep(
                id=s.id,
                service=s.service,
                description=s.description,
                next=nxt,
                on_failure=fails,
                on_timeout=timeouts,
                branches=branches,
            )
        )

    return Plan(name=plan.name, summary=plan.summary, steps=final)


# ---------------------------------------------------------------------------
# Layer B — fixed topology for validate → parallel → quality → load
# ---------------------------------------------------------------------------


def _rebuild_validate_parallel_quality_load(plan: Plan) -> Plan | None:
    """If pattern matches, emit one canonical graph. Else return None."""
    steps = list(plan.steps)
    if len(steps) < 4:
        return None

    validate = next((s for s in steps if _is_validate(s)), None)
    parallel = next((s for s in steps if _is_parallel_container(s)), None)
    quality = next((s for s in steps if _is_quality(s)), None)
    load = next((s for s in steps if _is_load(s)), None)
    success = next((s for s in steps if _is_success_notify(s)), None)
    failure = next((s for s in steps if _is_failure_notify(s)), None)

    # Collect workers: explicit parallel branches, else job-like siblings
    workers: list[PlanStep] = []
    if parallel and parallel.branches:
        by_id = {s.id: s for s in steps}
        for b in parallel.branches:
            w = by_id.get(b.next)
            if w and not _is_validate(w) and not _is_failure_notify(w):
                workers.append(w)
    if len(workers) < 2:
        # Heuristic: non-control work steps that aren't validate/quality/load/notify
        for s in steps:
            if s in workers:
                continue
            if s is validate or s is parallel or s is quality or s is load:
                continue
            if _is_failure_notify(s) or _is_success_notify(s) or _is_quality_choice(s):
                continue
            if _is_placeholder_service(s.service):
                continue
            blob = _blob(s).lower()
            if any(
                k in blob
                for k in (
                    "job",
                    "processor",
                    "processjob",
                    "process one",
                    "process two",
                    "independent",
                )
            ) or (s.service or "").lower() in ("glue", "lambda"):
                # avoid treating validate-like lambda twice
                if _is_validate(s):
                    continue
                workers.append(s)
        # Prefer at most the parallel fan-out count
        workers = workers[:4]

    # Pattern gate: need validate + >=2 workers + quality or load intent
    if validate is None or len(workers) < 2:
        return None
    if quality is None and load is None:
        return None

    # If summary/descriptions mention loading to a DB but no load step was
    # extracted, synthesize a load role so structure stays complete.
    summary_blob = f"{plan.name} {plan.summary} " + " ".join(
        _blob(s) for s in steps
    )
    if load is None and re.search(
        r"\b(load|database|dynamo|rds|target db)\b", summary_blob, re.I
    ):
        load = PlanStep(
            id="LoadResults",
            service="Lambda",
            description="Load quality-approved results into the target database.",
            next=None,
            on_failure=[],
            on_timeout=[],
            branches=[],
        )

    # Preserve LLM ids/names; lock only STRUCTURE (roles + edges).
    # Fall back to stable ids only when a role is missing a source step.
    used_ids: set[str] = set()

    def pick_id(source: PlanStep | None, fallback: str) -> str:
        if source and source.id and source.id not in used_ids:
            used_ids.add(source.id)
            return source.id
        base = fallback
        candidate = base
        n = 2
        while candidate in used_ids:
            candidate = f"{base}_{n}"
            n += 1
        used_ids.add(candidate)
        return candidate

    VID = pick_id(validate, "ValidateInput")
    PID = pick_id(parallel, "ProcessInParallel")
    workers = workers[:2]
    worker_ids = [pick_id(w, f"ProcessJob{['One','Two'][i]}") for i, w in enumerate(workers)]
    QID = pick_id(quality, "RunDataQualityCheck") if quality else None
    LID = pick_id(load, "LoadResults") if load else None
    SID = pick_id(success, "SendSuccessNotification")
    FID = pick_id(failure, "AlertOperations")

    has_quality = quality is not None and QID is not None
    has_load = load is not None and LID is not None

    worker_steps = []
    for i, (w, wid) in enumerate(zip(workers, worker_ids)):
        worker_steps.append(
            PlanStep(
                id=wid,
                service=_norm_svc(w.service, "Glue"),
                description=w.description
                or f"Run independent processing job {i + 1}.",
                next=None,
                on_failure=[FID],
                on_timeout=[],
                branches=[],  # structure: workers never carry decision branches
            )
        )

    out: list[PlanStep] = []

    out.append(
        PlanStep(
            id=VID,
            service=_norm_svc(validate.service, "Lambda"),
            description=validate.description
            or "Validate input files. On failure, alert operations.",
            next=PID,
            on_failure=[FID],
            on_timeout=[],
            branches=[],
        )
    )

    join_next = QID if has_quality else (LID if has_load else SID)
    out.append(
        PlanStep(
            id=PID,
            service="None",
            description=(
                parallel.description
                if parallel
                else "Run two independent processing jobs in parallel; continue when both succeed."
            ),
            next=join_next,
            on_failure=[FID],
            on_timeout=[],
            branches=[
                PlanBranch(condition=f"parallel arm: {wid}", next=wid)
                for wid in worker_ids
            ],
        )
    )
    out.extend(worker_steps)

    if has_quality:
        out.append(
            PlanStep(
                id=QID,
                service=_norm_svc(quality.service, "Lambda"),
                description=quality.description
                or "Run data quality check on processing outputs.",
                next=LID if has_load else SID,
                on_failure=[FID],
                on_timeout=[],
                branches=[],  # structure: no Choice — failure via on_failure
            )
        )

    if has_load:
        out.append(
            PlanStep(
                id=LID,
                service=_norm_svc(load.service, "Lambda"),
                description=load.description
                or "Load approved results into the target database.",
                next=SID,
                on_failure=[FID],
                on_timeout=[],
                branches=[],
            )
        )

    out.append(
        PlanStep(
            id=SID,
            service="SNS",
            description=(
                success.description
                if success
                else "Publish a successful pipeline completion notification."
            ),
            next=None,
            on_failure=[],
            on_timeout=[],
            branches=[],
        )
    )

    out.append(
        PlanStep(
            id=FID,
            service="SNS",
            description=(
                failure.description
                if failure
                else "Publish an operations failure alert and stop the workflow."
            ),
            next=None,
            on_failure=[],
            on_timeout=[],
            branches=[],
        )
    )

    return Plan(
        name=plan.name,
        summary=plan.summary,
        steps=out,
    )
