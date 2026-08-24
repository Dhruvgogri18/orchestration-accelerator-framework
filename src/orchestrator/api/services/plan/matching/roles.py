"""Role taxonomy + deterministic role tagger.

This replaces the single-family regex classifier (`_norm_step_id` in the
existing `extraction_service.py`) with something that generalizes across
*any* workflow archetype, not just validate->parallel->quality->load.

Design goal: paraphrase invariance. Two steps that mean the same thing
("check the file's valid" vs "validate the input") must get the same
role even though they share almost no literal tokens.

How it stays deterministic while being paraphrase-robust:
  - Every role has a canonical description (a sentence a human would use
    to explain that role) plus a keyword set.
  - A step's role is scored against every role's description using
    token-Jaccard similarity, boosted by keyword hits. This is the same
    fallback pattern already used in `semantic_cache_service.py`.
  - `embed_fn` is an optional hook: pass a real embedding function
    (cosine similarity) and scoring switches to that automatically, with
    ZERO caller-visible change -- same pattern as the semantic cache's
    "if gateway_client exposes get_embedding()" comment.
  - `llm_classify_fn` is a second, narrower hook, added after real-world
    testing showed keyword scoring alone genuinely struggles on a small
    slice of steps: generic action wording with no distinguishing verb
    ("carry out the approved action" vs. "ship the order"), and can be
    confidently wrong rather than usefully unsure -- so it can't be
    gated behind a low-confidence threshold. When given, it becomes the
    source of truth for the content-based part of the role decision (one
    constrained "pick a name from this list" call, temp=0); structural
    rules -- decision gates, fan-out, notify, terminal -- stay exactly
    as before, since those are shape-based and already reliable without
    any LLM. Without the hook (e.g. running offline), keyword scoring is
    still the fallback, so the module works standalone either way.
  - Ties are broken alphabetically by role name, so scoring is a pure
    function of the step content: same input, same role, always.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    # Split camelCase / snake_case ids too, e.g. "ValidateInputFile" ->
    # {"validate", "input", "file"}
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text or "")
    return set(_WORD_RE.findall(text.lower()))


@dataclass(frozen=True)
class Role:
    name: str
    description: str
    keywords: frozenset[str]


ROLES: dict[str, Role] = {
    r.name: r
    for r in [
        Role("INGEST", "Read or receive incoming input data or an event that starts the workflow",
             frozenset({"read", "receive", "fetch", "ingest", "incoming", "trigger", "download", "poll", "consume"})),
        Role("VALIDATE", "Validate or check the input for correctness before proceeding",
             frozenset({"validate", "valid", "check", "verify", "schema", "format", "sanity"})),
        Role("TRANSFORM", "Process transform or convert the data into a new form",
             frozenset({"process", "transform", "convert", "parse", "glue", "spark", "etl", "compute", "run", "job"})),
        Role("ENRICH", "Enrich the data with additional information from another source",
             frozenset({"enrich", "augment", "lookup", "supplement"})),
        Role("QUALITY_CHECK", "Run a data quality or business rule check on the processed output",
             frozenset({"quality", "dq", "audit", "reconcile", "rule"})),
        Role("DECISION", "Evaluate a condition and route the workflow down one of several exclusive branches",
             frozenset({"evaluate", "decide", "route", "choice", "approve", "reject", "branch", "eligib"})),
        Role("FAN_OUT", "Run two or more independent jobs at the same time in parallel",
             frozenset({"parallel", "concurrently", "concurrent", "simultaneously", "fanout"})),
        Role("WORKER", "One independent unit of work executed inside a parallel fan out",
             frozenset({"worker", "processor", "independent"})),
        Role("AGGREGATE", "Combine or join the results of parallel branches once they all complete",
             frozenset({"aggregate", "combine", "join", "merge", "collect", "gather"})),
        Role("PERSIST", "Write or load the final results into a database table or storage target",
             frozenset({"load", "write", "save", "persist", "store", "insert", "dynamo", "rds", "database", "table", "target"})),
        Role("MAP_ITERATE", "Iterate over a collection of items and process each one such as an array or list of files",
             frozenset({"each", "every", "iterate", "map", "collection", "batch", "items", "list"})),
        Role("RETRY_WAIT", "Wait pause or retry after a delay before continuing",
             frozenset({"wait", "retry", "delay", "pause", "backoff"})),
        Role("COMPENSATE", "Undo or roll back previously completed work because a later step failed",
             frozenset({"rollback", "compensate", "undo", "revert", "cancel"})),
        Role("EXTERNAL_CALL", "Invoke an external service api or third party system",
             frozenset({"invoke", "call", "api", "webhook", "charge", "ship", "reserve", "third"})),
        Role("ACTION", "Carry out a generic action step with no more specific role",
             frozenset({"carry", "proceed", "continue", "execute", "perform", "handle", "action"})),
        Role("NOTIFY_SUCCESS", "Send a notification announcing that the workflow completed successfully",
             frozenset({"success", "completion", "confirm", "publish"})),
        Role("NOTIFY_FAILURE", "Send a notification or alert that the workflow or a step failed",
             frozenset({"failure", "alert", "ops", "fail", "error", "page"})),
        Role("TERMINAL", "A pure end marker with no further work either succeed or fail the workflow",
             frozenset({"end", "succeed", "terminate", "finish", "complete"})),
    ]
}

_PLACEHOLDER_SERVICE = {"", "none", "n/a", "na", "null", "control", "terminal", "pass"}
_NOTIFY_SERVICES = {"sns", "amazon sns", "email", "ses", "slack"}

# Structural-only roles: reachable ONLY via a structural override below,
# never via generic text scoring. Both are properties of the GRAPH SHAPE
# around a step (does it have exclusive branches? do multiple steps feed
# from one container?), not of its wording -- a step whose description
# happens to contain "approved" isn't a decision gate unless it actually
# branches. Letting text score them caused false positives on ordinary
# action steps that merely mention an outcome word.
_STRUCTURAL_ONLY_ROLES = {"DECISION", "WORKER"}

_POSITIVE = {"approve", "eligib", "pass", "good", "success", "valid", "accept"}
_NEGATIVE = {"reject", "fail", "bad", "invalid", "denied", "declined"}
_TIMEOUT = {"timeout", "expired"}


def condition_bucket(condition: str) -> str:
    """Normalize a branch condition string into a small, deterministic
    set of buckets so wording ("approved" vs "eligible") collapses onto
    the same edge signature. Shared by the tagger and the signature
    builder so they never disagree.
    """
    c = (condition or "").lower()
    if any(k in c for k in _TIMEOUT):
        return "timeout"
    if any(k in c for k in _NEGATIVE):
        return "negative"
    if any(k in c for k in _POSITIVE):
        # "not eligible" contains the positive keyword "eligib" -- a
        # negation flips the bucket rather than being ignored.
        negated = re.search(r"\bnot\b|\bn't\b|\bno\b", c) is not None
        return "negative" if negated else "positive"
    return "other"

EmbedFn = Optional[Callable[[str], list[float]]]

# Takes (blob_text, candidate_role_names) -> one role name from that
# list. Called only for the low-confidence case below. Keep the real
# implementation this simple too: one gateway_client.chat_completion()
# call, temp=0, prompt = "pick one of these role names for this step",
# nothing fancier. See README for the exact wiring.
LlmClassifyFn = Optional[Callable[[str, list[str]], str]]

AMBIGUOUS_THRESHOLD = 0.20


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _score(blob_tokens: set[str], blob_text: str, role: Role, embed_fn: EmbedFn) -> float:
    if embed_fn is not None:
        # Pluggable real-embedding path -- swaps in automatically, no
        # caller-visible change, same pattern as semantic_cache_service.
        return _cosine(embed_fn(blob_text), embed_fn(role.description))
    desc_tokens = _tokenize(role.description)
    union = blob_tokens | desc_tokens
    jaccard = len(blob_tokens & desc_tokens) / len(union) if union else 0.0
    keyword_hits = sum(1 for t in blob_tokens if any(t.startswith(k) for k in role.keywords))
    return jaccard + 0.35 * keyword_hits


@dataclass
class TaggedStep:
    step_id: str
    role: str
    confidence: float


def tag_step(
    step_id: str,
    service: str | None,
    description: str | None,
    has_branches: bool,
    branch_conditions: list[str],
    has_incoming: bool,
    has_outgoing: bool,
    embed_fn: EmbedFn = None,
    llm_classify_fn: LlmClassifyFn = None,
) -> TaggedStep:
    """Deterministically assign one role to a step.

    Structural signals (branch shape, placeholder service, dead-end with
    no edges) are checked first because they're unambiguous and free --
    text scoring only breaks ties among the remaining plausible roles.
    """
    svc = (service or "").strip().lower()
    blob_text = f"{step_id} {service or ''} {description or ''}"
    tokens = _tokenize(blob_text)

    # --- structural overrides (cheap, unambiguous, checked first) ---
    if not has_outgoing and not has_branches and svc in _PLACEHOLDER_SERVICE:
        term_tokens = _tokenize(f"{step_id} {description or ''}")
        if term_tokens & ROLES["TERMINAL"].keywords or re.match(
            r"^(succeed|fail|complete|terminate|end)", step_id, re.I
        ):
            return TaggedStep(step_id, "TERMINAL", 1.0)

    if has_branches and len(branch_conditions) >= 2:
        buckets = {condition_bucket(c) for c in branch_conditions}
        if buckets - {"other"}:
            # Exclusive outcomes (approved/rejected, pass/fail...) -> a
            # real decision gate.
            return TaggedStep(step_id, "DECISION", 0.95)
        # Branches with no exclusive-outcome wording ("parallel arm: X")
        # are fan-out arms, not a choice.
        return TaggedStep(step_id, "FAN_OUT", 0.9)

    if has_branches and len(branch_conditions) == 1:
        # A container with exactly ONE outgoing branch is structurally a
        # Map: one reusable per-item body executed once per element of a
        # collection, not N distinct parallel jobs (that's >=2 branches,
        # handled above as FAN_OUT). This must be decided by shape, not
        # wording -- "process every file concurrently" and "process the
        # record" both contain "process", so text scoring alone can route
        # the container into TRANSFORM or FAN_OUT depending on which
        # keyword happens to win, which is wrong and non-deterministic
        # across paraphrases. Branch count is unambiguous and free.
        return TaggedStep(step_id, "MAP_ITERATE", 0.9)

    if svc in _NOTIFY_SERVICES:
        is_failure = any(k in blob_text.lower() for k in ("fail", "alert", "ops", "error"))
        is_success = any(k in blob_text.lower() for k in ("success", "complet", "confirm"))
        if is_failure and not is_success:
            return TaggedStep(step_id, "NOTIFY_FAILURE", 0.9)
        if is_success and not is_failure:
            return TaggedStep(step_id, "NOTIFY_SUCCESS", 0.9)

    # --- content-based role: this is the part that's genuinely ambiguous
    # from wording alone. If a classifier was given, it decides -- one
    # constrained call, always used here, no threshold to dodge. Without
    # one (e.g. running offline), fall back to keyword/embedding scoring.
    candidates = sorted(n for n in ROLES if n not in _STRUCTURAL_ONLY_ROLES)
    if llm_classify_fn is not None:
        choice = llm_classify_fn(blob_text, candidates)
        best_name = choice if choice in candidates else "ACTION"
        return TaggedStep(step_id, best_name, 0.85 if choice in candidates else 0.0)

    scores = {name: _score(tokens, blob_text, ROLES[name], embed_fn) for name in candidates}
    ranked = sorted(scores, key=lambda n: (-scores[n], n))
    best_name = ranked[0]
    # Weak/ambiguous signal (no role clearly fits) -> generic ACTION
    # rather than committing to whichever noisy role happened to score
    # marginally highest.
    if scores[best_name] < AMBIGUOUS_THRESHOLD:
        best_name = "ACTION"
    return TaggedStep(step_id, best_name, scores[best_name])
