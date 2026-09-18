"""Session plan history — exact restore without re-running extraction.

History is stored on locked_spec['plan_history'] so it survives in the
existing SQLite column without a migration. When locked_spec is cleared
for a structural revise, callers must preserve plan_history via
preserve_history_on_clear().
"""

from __future__ import annotations

from typing import Any

from orchestrator.api.schemas.common import Plan

_MAX = 10


def get_history(locked: dict | None) -> list[dict]:
    if not isinstance(locked, dict):
        return []
    hist = locked.get("plan_history")
    return list(hist) if isinstance(hist, list) else []


def push_plan(locked: dict | None, plan: Plan) -> dict:
    """Return a new locked dict with plan appended to history."""
    base = dict(locked) if isinstance(locked, dict) else {}
    hist = get_history(base)
    dump = plan.model_dump() if hasattr(plan, "model_dump") else plan
    # Avoid consecutive duplicates
    if hist and hist[-1] == dump:
        return base
    hist.append(dump)
    base["plan_history"] = hist[-_MAX:]
    return base


def preserve_history_on_clear(locked: dict | None) -> dict:
    """Keep only plan_history when invalidating downstream artifacts."""
    hist = get_history(locked)
    return {"plan_history": hist} if hist else {}


def previous_plan(locked: dict | None, current: Plan | None = None) -> Plan | None:
    """Return the plan before the current one, if any."""
    hist = get_history(locked)
    if not hist:
        return None
    if current is not None:
        cur = current.model_dump() if hasattr(current, "model_dump") else current
        # Walk from end: skip entries equal to current, return the one before
        for entry in reversed(hist):
            if entry != cur:
                try:
                    return Plan.model_validate(entry)
                except Exception:  # noqa: BLE001
                    continue
        return None
    if len(hist) < 2:
        # Only one snapshot — treat it as restorable previous if current differs
        try:
            return Plan.model_validate(hist[-1])
        except Exception:  # noqa: BLE001
            return None
    try:
        return Plan.model_validate(hist[-2])
    except Exception:  # noqa: BLE001
        return None
