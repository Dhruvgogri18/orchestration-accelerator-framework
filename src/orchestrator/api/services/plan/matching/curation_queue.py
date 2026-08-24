"""Human-review queue for prompts the template library couldn't confidently
match. This is what keeps the "almost 100%" claim honest over time -- the
library is expected to grow from real traffic, not be finished once.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from orchestrator.api.schemas.common import Plan

QUEUE_PATH = os.environ.get("PLAN_MATCH_CURATION_QUEUE_PATH", "curation_queue.jsonl")


def log_unmatched(
    plan: Plan,
    signature_hash: str,
    nearest_template: str | None,
    score: float,
    session_id: str | None = None,
    user_message: str | None = None,
) -> None:
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "user_message": user_message,
        "plan_name": plan.name,
        "plan_summary": plan.summary,
        "signature_hash": signature_hash,
        "nearest_template": nearest_template,
        "nearest_score": round(score, 3),
        "step_count": len(plan.steps),
    }
    with open(QUEUE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")