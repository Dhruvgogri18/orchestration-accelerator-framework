"""Lambda: retail-fraud-check.

Fraud lookup for high-value orders (total over 1000). Runs in parallel with
retail-credit-check. Returns {"check": "fraud", "passed": bool, "score": float}.

Environment:
    FRAUD_SCORE_THRESHOLD  Scores at or above this value fail the check.
"""
import os


def lambda_handler(event: dict, context: object) -> dict:
    threshold = float(os.environ.get("FRAUD_SCORE_THRESHOLD", "0.8"))
    score = min(1.0, float(event.get("total", 0)) / 100000)
    return {"check": "fraud", "passed": score < threshold, "score": score}
