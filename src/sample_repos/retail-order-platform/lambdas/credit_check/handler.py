"""Lambda: retail-credit-check.

Credit lookup for high-value orders. Runs in parallel with retail-fraud-check.
Returns {"check": "credit", "passed": bool, "approved": bool}.

Environment:
    MIN_CREDIT_SCORE  Minimum acceptable credit score.
"""
import os


def lambda_handler(event: dict, context: object) -> dict:
    minimum = int(os.environ.get("MIN_CREDIT_SCORE", "620"))
    credit_score = int(event.get("credit_score", 700))
    approved = credit_score >= minimum
    return {"check": "credit", "passed": approved, "approved": approved}
