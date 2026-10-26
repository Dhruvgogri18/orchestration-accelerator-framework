"""Lambda: retail-validate-order.

Validates the required fields and the total of a single order.
Returns {"valid": bool, "errors": [...]}.

Environment:
    REQUIRED_FIELDS  Comma-separated list of required fields.
"""
import os


def lambda_handler(event: dict, context: object) -> dict:
    required = os.environ.get("REQUIRED_FIELDS", "order_id,customer_id,total").split(",")
    errors = [f"missing {field}" for field in required if field not in event]
    if "total" in event and float(event["total"]) <= 0:
        errors.append("total must be positive")
    return {"valid": not errors, "errors": errors, "order": event}
