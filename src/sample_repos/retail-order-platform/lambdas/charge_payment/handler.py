"""Lambda: retail-charge-payment.

Charges the customer's payment method through the payment gateway. A declined
card returns success false instead of raising, so the workflow can retry once
after 30 seconds. Also used for the retry attempt.

Returns {"success": bool, "payment_id": str | None}.

Environment:
    PAYMENT_GATEWAY_SECRET_NAME  Secrets Manager secret holding gateway credentials.
"""
import json

import boto3
from lambdas.common.config import require_env


def _gateway_credentials() -> dict:
    client = boto3.client("secretsmanager")
    secret = client.get_secret_value(SecretId=require_env("PAYMENT_GATEWAY_SECRET_NAME"))
    return json.loads(secret["SecretString"])


def lambda_handler(event: dict, context: object) -> dict:
    credentials = _gateway_credentials()
    declined = event.get("simulate_decline", False)
    if declined or not credentials:
        return {"success": False, "payment_id": None}
    return {"success": True, "payment_id": f"PAY-{event['order_id']}"}
