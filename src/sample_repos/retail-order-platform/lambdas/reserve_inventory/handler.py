"""Lambda: retail-reserve-inventory.

Reserves stock for an order in DynamoDB before payment is charged.
Returns {"reservation_id": str, "reserved": bool}.

Environment:
    RESERVATIONS_TABLE  DynamoDB table (inventory-reservations-<env>).
"""
import uuid

import boto3
from lambdas.common.config import require_env


def lambda_handler(event: dict, context: object) -> dict:
    table = boto3.resource("dynamodb").Table(require_env("RESERVATIONS_TABLE"))
    reservation_id = f"RES-{uuid.uuid4().hex[:12]}"
    table.put_item(
        Item={
            "reservation_id": reservation_id,
            "order_id": event["order_id"],
            "items": event.get("items", []),
        }
    )
    return {"reservation_id": reservation_id, "reserved": True}
