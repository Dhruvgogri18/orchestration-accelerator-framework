"""Lambda: retail-release-inventory.

Rollback step after a payment failed twice: deletes the reservation created by
retail-reserve-inventory and alerts operations on the retail-ops-alerts topic.

Environment:
    RESERVATIONS_TABLE  DynamoDB table (inventory-reservations-<env>).
    OPS_TOPIC_ARN       SNS topic for operations alerts (retail-ops-alerts-<env>).
"""
import boto3
from lambdas.common.config import require_env


def lambda_handler(event: dict, context: object) -> dict:
    table = boto3.resource("dynamodb").Table(require_env("RESERVATIONS_TABLE"))
    table.delete_item(Key={"reservation_id": event["reservation_id"]})
    boto3.client("sns").publish(
        TopicArn=require_env("OPS_TOPIC_ARN"),
        Subject="Payment failed - inventory released",
        Message=f"Order {event['order_id']} payment failed twice; reservation released.",
    )
    return {"released": True, "order_id": event["order_id"]}
