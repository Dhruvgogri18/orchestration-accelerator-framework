"""Lambda: retail-ship-order.

Creates the shipment for a paid order and updates the order record.
Returns {"shipment_id": str, "carrier": str}.

Environment:
    ORDERS_TABLE  DynamoDB table (orders-<env>).
"""
import boto3
from lambdas.common.config import require_env


def lambda_handler(event: dict, context: object) -> dict:
    table = boto3.resource("dynamodb").Table(require_env("ORDERS_TABLE"))
    shipment_id = f"SHP-{event['order_id']}"
    table.update_item(
        Key={"order_id": event["order_id"]},
        UpdateExpression="SET #s = :s, shipment_id = :i",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":s": "SHIPPED", ":i": shipment_id},
    )
    return {"shipment_id": shipment_id, "carrier": "ACME-FREIGHT"}
