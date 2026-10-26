"""Lambda: retail-load-curated.

Loads curated order rows from S3 into the DynamoDB orders table. Runs after
retail-dq-check passes.

Environment:
    ORDERS_TABLE    DynamoDB table (orders-<env>).
    CURATED_BUCKET  Curated bucket (retail-orders-curated-<env>).
"""
import json

import boto3
from lambdas.common.config import require_env


def lambda_handler(event: dict, context: object) -> dict:
    s3 = boto3.client("s3")
    table = boto3.resource("dynamodb").Table(require_env("ORDERS_TABLE"))
    body = s3.get_object(Bucket=require_env("CURATED_BUCKET"), Key=event["key"])["Body"]
    rows = json.loads(body.read())
    with table.batch_writer() as batch:
        for row in rows:
            batch.put_item(Item=row)
    return {"loaded": len(rows)}
