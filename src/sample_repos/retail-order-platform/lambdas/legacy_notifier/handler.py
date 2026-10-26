"""Lambda: retail-legacy-notifier.

DEPRECATED. Replaced by the retail-ops-alerts-<env> SNS topic. Do not use this
function or the retail-ops-alerts-legacy topic in new workflows.

Environment:
    TOPIC_ARN  The legacy topic arn:aws:sns:us-east-1:123456789012:retail-ops-alerts-legacy.
"""
import boto3
from lambdas.common.config import require_env


def lambda_handler(event: dict, context: object) -> dict:
    boto3.client("sns").publish(TopicArn=require_env("TOPIC_ARN"), Message=str(event))
    return {"sent": True}
