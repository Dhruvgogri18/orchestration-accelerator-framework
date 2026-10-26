"""Lambda: retail-aggregate-report.

Runs once after every order in a batch has finished. Aggregates the per-order
outcomes into one summary report, writes it to S3 and publishes it to the
retail-order-summary SNS topic.

Environment:
    REPORT_BUCKET       Reports bucket (retail-reports-<env>); report goes to summaries/.
    SUMMARY_TOPIC_ARN   SNS topic retail-order-summary-<env>.
"""
import json
from collections import Counter

import boto3
from lambdas.common.config import require_env


def lambda_handler(event: list, context: object) -> dict:
    outcomes = Counter(item.get("status", "UNKNOWN") for item in event)
    report = {"total_orders": len(event), "by_status": dict(outcomes)}
    boto3.client("s3").put_object(
        Bucket=require_env("REPORT_BUCKET"),
        Key="summaries/latest.json",
        Body=json.dumps(report).encode("utf-8"),
    )
    boto3.client("sns").publish(
        TopicArn=require_env("SUMMARY_TOPIC_ARN"),
        Subject="Order batch summary",
        Message=json.dumps(report),
    )
    return report
