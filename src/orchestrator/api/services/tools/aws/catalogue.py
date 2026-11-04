"""Static service catalogue used by describe_service and list_aws_services.

This is *knowledge*, not live AWS data.  It tells the LLM what resource
types and identifier fields exist so it can form correct tool calls
without hallucinating API shapes.
"""

from __future__ import annotations

from ..schemas import ServiceInfo

SERVICE_CATALOGUE: dict[str, ServiceInfo] = {
    "lambda": ServiceInfo(
        id="lambda",
        name="AWS Lambda",
        category="compute",
        resource_types=["function", "layer", "event_source_mapping"],
        common_operations=["list_functions", "get_function", "get_function_configuration", "list_layers"],
        identifier_fields=["FunctionName", "FunctionArn"],
        notes="Primary compute target for Step Functions Task states.",
    ),
    "s3": ServiceInfo(
        id="s3",
        name="Amazon S3",
        category="storage",
        resource_types=["bucket", "object"],
        common_operations=["list_buckets", "list_objects_v2", "get_bucket_location", "get_bucket_tagging"],
        identifier_fields=["Bucket", "Key"],
        notes="Buckets are global; objects are region-scoped via the bucket.",
    ),
    "dynamodb": ServiceInfo(
        id="dynamodb",
        name="Amazon DynamoDB",
        category="storage",
        resource_types=["table"],
        common_operations=["list_tables", "describe_table", "list_tags_of_resource"],
        identifier_fields=["TableName", "TableArn"],
    ),
    "sqs": ServiceInfo(
        id="sqs",
        name="Amazon SQS",
        category="messaging",
        resource_types=["queue"],
        common_operations=["list_queues", "get_queue_attributes", "list_queue_tags"],
        identifier_fields=["QueueUrl", "QueueArn"],
    ),
    "sns": ServiceInfo(
        id="sns",
        name="Amazon SNS",
        category="messaging",
        resource_types=["topic", "subscription"],
        common_operations=["list_topics", "get_topic_attributes", "list_subscriptions"],
        identifier_fields=["TopicArn"],
    ),
    "states": ServiceInfo(
        id="states",
        name="AWS Step Functions",
        category="orchestration",
        resource_types=["stateMachine", "activity", "execution"],
        common_operations=["list_state_machines", "describe_state_machine", "list_activities"],
        identifier_fields=["stateMachineArn", "name"],
        notes="Existing state machines can be composed or used as reference patterns.",
    ),
    "glue": ServiceInfo(
        id="glue",
        name="AWS Glue",
        category="analytics",
        resource_types=["job", "crawler", "database", "table"],
        common_operations=["list_jobs", "get_job", "get_crawler", "get_databases"],
        identifier_fields=["JobName", "Name"],
    ),
    "ecs": ServiceInfo(
        id="ecs",
        name="Amazon ECS",
        category="compute",
        resource_types=["cluster", "service", "task_definition"],
        common_operations=["list_clusters", "list_services", "describe_task_definition", "list_task_definitions"],
        identifier_fields=["cluster", "service", "taskDefinition"],
    ),
    "events": ServiceInfo(
        id="events",
        name="Amazon EventBridge",
        category="integration",
        resource_types=["rule", "event_bus"],
        common_operations=["list_rules", "describe_rule", "list_event_buses"],
        identifier_fields=["Name", "EventBusName"],
    ),
    "kinesis": ServiceInfo(
        id="kinesis",
        name="Amazon Kinesis Data Streams",
        category="messaging",
        resource_types=["stream"],
        common_operations=["list_streams", "describe_stream", "list_tags_for_stream"],
        identifier_fields=["StreamName", "StreamARN"],
    ),
    "firehose": ServiceInfo(
        id="firehose",
        name="Amazon Kinesis Data Firehose",
        category="messaging",
        resource_types=["delivery_stream"],
        common_operations=["list_delivery_streams", "describe_delivery_stream"],
        identifier_fields=["DeliveryStreamName"],
    ),
    "batch": ServiceInfo(
        id="batch",
        name="AWS Batch",
        category="compute",
        resource_types=["job_queue", "job_definition", "compute_environment"],
        common_operations=["describe_job_queues", "describe_job_definitions", "describe_compute_environments"],
        identifier_fields=["jobQueueName", "jobDefinitionName"],
    ),
    "sagemaker": ServiceInfo(
        id="sagemaker",
        name="Amazon SageMaker",
        category="analytics",
        resource_types=["endpoint", "model", "training_job", "pipeline"],
        common_operations=["list_endpoints", "list_models", "list_pipelines", "describe_endpoint"],
        identifier_fields=["EndpointName", "ModelName", "PipelineName"],
    ),
    "iam": ServiceInfo(
        id="iam",
        name="AWS IAM",
        category="security",
        resource_types=["role", "policy"],
        common_operations=["list_roles", "get_role", "list_attached_role_policies"],
        identifier_fields=["RoleName", "Arn"],
        notes="Read-only. Used to discover roles that Step Functions can assume.",
    ),
    "secretsmanager": ServiceInfo(
        id="secretsmanager",
        name="AWS Secrets Manager",
        category="security",
        resource_types=["secret"],
        common_operations=["list_secrets", "describe_secret"],
        identifier_fields=["Name", "ARN"],
        notes="Never returns secret *values* — only metadata.",
    ),
    "ssm": ServiceInfo(
        id="ssm",
        name="AWS Systems Manager Parameter Store",
        category="security",
        resource_types=["parameter"],
        common_operations=["describe_parameters", "get_parameter"],
        identifier_fields=["Name"],
        notes="get_parameter may return values; treat as sensitive in logs.",
    ),
    "cloudformation": ServiceInfo(
        id="cloudformation",
        name="AWS CloudFormation",
        category="orchestration",
        resource_types=["stack"],
        common_operations=["list_stacks", "describe_stacks", "list_stack_resources"],
        identifier_fields=["StackName", "StackId"],
    ),
    "ecr": ServiceInfo(
        id="ecr",
        name="Amazon ECR",
        category="compute",
        resource_types=["repository"],
        common_operations=["describe_repositories", "list_images"],
        identifier_fields=["repositoryName", "repositoryArn"],
    ),
    "logs": ServiceInfo(
        id="logs",
        name="Amazon CloudWatch Logs",
        category="observability",
        resource_types=["log_group"],
        common_operations=["describe_log_groups", "describe_log_streams"],
        identifier_fields=["logGroupName"],
    ),
}


CATEGORY_ORDER = (
    "compute",
    "storage",
    "messaging",
    "orchestration",
    "analytics",
    "integration",
    "security",
    "observability",
)


def list_catalogue(category: str | None = None) -> list[ServiceInfo]:
    items = list(SERVICE_CATALOGUE.values())
    if category:
        cat = category.strip().lower()
        items = [s for s in items if s.category == cat]
    # Stable order for determinism
    items.sort(key=lambda s: (CATEGORY_ORDER.index(s.category) if s.category in CATEGORY_ORDER else 99, s.id))
    return items


def get_service_info(service: str) -> ServiceInfo | None:
    from .allowlist import canonical_service
    return SERVICE_CATALOGUE.get(canonical_service(service))
