# retail-order-platform

Order-processing data platform for the Retail Digital team. Orders arrive as JSON
batches in S3, are validated and enriched by Lambda functions, and are loaded
into DynamoDB. Batch analytics run as AWS Glue jobs. Operations are alerted
through SNS.

> Sample repository used to test repository ingestion and retrieval for the
> Orchestration Accelerator Framework. All names, account ids and ARNs are fictional.

## Layout

| Path | What lives there |
|---|---|
| `glue_jobs/` | AWS Glue (PySpark) job scripts |
| `lambdas/` | One folder per Lambda function, entry point `handler.lambda_handler` |
| `config/jobs.yaml` | Catalog of every Glue job and Lambda: script, entry point, inputs, outputs, downstream |
| `config/environments/` | Per-environment resource names and ARNs (`dev`, `prod`) |
| `config/pipelines.json` | Which jobs make up each pipeline, in order |
| `docs/` | Architecture, runbook and data contracts |

## Environments

| Environment | AWS account | Region |
|---|---|---|
| dev | 123456789012 | us-east-1 |
| prod | 210987654321 | us-east-1 |

Resource names end in `-dev` or `-prod`. Always read ARNs from
`config/environments/<env>.yaml` instead of hardcoding them.

## Pipelines

- **order-fulfillment** - validate each order, run fraud and credit checks for
  high-value orders (total over 1000), reserve inventory, charge payment, ship.
  Failed payments are retried once after 30 seconds, then inventory is released
  and operations are notified.
- **daily-batch-etl** - run `orders-etl-job` and `inventory-etl-job` in parallel,
  run the data quality check, then load curated data into DynamoDB.
- **csv-ingest** - load a CSV file from S3 into DynamoDB with the
  `csv-transform-and-load` Glue job.

## Alerting

Failures go to the SNS topic `retail-ops-alerts-<env>`. Business summaries go to
`retail-order-summary-<env>`. The topic `retail-ops-alerts-legacy` is deprecated.
See `docs/runbook.md`.
