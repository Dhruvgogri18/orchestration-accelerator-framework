# Sample repositories

Fictional repositories used to develop and test Part B (repository ingestion and
retrieval). They are **not** part of the application. Do not import from them.

| Folder | Domain | Contents |
|---|---|---|
| `retail-order-platform/` | Order processing and batch analytics | 3 Glue jobs, 11 Lambda functions (10 active, 1 deprecated), job catalog, per-environment resource config, docs |

Point ingestion at one repo at a time:

```
REPO_PATH=sample_repos/retail-order-platform
```

All account ids, ARNs and bucket names are fictional (accounts `123456789012` for
dev and `210987654321` for prod). There are no credentials in these repos.

## `retail-order-platform`: what retrieval should be able to find

Use these as spot-check questions when tuning chunking and retrieval. Each row names
the artifact the right chunk should come from.

| Question | Expected answer | Source |
|---|---|---|
| Which Glue job loads a CSV from S3 into DynamoDB? | `csv-transform-and-load` | `config/jobs.yaml`, `glue_jobs/csv_transform_and_load.py` |
| What arguments does that job take? | `--source_s3_uri`, `--dynamodb_table` | same |
| Which SNS topic receives operational failure alerts (dev)? | `arn:aws:sns:us-east-1:123456789012:retail-ops-alerts-dev` | `config/environments/dev.yaml`, `docs/runbook.md` |
| Which topic publishes the batch summary report? | `retail-order-summary-dev` | `config/environments/dev.yaml` |
| Which topic is deprecated? | `retail-ops-alerts-legacy` | `docs/runbook.md`, `lambdas/legacy_notifier/handler.py` |
| Which function validates an order and what does it return? | `retail-validate-order`; `valid`, `errors` | `lambdas/validate_order/handler.py`, `docs/data_contracts.md` |
| Which functions run the high-value checks? | `retail-fraud-check`, `retail-credit-check` | `docs/architecture.md`, `config/pipelines.json` |
| What does the payment function return on a decline? | `success: false`, no exception | `lambdas/charge_payment/handler.py` |
| What happens after a payment fails twice? | `retail-release-inventory`, then alert ops | `docs/runbook.md`, `config/pipelines.json` |
| Which Glue jobs run in parallel in the daily ETL? | `orders-etl-job`, `inventory-etl-job` | `config/pipelines.json` |
| What runs after both Glue jobs? | `retail-dq-check`, then `retail-load-curated` | `config/pipelines.json` |
| Which DynamoDB table holds stock reservations? | `inventory-reservations-<env>` | `config/environments/*.yaml` |
| What is the prod ARN of the charge-payment function? | `arn:aws:lambda:us-east-1:210987654321:function:retail-charge-payment` | `config/environments/prod.yaml` |

### Deliberate traps

- **Dev vs prod:** the two environment files have identical keys and different
  values. A good retriever filters by environment instead of mixing them.
- **Deprecated resources:** `retail-legacy-notifier` and `retail-ops-alerts-legacy`
  look like valid targets but are marked deprecated.
- **Similar names:** `retail-ops-alerts` (failures) and `retail-order-summary`
  (business summary) are both SNS topics with different purposes.
