# Runbook

## Who gets alerted

All operational failures are published to the SNS topic `retail-ops-alerts-<env>`
(dev: `arn:aws:sns:us-east-1:123456789012:retail-ops-alerts-dev`). The on-call
engineer for the Retail Data Platform team subscribes by e-mail.

> **Deprecated:** `retail-ops-alerts-legacy` was used by the old notifier in
> `lambdas/legacy_notifier`. Do not use it for new workflows.

## Failed payment

`retail-charge-payment` returns `success: false` when the card is declined.
The order workflow waits 30 seconds and retries once. If the retry also fails:

1. `retail-release-inventory` releases the reservation.
2. A message is published to `retail-ops-alerts-<env>`.
3. The order outcome is written to `order-outcomes-<env>` with status `PAYMENT_FAILED`.

## Data quality check fails

`retail-dq-check` raises `DataQualityError` when the row count differs from the
previous run by more than 20 percent, or when more than 2 percent of `order_id`
values are null. Re-run `orders-etl-job` after fixing the source file.

## Re-running a Glue job

Start the job with the same arguments from `config/jobs.yaml`. Glue job names are
the same in every environment; only the arguments change.
