# Architecture

## Data flow

1. Partners drop order batches into `s3://retail-orders-raw-<env>/incoming/` as
   JSON arrays (see `data_contracts.md`).
2. The `order-fulfillment` pipeline reads the batch and processes every order
   independently and concurrently.
3. Per order:
   - `retail-validate-order` checks required fields and totals.
   - Orders with `total > 1000` also run `retail-fraud-check` and
     `retail-credit-check` in parallel.
   - `retail-reserve-inventory` reserves stock in `inventory-reservations-<env>`.
   - `retail-charge-payment` charges the customer.
   - On success `retail-ship-order` creates the shipment.
   - If the charge fails twice, `retail-release-inventory` returns the stock and
     the failure is published to `retail-ops-alerts-<env>`.
4. When every order has finished, `retail-aggregate-report` writes the summary
   report to `s3://retail-reports-<env>/summaries/` and publishes it to
   `retail-order-summary-<env>`.

## Batch analytics

Two Glue jobs read raw orders and inventory snapshots, write Parquet to
`s3://retail-orders-curated-<env>/`, and are followed by `retail-dq-check`, a
Lambda that fails the pipeline when row counts or null rates drift.
`retail-load-curated` then writes the curated rows into the DynamoDB table
`orders-<env>`.

## DynamoDB tables

| Table | Key | Purpose |
|---|---|---|
| `orders-<env>` | `order_id` | Current state of each order |
| `inventory-reservations-<env>` | `reservation_id` | Stock held for an order |
| `order-outcomes-<env>` | `order_id` | Final outcome per order for reporting |
