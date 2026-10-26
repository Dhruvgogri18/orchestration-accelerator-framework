# Data contracts

## Order batch (`incoming/*.json`)

A JSON array. Each element:

```json
{
  "order_id": "ORD-1001",
  "customer_id": "CUST-77",
  "total": 1249.50,
  "currency": "USD",
  "items": [{"sku": "SKU-1", "quantity": 2}]
}
```

## Lambda responses

| Function | Response fields |
|---|---|
| `retail-validate-order` | `valid` (bool), `errors` (list) |
| `retail-fraud-check` | `check` = "fraud", `passed` (bool), `score` |
| `retail-credit-check` | `check` = "credit", `passed` (bool), `approved` (bool) |
| `retail-reserve-inventory` | `reservation_id`, `reserved` (bool) |
| `retail-charge-payment` | `success` (bool), `payment_id` |
| `retail-ship-order` | `shipment_id`, `carrier` |

## CSV ingest

`csv-transform-and-load` expects a header row with `order_id,customer_id,total`.
