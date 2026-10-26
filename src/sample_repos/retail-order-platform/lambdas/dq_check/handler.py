"""Lambda: retail-dq-check.

Data quality gate that runs after orders-etl-job and inventory-etl-job have both
finished. Raises DataQualityError when the row count drifts more than the
allowed fraction from the previous run or when too many order ids are null.

Environment:
    MAX_ROW_COUNT_DRIFT  Allowed fractional change in row count (default 0.20).
    MAX_NULL_RATE        Allowed fraction of null order_id values (default 0.02).
"""
import os


class DataQualityError(Exception):
    """Raised when the curated data fails a quality rule."""


def lambda_handler(event: dict, context: object) -> dict:
    max_drift = float(os.environ.get("MAX_ROW_COUNT_DRIFT", "0.20"))
    max_nulls = float(os.environ.get("MAX_NULL_RATE", "0.02"))
    current, previous = event["row_count"], event["previous_row_count"]
    if previous and abs(current - previous) / previous > max_drift:
        raise DataQualityError("row count drift exceeds the allowed limit")
    if event.get("null_order_id_rate", 0.0) > max_nulls:
        raise DataQualityError("too many null order_id values")
    return {"passed": True, "row_count": current}
