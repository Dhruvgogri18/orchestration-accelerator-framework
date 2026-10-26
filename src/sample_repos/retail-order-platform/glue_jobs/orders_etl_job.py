"""Glue job: orders-etl-job.

Cleans and de-duplicates raw order batches and writes curated Parquet.
Runs in parallel with inventory-etl-job in the daily-batch-etl pipeline; the
retail-dq-check Lambda runs after both have finished.

Job arguments:
    --input_path    Raw orders prefix (s3://retail-orders-raw-<env>/incoming/)
    --output_path   Curated orders prefix (s3://retail-orders-curated-<env>/orders/)
"""
import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext


def main() -> None:
    args = getResolvedOptions(sys.argv, ["JOB_NAME", "input_path", "output_path"])
    glue_context = GlueContext(SparkContext.getOrCreate())
    job = Job(glue_context)
    job.init(args["JOB_NAME"], args)

    orders = glue_context.spark_session.read.json(args["input_path"])
    curated = orders.dropDuplicates(["order_id"]).filter("total >= 0")
    curated.write.mode("overwrite").parquet(args["output_path"])
    job.commit()


if __name__ == "__main__":
    main()
