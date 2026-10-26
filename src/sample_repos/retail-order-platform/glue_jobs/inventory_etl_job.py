"""Glue job: inventory-etl-job.

Builds the latest inventory snapshot per SKU and writes curated Parquet.
Runs in parallel with orders-etl-job; retail-dq-check runs afterwards.

Job arguments:
    --input_path    Raw inventory prefix (s3://retail-orders-raw-<env>/inventory/)
    --output_path   Curated inventory prefix (s3://retail-orders-curated-<env>/inventory/)
"""
import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import Window
from pyspark.sql import functions as F


def main() -> None:
    args = getResolvedOptions(sys.argv, ["JOB_NAME", "input_path", "output_path"])
    glue_context = GlueContext(SparkContext.getOrCreate())
    job = Job(glue_context)
    job.init(args["JOB_NAME"], args)

    stock = glue_context.spark_session.read.json(args["input_path"])
    latest = Window.partitionBy("sku").orderBy(F.col("updated_at").desc())
    snapshot = stock.withColumn("rank", F.row_number().over(latest)).filter("rank = 1")
    snapshot.drop("rank").write.mode("overwrite").parquet(args["output_path"])
    job.commit()


if __name__ == "__main__":
    main()
