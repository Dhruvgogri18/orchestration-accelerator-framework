"""Glue job: csv-transform-and-load.

Reads a CSV file from S3, normalises the columns and loads the rows into a
DynamoDB table.

Job arguments:
    --source_s3_uri     S3 URI of the CSV file (e.g. s3://retail-orders-raw-dev/uploads/data.csv)
    --dynamodb_table    Target DynamoDB table (e.g. orders-dev)

Downstream: failures are reported by the calling workflow to the
retail-ops-alerts SNS topic.
"""
import sys

from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from pyspark.sql import functions as F


def main() -> None:
    args = getResolvedOptions(sys.argv, ["JOB_NAME", "source_s3_uri", "dynamodb_table"])
    glue_context = GlueContext(SparkContext.getOrCreate())
    job = Job(glue_context)
    job.init(args["JOB_NAME"], args)

    frame = (
        glue_context.spark_session.read.option("header", True).csv(args["source_s3_uri"])
    )
    frame = frame.select(
        F.col("order_id"),
        F.col("customer_id"),
        F.col("total").cast("double").alias("total"),
    ).dropna(subset=["order_id"])

    glue_context.write_dynamic_frame.from_options(
        frame=glue_context.create_dynamic_frame.from_df(frame, glue_context, "orders"),
        connection_type="dynamodb",
        connection_options={"dynamodb.output.tableName": args["dynamodb_table"]},
    )
    job.commit()


if __name__ == "__main__":
    main()
