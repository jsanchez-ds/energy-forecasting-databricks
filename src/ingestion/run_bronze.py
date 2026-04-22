"""
Bronze ingestion runner.

Fetches raw load data from ENTSO-E and persists it as a Delta table
(append-only, schema-preserved). Idempotent: re-runs overwrite overlapping
dates via a merge by (timestamp_utc, country, kind).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from delta.tables import DeltaTable
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.ingestion.entsoe_client import EntsoeClient
from src.utils.config import get_env, load_yaml_config
from src.utils.logging import configure_logging, get_logger
from src.utils.spark import get_spark

log = get_logger(__name__)

BRONZE_SCHEMA = StructType(
    [
        StructField("timestamp_utc", TimestampType(), nullable=False),
        StructField("country", StringType(), nullable=False),
        StructField("load_mw", DoubleType(), nullable=True),
        StructField("kind", StringType(), nullable=False),
        StructField("ingested_at", TimestampType(), nullable=False),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest ENTSO-E load → Bronze Delta table")
    parser.add_argument("--start", type=str, help="YYYY-MM-DD (UTC). Defaults to 7 days ago.")
    parser.add_argument("--end", type=str, help="YYYY-MM-DD (UTC). Defaults to today.")
    parser.add_argument("--country", type=str, default=None, help="Override target country.")
    return parser.parse_args()


def run(start: str | None = None, end: str | None = None, country: str | None = None) -> None:
    env = get_env()
    cfg = load_yaml_config()

    country = country or cfg["ingestion"]["target_country"]
    now_utc = datetime.now(tz=timezone.utc)
    start_dt = datetime.fromisoformat(start) if start else (now_utc - timedelta(days=7))
    end_dt = datetime.fromisoformat(end) if end else now_utc

    log.info("bronze.start", country=country, start=str(start_dt), end=str(end_dt))

    client = EntsoeClient(api_token=env.entsoe_api_token, country_code=country)
    pdf = client.fetch_load(start=start_dt, end=end_dt, kind="actual")
    pdf["ingested_at"] = now_utc

    if pdf.empty:
        log.warning("bronze.empty_response")
        return

    spark = get_spark("bronze-ingestion")
    sdf = spark.createDataFrame(pdf, schema=BRONZE_SCHEMA)

    bronze_path = str(Path(env.bronze_path).resolve() / "load")
    Path(bronze_path).parent.mkdir(parents=True, exist_ok=True)

    if DeltaTable.isDeltaTable(spark, bronze_path):
        log.info("bronze.merge", path=bronze_path)
        target = DeltaTable.forPath(spark, bronze_path)
        (
            target.alias("t")
            .merge(
                sdf.alias("s"),
                "t.timestamp_utc = s.timestamp_utc AND t.country = s.country AND t.kind = s.kind",
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        log.info("bronze.create", path=bronze_path)
        (
            sdf.write.format("delta")
            .partitionBy("country")
            .mode("overwrite")
            .save(bronze_path)
        )

    rows = spark.read.format("delta").load(bronze_path).count()
    log.info("bronze.done", path=bronze_path, total_rows=rows)


if __name__ == "__main__":
    configure_logging(level=get_env().log_level)
    args = parse_args()
    run(start=args.start, end=args.end, country=args.country)
