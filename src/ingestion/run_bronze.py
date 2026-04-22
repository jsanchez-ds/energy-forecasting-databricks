"""
Bronze ingestion runner.

Fetches raw load data from the configured source (ENTSO-E or EIA) and persists
it as a Delta table (append-only, schema-preserved). Idempotent: re-runs
overwrite overlapping dates via a merge by (timestamp_utc, country, kind).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from delta.tables import DeltaTable
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

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
    parser = argparse.ArgumentParser(description="Ingest load data → Bronze Delta table")
    parser.add_argument("--start", type=str, help="YYYY-MM-DD (UTC). Defaults to 7 days ago.")
    parser.add_argument("--end", type=str, help="YYYY-MM-DD (UTC). Defaults to today.")
    parser.add_argument(
        "--source",
        type=str,
        choices=["entsoe", "eia"],
        default=None,
        help="Override DATA_SOURCE from .env",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="ENTSO-E country (ES, DE_LU, ...) or EIA region (CAL, ERCO, PJM, ...)",
    )
    return parser.parse_args()


def _fetch(
    source: str,
    region: str,
    start_dt: datetime,
    end_dt: datetime,
    env,
):
    """Dispatch to the correct client based on source."""
    if source == "entsoe":
        from src.ingestion.entsoe_client import EntsoeClient

        client = EntsoeClient(api_token=env.entsoe_api_token, country_code=region)
        return client.fetch_load(start=start_dt, end=end_dt, kind="actual")
    elif source == "eia":
        from src.ingestion.eia_client import EiaClient

        with EiaClient(api_key=env.eia_api_key, region=region) as client:
            return client.fetch_load(start=start_dt, end=end_dt, kind="actual")
    else:
        raise ValueError(f"Unknown source: {source!r}")


def run(
    start: str | None = None,
    end: str | None = None,
    source: str | None = None,
    region: str | None = None,
) -> None:
    env = get_env()
    cfg = load_yaml_config()

    source = source or env.data_source
    if source == "entsoe":
        region = region or env.target_country
    else:
        region = region or env.target_region

    now_utc = datetime.now(tz=timezone.utc)
    start_dt = datetime.fromisoformat(start) if start else (now_utc - timedelta(days=7))
    end_dt = datetime.fromisoformat(end) if end else now_utc

    log.info("bronze.start", source=source, region=region, start=str(start_dt), end=str(end_dt))

    pdf = _fetch(source, region, start_dt, end_dt, env)
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
    run(start=args.start, end=args.end, source=args.source, region=args.region)
