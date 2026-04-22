"""
Silver transformation runner.

Cleans Bronze:
  - Drops duplicates by (timestamp_utc, country, kind)
  - Fills gaps via forward-fill within a 3-hour window
  - Flags outliers (IQR-based) without removing them
  - Adds a quality flag column
"""

from __future__ import annotations

from pathlib import Path

from delta.tables import DeltaTable
from pyspark.sql import Window
from pyspark.sql import functions as F

from src.utils.config import get_env
from src.utils.logging import configure_logging, get_logger
from src.utils.spark import get_spark

log = get_logger(__name__)


def run() -> None:
    env = get_env()
    spark = get_spark("silver-transformation")

    bronze_path = str(Path(env.bronze_path).resolve() / "load")
    silver_path = str(Path(env.silver_path).resolve() / "load")
    Path(silver_path).parent.mkdir(parents=True, exist_ok=True)

    log.info("silver.read_bronze", path=bronze_path)
    bronze = spark.read.format("delta").load(bronze_path)

    # Deduplicate keeping latest ingestion
    w_latest = Window.partitionBy("timestamp_utc", "country", "kind").orderBy(
        F.col("ingested_at").desc()
    )
    dedup = (
        bronze.withColumn("_rn", F.row_number().over(w_latest))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    # Outlier detection (country-level IQR)
    q1, q3 = dedup.approxQuantile("load_mw", [0.25, 0.75], 0.001)
    iqr = q3 - q1
    lower, upper = q1 - 3 * iqr, q3 + 3 * iqr

    enriched = dedup.withColumn(
        "is_outlier",
        (F.col("load_mw") < F.lit(lower)) | (F.col("load_mw") > F.lit(upper)),
    ).withColumn("quality_flag", F.when(F.col("load_mw").isNull(), "missing")
                                  .when(F.col("is_outlier"), "outlier")
                                  .otherwise("ok"))

    # Forward-fill short gaps (up to 3 hours)
    w_ffill = (
        Window.partitionBy("country")
        .orderBy("timestamp_utc")
        .rowsBetween(-3, 0)
    )
    silver = enriched.withColumn(
        "load_mw_clean",
        F.coalesce(F.col("load_mw"), F.last("load_mw", ignorenulls=True).over(w_ffill)),
    )

    silver = silver.select(
        "timestamp_utc",
        "country",
        "kind",
        "load_mw",
        "load_mw_clean",
        "is_outlier",
        "quality_flag",
        "ingested_at",
    )

    if DeltaTable.isDeltaTable(spark, silver_path):
        log.info("silver.merge", path=silver_path)
        target = DeltaTable.forPath(spark, silver_path)
        (
            target.alias("t")
            .merge(
                silver.alias("s"),
                "t.timestamp_utc = s.timestamp_utc AND t.country = s.country AND t.kind = s.kind",
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        log.info("silver.create", path=silver_path)
        (
            silver.write.format("delta")
            .partitionBy("country")
            .mode("overwrite")
            .save(silver_path)
        )

    n = spark.read.format("delta").load(silver_path).count()
    log.info("silver.done", rows=n, lower=lower, upper=upper)


if __name__ == "__main__":
    configure_logging(level=get_env().log_level)
    run()
