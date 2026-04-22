"""Spark session factory with Delta Lake configured — works local and on Databricks."""

from __future__ import annotations

from functools import lru_cache

from pyspark.sql import SparkSession


@lru_cache(maxsize=1)
def get_spark(app_name: str = "energy-forecasting") -> SparkSession:
    """
    Return a Spark session with Delta Lake enabled.

    On Databricks this just returns the existing session. Locally, it configures
    Delta + a warehouse directory so you can do `.format("delta")` without pain.
    """
    builder = (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
    )

    try:
        from delta import configure_spark_with_delta_pip

        builder = configure_spark_with_delta_pip(builder)
    except ImportError:
        pass

    return builder.getOrCreate()
