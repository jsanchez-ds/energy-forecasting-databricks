"""
Gold feature runner.

Reads Silver, computes ML-ready features per country, and writes a Gold Delta
table ready for training. Uses pandas for feature math (single-country volumes
are small) and writes back via Spark.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pyspark.sql.types import StructType

from src.features.calendar import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
)
from src.utils.config import get_env, load_yaml_config
from src.utils.logging import configure_logging, get_logger
from src.utils.spark import get_spark

log = get_logger(__name__)


def build_features(pdf: pd.DataFrame, cfg: dict, country: str = "ES") -> pd.DataFrame:
    """Build the Gold feature set from a cleaned Silver pandas DataFrame."""
    pdf = pdf.sort_values("timestamp_utc").reset_index(drop=True)
    pdf = add_calendar_features(pdf, ts_col="timestamp_utc", country=country)
    pdf = add_lag_features(pdf, target="load_mw_clean", lags=cfg["features"]["lags"])
    pdf = add_rolling_features(
        pdf, target="load_mw_clean", windows=cfg["features"]["rolling_windows"]
    )
    pdf = pdf.dropna(subset=[c for c in pdf.columns if c.startswith("load_mw_clean_lag_")])
    return pdf


def run() -> None:
    env = get_env()
    cfg = load_yaml_config()
    spark = get_spark("gold-features")

    silver_path = str(Path(env.silver_path).resolve() / "load")
    gold_path = str(Path(env.gold_path).resolve() / "load_features")
    Path(gold_path).parent.mkdir(parents=True, exist_ok=True)

    log.info("gold.read_silver", path=silver_path)
    silver_pdf = (
        spark.read.format("delta")
        .load(silver_path)
        .filter("kind = 'actual'")
        .toPandas()
    )

    frames = []
    for country, group in silver_pdf.groupby("country"):
        feats = build_features(group, cfg, country=str(country))
        frames.append(feats)
    gold_pdf = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    log.info("gold.built", rows=len(gold_pdf), countries=gold_pdf["country"].nunique() if not gold_pdf.empty else 0)

    if gold_pdf.empty:
        log.warning("gold.empty")
        return

    gold_sdf = spark.createDataFrame(gold_pdf)
    (
        gold_sdf.write.format("delta")
        .partitionBy("country")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(gold_path)
    )

    log.info("gold.done", path=gold_path, rows=gold_pdf.shape[0], cols=gold_pdf.shape[1])


if __name__ == "__main__":
    configure_logging(level=get_env().log_level)
    run()
