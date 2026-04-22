"""
Drift monitoring with Evidently.

Compares a reference window (stable period) against a current window (recent
production data) and emits:
  - HTML report (visual)
  - JSON summary (for alerting)
  - Boolean "drift_detected" flag based on config threshold
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

from src.utils.config import get_env, load_yaml_config
from src.utils.logging import configure_logging, get_logger
from src.utils.spark import get_spark

log = get_logger(__name__)


@dataclass
class DriftResult:
    drift_share: float
    drift_detected: bool
    report_html_path: str
    report_json_path: str


def run_drift_check(output_dir: str | Path = "./data/drift_reports") -> DriftResult:
    env = get_env()
    cfg = load_yaml_config()
    spark = get_spark("drift-check")

    gold_path = str(Path(env.gold_path).resolve() / "load_features")
    pdf = spark.read.format("delta").load(gold_path).toPandas()
    pdf["timestamp_utc"] = pd.to_datetime(pdf["timestamp_utc"], utc=True)
    pdf = pdf.sort_values("timestamp_utc")

    ref_days = cfg["monitoring"]["reference_window_days"]
    cur_days = cfg["monitoring"]["current_window_days"]
    threshold = cfg["monitoring"]["drift_threshold"]

    max_ts = pdf["timestamp_utc"].max()
    cur_start = max_ts - timedelta(days=cur_days)
    ref_end = cur_start
    ref_start = ref_end - timedelta(days=ref_days)

    reference = pdf[(pdf["timestamp_utc"] >= ref_start) & (pdf["timestamp_utc"] < ref_end)].copy()
    current = pdf[pdf["timestamp_utc"] >= cur_start].copy()

    log.info(
        "drift.windows",
        ref_rows=len(reference),
        cur_rows=len(current),
        ref=f"{ref_start}..{ref_end}",
        cur=f"{cur_start}..{max_ts}",
    )

    if len(reference) < 100 or len(current) < 100:
        log.warning("drift.insufficient_rows")
        return DriftResult(0.0, False, "", "")

    feature_cols = [
        "load_mw_clean", "hour", "day_of_week", "is_weekend",
        "load_mw_clean_lag_24", "load_mw_clean_lag_168",
        "load_mw_clean_roll_mean_24", "load_mw_clean_roll_std_24",
    ]
    mapping = ColumnMapping(numerical_features=feature_cols, target="load_mw_clean")

    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(
        reference_data=reference[feature_cols],
        current_data=current[feature_cols],
        column_mapping=mapping,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts_tag = max_ts.strftime("%Y%m%d_%H%M")
    html_path = out / f"drift_{ts_tag}.html"
    json_path = out / f"drift_{ts_tag}.json"
    report.save_html(str(html_path))
    report.save_json(str(json_path))

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    drift_share = 0.0
    for metric in payload.get("metrics", []):
        res = metric.get("result", {})
        if "share_of_drifted_columns" in res:
            drift_share = float(res["share_of_drifted_columns"])
            break

    drift_detected = drift_share >= threshold
    log.info(
        "drift.done",
        drift_share=drift_share,
        threshold=threshold,
        detected=drift_detected,
        html=str(html_path),
    )
    return DriftResult(
        drift_share=drift_share,
        drift_detected=drift_detected,
        report_html_path=str(html_path),
        report_json_path=str(json_path),
    )


if __name__ == "__main__":
    configure_logging(level=get_env().log_level)
    result = run_drift_check()
    print(json.dumps(result.__dict__, indent=2))
