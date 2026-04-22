"""
End-to-end training script.

Reads Gold table, trains forecasting + anomaly models, logs everything to
MLflow, and registers the best forecaster + anomaly detector in the MLflow
Model Registry with a "Staging" alias.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import pandas as pd

from src.models.anomaly import AnomalyDetector
from src.models.forecasting import (
    LightGBMForecaster,
    ProphetForecaster,
    time_split,
)
from src.utils.config import get_env, load_yaml_config
from src.utils.logging import configure_logging, get_logger
from src.utils.spark import get_spark

log = get_logger(__name__)


def load_gold() -> pd.DataFrame:
    env = get_env()
    spark = get_spark("train-all")
    gold_path = str(Path(env.gold_path).resolve() / "load_features")
    pdf = spark.read.format("delta").load(gold_path).toPandas()
    pdf["timestamp_utc"] = pd.to_datetime(pdf["timestamp_utc"], utc=True)
    return pdf


def train_forecasting(pdf: pd.DataFrame, cfg: dict) -> tuple[str, float]:
    """Trains LightGBM + Prophet, logs both, returns the better run_id + MAPE."""
    test_days = cfg["evaluation"]["test_size_days"]
    train_df, test_df = time_split(pdf, test_days=test_days)
    log.info("forecast.split", train_rows=len(train_df), test_rows=len(test_df))

    best_mape = float("inf")
    best_run_id: str | None = None

    # LightGBM
    with mlflow.start_run(run_name="lightgbm-forecaster") as run:
        lgb_params = next(m for m in cfg["models"]["forecasting"] if m["name"] == "lightgbm")["params"]
        model = LightGBMForecaster(params={**lgb_params, "objective": "regression",
                                            "metric": "mape", "verbose": -1, "random_state": 42})
        model.fit(train_df)
        metrics = model.evaluate(test_df)

        mlflow.log_params({f"lgb_{k}": v for k, v in lgb_params.items()})
        mlflow.log_metrics(metrics.to_dict())
        mlflow.log_param("model_family", "lightgbm")
        mlflow.log_param("test_size_days", test_days)

        importances = model.feature_importance()
        mlflow.log_text(importances.to_csv(index=False), "feature_importances.csv")
        mlflow.lightgbm.log_model(model.model, artifact_path="model")

        log.info("forecast.lightgbm.done", **metrics.to_dict())
        if metrics.mape < best_mape:
            best_mape = metrics.mape
            best_run_id = run.info.run_id

    # Prophet
    try:
        with mlflow.start_run(run_name="prophet-forecaster") as run:
            prop_params = next(
                m for m in cfg["models"]["forecasting"] if m["name"] == "prophet"
            )["params"]
            model = ProphetForecaster(params=prop_params)
            model.fit(train_df)
            metrics = model.evaluate(test_df)

            mlflow.log_params({f"prophet_{k}": v for k, v in prop_params.items()})
            mlflow.log_metrics(metrics.to_dict())
            mlflow.log_param("model_family", "prophet")

            log.info("forecast.prophet.done", **metrics.to_dict())
            if metrics.mape < best_mape:
                best_mape = metrics.mape
                best_run_id = run.info.run_id
    except Exception as exc:  # prophet can be finicky on Windows
        log.warning("forecast.prophet.failed", error=str(exc))

    assert best_run_id is not None
    return best_run_id, best_mape


def train_anomaly(pdf: pd.DataFrame, cfg: dict) -> str:
    params = next(m for m in cfg["models"]["anomaly"] if m["name"] == "isolation_forest")["params"]
    with mlflow.start_run(run_name="isolation-forest-anomaly") as run:
        detector = AnomalyDetector(params={**params, "n_jobs": -1})
        detector.fit(pdf)
        summary = detector.summary(pdf)

        mlflow.log_params({f"if_{k}": v for k, v in params.items()})
        mlflow.log_metrics(summary)
        mlflow.sklearn.log_model(detector.model, artifact_path="model")

        log.info("anomaly.done", **summary)
        return run.info.run_id


def register(run_id: str, model_name: str) -> None:
    client = mlflow.tracking.MlflowClient()
    result = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=model_name,
    )
    client.set_registered_model_alias(
        name=model_name, alias="staging", version=result.version
    )
    log.info("registry.updated", model=model_name, version=result.version, alias="staging")


def main() -> None:
    env = get_env()
    cfg = load_yaml_config()

    mlflow.set_tracking_uri(env.mlflow_tracking_uri)
    mlflow.set_experiment(env.mlflow_experiment_name)

    log.info("train.load_gold")
    pdf = load_gold()
    log.info("train.loaded", rows=len(pdf), countries=pdf["country"].nunique())

    if len(pdf) < 2000:
        log.warning("train.insufficient_data", rows=len(pdf))

    fc_run_id, fc_mape = train_forecasting(pdf, cfg)
    register(fc_run_id, cfg["mlflow"]["registry"]["forecast_model_name"])

    an_run_id = train_anomaly(pdf, cfg)
    register(an_run_id, cfg["mlflow"]["registry"]["anomaly_model_name"])

    log.info("train.done", best_forecast_run=fc_run_id, best_mape=fc_mape, anomaly_run=an_run_id)


if __name__ == "__main__":
    configure_logging(level=get_env().log_level)
    main()
