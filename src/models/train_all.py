"""
End-to-end training script.

Reads Gold table, trains forecasting (LightGBM, Prophet, LSTM) + anomaly
detection (Isolation Forest), logs everything to MLflow, and registers the
best forecaster + anomaly detector in the MLflow Model Registry with a
"@staging" alias.

On the best LightGBM run, also logs SHAP summary/bar plots as artifacts so
stakeholders can see which features drive predictions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.pytorch
import mlflow.sklearn
import pandas as pd
import shap

from src.models.anomaly import AnomalyDetector
from src.models.forecasting import (
    FEATURE_COLS,
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


# ── SHAP ─────────────────────────────────────────────────────────────────────
def log_shap_for_lightgbm(model, X_sample: pd.DataFrame, out_dir: Path) -> None:
    """Compute SHAP values for a LightGBM model and log plots as MLflow artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Bar (mean |SHAP|)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    bar_path = out_dir / "shap_bar.png"
    plt.tight_layout()
    plt.savefig(bar_path, dpi=140, bbox_inches="tight")
    plt.close()

    # Beeswarm (distribution of impact per feature)
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, show=False)
    bees_path = out_dir / "shap_beeswarm.png"
    plt.tight_layout()
    plt.savefig(bees_path, dpi=140, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(str(bar_path), artifact_path="shap")
    mlflow.log_artifact(str(bees_path), artifact_path="shap")

    # Also log the raw values as CSV for further analysis
    df_shap = pd.DataFrame(shap_values, columns=X_sample.columns)
    csv_path = out_dir / "shap_values.csv"
    df_shap.to_csv(csv_path, index=False)
    mlflow.log_artifact(str(csv_path), artifact_path="shap")


# ── Forecasters ─────────────────────────────────────────────────────────────
def train_forecasting(pdf: pd.DataFrame, cfg: dict) -> tuple[str, float, dict[str, float]]:
    """Trains LightGBM + Prophet + LSTM, logs each. Returns (best_run_id, best_mape, all_mapes)."""
    test_days = cfg["evaluation"]["test_size_days"]
    train_df, test_df = time_split(pdf, test_days=test_days)
    log.info("forecast.split", train_rows=len(train_df), test_rows=len(test_df))

    best_mape = float("inf")
    best_run_id: str | None = None
    all_mapes: dict[str, float] = {}

    # ── LightGBM ────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="lightgbm-forecaster") as run:
        lgb_params = next(
            m for m in cfg["models"]["forecasting"] if m["name"] == "lightgbm"
        )["params"]
        model = LightGBMForecaster(
            params={
                **lgb_params,
                "objective": "regression",
                "metric": "mape",
                "verbose": -1,
                "random_state": 42,
            }
        )
        model.fit(train_df)
        metrics = model.evaluate(test_df)

        mlflow.log_params({f"lgb_{k}": v for k, v in lgb_params.items()})
        mlflow.log_metrics(metrics.to_dict())
        mlflow.log_param("model_family", "lightgbm")
        mlflow.log_param("test_size_days", test_days)

        importances = model.feature_importance()
        mlflow.log_text(importances.to_csv(index=False), "feature_importances.csv")
        mlflow.lightgbm.log_model(model.model, artifact_path="model")

        # SHAP explainability (use a 2k sample so the TreeExplainer stays fast)
        try:
            sample = test_df[FEATURE_COLS].sample(
                n=min(2000, len(test_df)), random_state=42
            )
            log_shap_for_lightgbm(model.model, sample, Path("./artifacts/shap"))
            log.info("forecast.lightgbm.shap.logged")
        except Exception as exc:
            log.warning("forecast.lightgbm.shap.failed", error=str(exc))

        log.info("forecast.lightgbm.done", **metrics.to_dict())
        all_mapes["lightgbm"] = metrics.mape
        if metrics.mape < best_mape:
            best_mape = metrics.mape
            best_run_id = run.info.run_id

    # ── Prophet ─────────────────────────────────────────────────────────────
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
            all_mapes["prophet"] = metrics.mape
            if metrics.mape < best_mape:
                best_mape = metrics.mape
                best_run_id = run.info.run_id
    except Exception as exc:
        log.warning("forecast.prophet.failed", error=str(exc))

    # ── LSTM ────────────────────────────────────────────────────────────────
    try:
        from src.models.lstm_forecaster import LSTMConfig, LSTMForecaster

        with mlflow.start_run(run_name="lstm-forecaster") as run:
            lstm_cfg = LSTMConfig(
                seq_len=168, hidden_size=64, num_layers=2, dropout=0.2,
                lr=1e-3, batch_size=128, epochs=30, patience=5, device="cpu",
            )
            model = LSTMForecaster(cfg=lstm_cfg)
            model.fit(train_df)
            metrics = model.evaluate(test_df)

            mlflow.log_params({
                "lstm_seq_len": lstm_cfg.seq_len,
                "lstm_hidden_size": lstm_cfg.hidden_size,
                "lstm_num_layers": lstm_cfg.num_layers,
                "lstm_dropout": lstm_cfg.dropout,
                "lstm_lr": lstm_cfg.lr,
                "lstm_batch_size": lstm_cfg.batch_size,
                "lstm_max_epochs": lstm_cfg.epochs,
            })
            mlflow.log_metrics(metrics.to_dict())
            mlflow.log_param("model_family", "lstm")
            mlflow.log_metric("lstm_final_val_loss", model.history["val"][-1])
            mlflow.log_metric("lstm_epochs_run", len(model.history["train"]))

            # Plot training curve
            try:
                plt.figure(figsize=(8, 4))
                plt.plot(model.history["train"], label="train")
                plt.plot(model.history["val"], label="val")
                plt.xlabel("epoch"); plt.ylabel("MSE loss (scaled)"); plt.legend()
                Path("./artifacts/lstm").mkdir(parents=True, exist_ok=True)
                loss_path = Path("./artifacts/lstm/training_curve.png")
                plt.tight_layout(); plt.savefig(loss_path, dpi=140); plt.close()
                mlflow.log_artifact(str(loss_path), artifact_path="plots")
            except Exception as exc:
                log.warning("forecast.lstm.plot.failed", error=str(exc))

            mlflow.pytorch.log_model(model.model, artifact_path="model")

            log.info("forecast.lstm.done", **metrics.to_dict())
            all_mapes["lstm"] = metrics.mape
            if metrics.mape < best_mape:
                best_mape = metrics.mape
                best_run_id = run.info.run_id
    except ImportError as exc:
        log.warning("forecast.lstm.skipped", error=f"PyTorch not installed: {exc}")
    except Exception as exc:
        log.warning("forecast.lstm.failed", error=str(exc))

    assert best_run_id is not None, "No forecasting model trained successfully"
    return best_run_id, best_mape, all_mapes


# ── Anomaly ─────────────────────────────────────────────────────────────────
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

    fc_run_id, fc_mape, all_mapes = train_forecasting(pdf, cfg)
    register(fc_run_id, cfg["mlflow"]["registry"]["forecast_model_name"])

    an_run_id = train_anomaly(pdf, cfg)
    register(an_run_id, cfg["mlflow"]["registry"]["anomaly_model_name"])

    log.info(
        "train.done",
        best_forecast_run=fc_run_id,
        best_mape=fc_mape,
        all_mapes=all_mapes,
        anomaly_run=an_run_id,
    )


if __name__ == "__main__":
    configure_logging(level=get_env().log_level)
    main()
