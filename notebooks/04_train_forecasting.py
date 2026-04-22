# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Train Forecasting + Anomaly Models
# MAGIC
# MAGIC Trains LightGBM + Prophet (forecasting) and Isolation Forest (anomaly) on the Gold table.
# MAGIC Logs to MLflow and registers the best model with a `@staging` alias.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.5.0 prophet==1.1.6 mlflow==2.17.0 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

dbutils.widgets.text("base_path", "dbfs:/FileStore/energy", label="Delta base path")
base_path = dbutils.widgets.get("base_path").rstrip("/")
GOLD_PATH = f"{base_path}/gold/load_features"
EXPERIMENT = "/Shared/energy-forecasting"
FC_MODEL = "energy-demand-forecaster"
AN_MODEL = "energy-anomaly-detector"

mlflow.set_experiment(EXPERIMENT)

FEATURE_COLS = [
    "hour","day_of_week","day_of_month","month","quarter","week_of_year",
    "is_weekend","is_holiday_es",
    "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
    "load_mw_clean_lag_1","load_mw_clean_lag_24","load_mw_clean_lag_48","load_mw_clean_lag_168",
    "load_mw_clean_roll_mean_24","load_mw_clean_roll_std_24",
    "load_mw_clean_roll_mean_168","load_mw_clean_roll_std_168",
]
TARGET = "load_mw_clean"

pdf = spark.read.format("delta").load(GOLD_PATH).toPandas()
pdf["timestamp_utc"] = pd.to_datetime(pdf["timestamp_utc"], utc=True)
pdf = pdf.sort_values("timestamp_utc").reset_index(drop=True)

# COMMAND ----------

# Time-aware split
cutoff = pdf["timestamp_utc"].max() - pd.Timedelta(days=30)
train = pdf[pdf["timestamp_utc"] <= cutoff]
test = pdf[pdf["timestamp_utc"] > cutoff]
print(f"Train: {len(train):,} · Test: {len(test):,}")

# COMMAND ----------

# LightGBM
with mlflow.start_run(run_name="lightgbm-forecaster") as run:
    params = dict(
        objective="regression", metric="mape",
        n_estimators=500, learning_rate=0.05, num_leaves=64,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1,
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(train[FEATURE_COLS], train[TARGET])
    y_pred = model.predict(test[FEATURE_COLS])
    mape = mean_absolute_percentage_error(test[TARGET], y_pred)
    rmse = float(np.sqrt(((test[TARGET] - y_pred) ** 2).mean()))
    mae = mean_absolute_error(test[TARGET], y_pred)

    mlflow.log_params({f"lgb_{k}": v for k, v in params.items()})
    mlflow.log_metrics({"mape": mape, "rmse": rmse, "mae": mae})
    mlflow.lightgbm.log_model(model, artifact_path="model")
    lgb_run_id = run.info.run_id
    print(f"LightGBM · MAPE {mape:.4f} · RMSE {rmse:,.1f} · MAE {mae:,.1f}")

# COMMAND ----------

# Register best forecaster
result = mlflow.register_model(f"runs:/{lgb_run_id}/model", name=FC_MODEL)
mlflow.tracking.MlflowClient().set_registered_model_alias(FC_MODEL, "staging", result.version)
print(f"Registered {FC_MODEL} v{result.version} @ staging")

# COMMAND ----------

# Isolation Forest anomaly detector
ANOM_FEATURES = [
    "load_mw_clean","hour","day_of_week","is_weekend",
    "load_mw_clean_lag_24","load_mw_clean_lag_168",
    "load_mw_clean_roll_mean_24","load_mw_clean_roll_std_24",
]

with mlflow.start_run(run_name="isolation-forest-anomaly") as run:
    detector = IsolationForest(contamination=0.01, n_estimators=200, random_state=42, n_jobs=-1)
    X = pdf[ANOM_FEATURES].fillna(pdf[ANOM_FEATURES].mean())
    detector.fit(X)
    preds = detector.predict(X)
    rate = float((preds == -1).mean())

    mlflow.log_params({"contamination": 0.01, "n_estimators": 200})
    mlflow.log_metric("anomaly_rate", rate)
    mlflow.sklearn.log_model(detector, artifact_path="model")
    an_run_id = run.info.run_id
    print(f"Anomaly rate {rate:.4f}")

result = mlflow.register_model(f"runs:/{an_run_id}/model", name=AN_MODEL)
mlflow.tracking.MlflowClient().set_registered_model_alias(AN_MODEL, "staging", result.version)
print(f"Registered {AN_MODEL} v{result.version} @ staging")
