"""
FastAPI serving layer.

Loads the @staging model from the MLflow registry at startup and exposes:
  - POST /predict     → 24h-ahead demand forecast
  - POST /anomaly     → anomaly score + flag for a batch of observations
  - GET  /health      → liveness
  - GET  /model-info  → current model version and metadata
  - GET  /metrics     → Prometheus metrics
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response

from src.utils.config import get_env, load_yaml_config
from src.utils.logging import configure_logging, get_logger

log = get_logger(__name__)

# ── Prometheus metrics ──────────────────────────────────────────────────────
PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Total predictions served", ["model", "status"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency", ["model"]
)

STATE: dict[str, Any] = {}


# ── Schemas ─────────────────────────────────────────────────────────────────
class ForecastRow(BaseModel):
    timestamp_utc: str
    hour: int
    day_of_week: int
    day_of_month: int
    month: int
    quarter: int
    week_of_year: int
    is_weekend: int
    is_holiday_es: int
    hour_sin: float
    hour_cos: float
    dow_sin: float
    dow_cos: float
    month_sin: float
    month_cos: float
    load_mw_clean_lag_1: float
    load_mw_clean_lag_24: float
    load_mw_clean_lag_48: float
    load_mw_clean_lag_168: float
    load_mw_clean_roll_mean_24: float
    load_mw_clean_roll_std_24: float
    load_mw_clean_roll_mean_168: float
    load_mw_clean_roll_std_168: float


class ForecastRequest(BaseModel):
    rows: list[ForecastRow] = Field(..., min_length=1, max_length=10_000)


class ForecastResponse(BaseModel):
    # Disable Pydantic's 'model_*' protected namespace since our domain uses
    # 'model_version' / 'model_name' in the ML sense, not the Pydantic sense.
    model_config = ConfigDict(protected_namespaces=())

    predictions: list[float]
    model_version: str
    model_name: str


class AnomalyRow(BaseModel):
    timestamp_utc: str
    country: str
    load_mw_clean: float
    hour: int
    day_of_week: int
    is_weekend: int
    load_mw_clean_lag_24: float
    load_mw_clean_lag_168: float
    load_mw_clean_roll_mean_24: float
    load_mw_clean_roll_std_24: float


class AnomalyRequest(BaseModel):
    rows: list[AnomalyRow] = Field(..., min_length=1, max_length=10_000)


class AnomalyResult(BaseModel):
    timestamp_utc: str
    country: str
    anomaly_score: float
    is_anomaly: int


# ── Lifespan: load models at startup ────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    env = get_env()
    cfg = load_yaml_config()
    mlflow.set_tracking_uri(env.mlflow_tracking_uri)

    fc_name = cfg["mlflow"]["registry"]["forecast_model_name"]
    an_name = cfg["mlflow"]["registry"]["anomaly_model_name"]

    try:
        STATE["forecast_model"] = mlflow.pyfunc.load_model(f"models:/{fc_name}@staging")
        STATE["forecast_model_name"] = fc_name
        log.info("api.forecast_model.loaded", name=fc_name)
    except Exception as exc:
        log.warning("api.forecast_model.load_failed", error=str(exc))
        STATE["forecast_model"] = None
        STATE["forecast_model_name"] = fc_name

    try:
        STATE["anomaly_model"] = mlflow.pyfunc.load_model(f"models:/{an_name}@staging")
        STATE["anomaly_model_name"] = an_name
        log.info("api.anomaly_model.loaded", name=an_name)
    except Exception as exc:
        log.warning("api.anomaly_model.load_failed", error=str(exc))
        STATE["anomaly_model"] = None
        STATE["anomaly_model_name"] = an_name

    yield
    STATE.clear()


app = FastAPI(
    title="Energy Forecasting API",
    version="0.1.0",
    description="Demand forecasting + anomaly detection for grid electricity.",
    lifespan=lifespan,
)


# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "forecast_model": STATE.get("forecast_model_name"),
        "forecast_loaded": STATE.get("forecast_model") is not None,
        "anomaly_model": STATE.get("anomaly_model_name"),
        "anomaly_loaded": STATE.get("anomaly_model") is not None,
    }


@app.post("/predict", response_model=ForecastResponse)
def predict(req: ForecastRequest) -> ForecastResponse:
    model = STATE.get("forecast_model")
    if model is None:
        PREDICTIONS_TOTAL.labels(model="forecast", status="unavailable").inc()
        raise HTTPException(status_code=503, detail="Forecast model not loaded")

    df = pd.DataFrame([r.model_dump() for r in req.rows]).drop(columns=["timestamp_utc"])
    with PREDICTION_LATENCY.labels(model="forecast").time():
        preds = model.predict(df)

    PREDICTIONS_TOTAL.labels(model="forecast", status="ok").inc(len(preds))
    return ForecastResponse(
        predictions=[float(p) for p in preds],
        model_version="staging",
        model_name=STATE["forecast_model_name"],
    )


@app.post("/anomaly", response_model=list[AnomalyResult])
def detect_anomaly(req: AnomalyRequest) -> list[AnomalyResult]:
    model = STATE.get("anomaly_model")
    if model is None:
        PREDICTIONS_TOTAL.labels(model="anomaly", status="unavailable").inc()
        raise HTTPException(status_code=503, detail="Anomaly model not loaded")

    df = pd.DataFrame([r.model_dump() for r in req.rows])
    features = df.drop(columns=["timestamp_utc", "country"])
    with PREDICTION_LATENCY.labels(model="anomaly").time():
        preds = model.predict(features)

    PREDICTIONS_TOTAL.labels(model="anomaly", status="ok").inc(len(df))
    return [
        AnomalyResult(
            timestamp_utc=row.timestamp_utc,
            country=row.country,
            anomaly_score=float(score),
            is_anomaly=int(score == -1),
        )
        for row, score in zip(req.rows, preds, strict=True)
    ]


@app.get("/metrics")
def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


configure_logging(level=get_env().log_level)
