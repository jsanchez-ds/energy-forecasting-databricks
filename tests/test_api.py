"""Smoke tests for the FastAPI app (without requiring a loaded model)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.serving import api as api_module


def test_health_endpoint() -> None:
    # Bypass lifespan (no MLflow available in unit tests)
    with TestClient(api_module.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_model_info_endpoint() -> None:
    with TestClient(api_module.app) as client:
        r = client.get("/model-info")
        assert r.status_code == 200
        body = r.json()
        assert "forecast_model" in body
        assert "anomaly_model" in body


def test_predict_without_model_returns_503() -> None:
    # Entering the TestClient fires the lifespan, which will try to load
    # a real model from MLflow if one is registered. We override STATE
    # *after* lifespan has run so the endpoint sees no model.
    with TestClient(api_module.app) as client:
        api_module.STATE["forecast_model"] = None
        api_module.STATE["forecast_model_name"] = "energy-demand-forecaster"
        payload = {
            "rows": [
                {
                    "timestamp_utc": "2024-06-01T00:00:00Z",
                    "hour": 0, "day_of_week": 5, "day_of_month": 1,
                    "month": 6, "quarter": 2, "week_of_year": 22,
                    "is_weekend": 1, "is_holiday_es": 0,
                    "hour_sin": 0.0, "hour_cos": 1.0,
                    "dow_sin": -0.43, "dow_cos": -0.9,
                    "month_sin": 1.0, "month_cos": 0.0,
                    "load_mw_clean_lag_1": 25000.0, "load_mw_clean_lag_24": 24000.0,
                    "load_mw_clean_lag_48": 24500.0, "load_mw_clean_lag_168": 26000.0,
                    "load_mw_clean_roll_mean_24": 25000.0, "load_mw_clean_roll_std_24": 800.0,
                    "load_mw_clean_roll_mean_168": 25500.0, "load_mw_clean_roll_std_168": 1200.0,
                }
            ]
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 503
