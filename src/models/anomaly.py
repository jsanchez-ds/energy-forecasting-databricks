"""Anomaly detection — Isolation Forest over residuals and raw features."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


ANOMALY_FEATURES = [
    "load_mw_clean",
    "hour", "day_of_week", "is_weekend",
    "load_mw_clean_lag_24", "load_mw_clean_lag_168",
    "load_mw_clean_roll_mean_24", "load_mw_clean_roll_std_24",
]


@dataclass
class AnomalyDetector:
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "contamination": 0.01,
            "n_estimators": 200,
            "random_state": 42,
            "n_jobs": -1,
        }
    )
    model: IsolationForest | None = None

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        X = df[ANOMALY_FEATURES].dropna()
        self.model = IsolationForest(**self.params)
        self.model.fit(X)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        X = df[ANOMALY_FEATURES].copy()
        scores = self.model.decision_function(X.fillna(X.mean()))
        preds = self.model.predict(X.fillna(X.mean()))
        out = df[["timestamp_utc", "country", "load_mw_clean"]].copy()
        out["anomaly_score"] = scores
        out["is_anomaly"] = (preds == -1).astype(int)
        return out

    def summary(self, df: pd.DataFrame) -> dict[str, float]:
        results = self.predict(df)
        return {
            "n_anomalies": int(results["is_anomaly"].sum()),
            "anomaly_rate": float(results["is_anomaly"].mean()),
            "score_min": float(results["anomaly_score"].min()),
            "score_max": float(results["anomaly_score"].max()),
        }
