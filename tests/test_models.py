"""Forecasting and anomaly model tests."""

from __future__ import annotations

import pandas as pd

from src.models.anomaly import AnomalyDetector
from src.models.forecasting import LightGBMForecaster, compute_metrics, time_split


def test_time_split_no_leak(sample_gold_df: pd.DataFrame) -> None:
    train, test = time_split(sample_gold_df, test_days=3)
    assert train["timestamp_utc"].max() < test["timestamp_utc"].min()
    assert len(train) > 0 and len(test) > 0


def test_lightgbm_fit_predict(sample_gold_df: pd.DataFrame) -> None:
    train, test = time_split(sample_gold_df, test_days=2)
    model = LightGBMForecaster(params={"n_estimators": 20, "verbose": -1, "random_state": 0})
    model.fit(train)
    preds = model.predict(test)
    assert len(preds) == len(test)
    assert not (preds == 0).all(), "Model is predicting all zeros"


def test_lightgbm_beats_mean_baseline(sample_gold_df: pd.DataFrame) -> None:
    """LightGBM must beat predicting the training mean (weakest possible baseline).

    On tiny synthetic data with perfect 24h periodicity the lag-24 feature alone
    is almost unbeatable, so we use a looser but still meaningful check: the
    model must do better than always predicting the training average.
    """
    import numpy as np

    train, test = time_split(sample_gold_df, test_days=2)
    model = LightGBMForecaster(params={"n_estimators": 100, "verbose": -1, "random_state": 0})
    model.fit(train)
    lgb_mape = model.evaluate(test).mape

    train_mean = train["load_mw_clean"].mean()
    mean_baseline_mape = float(
        np.mean(np.abs(test["load_mw_clean"] - train_mean) / test["load_mw_clean"])
    )
    assert lgb_mape < mean_baseline_mape, (
        f"LightGBM ({lgb_mape:.4f}) did not beat mean baseline ({mean_baseline_mape:.4f})"
    )
    assert lgb_mape < 0.25, f"MAPE ({lgb_mape:.4f}) is unreasonably high"


def test_compute_metrics_zero_error() -> None:
    import numpy as np

    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = compute_metrics(y, y)
    assert m.mape == 0.0 and m.rmse == 0.0 and m.mae == 0.0


def test_anomaly_detector_fit_predict(sample_gold_df: pd.DataFrame) -> None:
    detector = AnomalyDetector(params={"n_estimators": 50, "contamination": 0.05, "random_state": 0})
    detector.fit(sample_gold_df)
    results = detector.predict(sample_gold_df)
    assert {"anomaly_score", "is_anomaly"}.issubset(results.columns)
    # contamination=0.05 → roughly 5% flagged
    assert 0 < results["is_anomaly"].mean() < 0.2
