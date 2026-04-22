"""Feature engineering tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.calendar import (
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
)


def test_calendar_features_shape(sample_load_df: pd.DataFrame) -> None:
    out = add_calendar_features(sample_load_df)
    expected_cols = {
        "hour", "day_of_week", "day_of_month", "month", "quarter",
        "week_of_year", "is_weekend", "is_holiday_es",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    }
    assert expected_cols.issubset(out.columns)


def test_cyclic_encodings_range(sample_load_df: pd.DataFrame) -> None:
    out = add_calendar_features(sample_load_df)
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
        assert out[col].between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"


def test_is_weekend_matches_dow(sample_load_df: pd.DataFrame) -> None:
    out = add_calendar_features(sample_load_df)
    assert ((out["day_of_week"] >= 5) == out["is_weekend"].astype(bool)).all()


def test_lag_features_shift(sample_load_df: pd.DataFrame) -> None:
    out = add_lag_features(sample_load_df, lags=[1, 24])
    assert "load_mw_clean_lag_1" in out.columns
    assert "load_mw_clean_lag_24" in out.columns
    # First 24 rows of lag_24 must be NaN
    assert out["load_mw_clean_lag_24"].iloc[:24].isna().all()


def test_rolling_features_leak_free(sample_load_df: pd.DataFrame) -> None:
    """Rolling stats must be computed on shifted data — no target leakage."""
    out = add_rolling_features(sample_load_df, windows=[24])
    # rolling_mean[t] should NOT equal load_mw_clean[t] (would mean leakage)
    equal_count = (out["load_mw_clean_roll_mean_24"] == out["load_mw_clean"]).sum()
    assert equal_count < len(out), "Rolling mean equals target — leakage detected"
