"""Calendar features (cyclic encodings, holidays, business-day flags)."""

from __future__ import annotations

import numpy as np
import pandas as pd


SPAIN_HOLIDAYS_MMDD = {
    "01-01", "01-06", "05-01", "08-15", "10-12", "11-01", "12-06", "12-08", "12-25",
}


def add_calendar_features(df: pd.DataFrame, ts_col: str = "timestamp_utc") -> pd.DataFrame:
    """Append calendar/time features in-place-ish (returns a copy)."""
    out = df.copy()
    ts = pd.to_datetime(out[ts_col], utc=True)

    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["day_of_month"] = ts.dt.day
    out["month"] = ts.dt.month
    out["quarter"] = ts.dt.quarter
    out["year"] = ts.dt.year
    out["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    # Cyclic encodings — critical for forecasting models
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    mmdd = ts.dt.strftime("%m-%d")
    out["is_holiday_es"] = mmdd.isin(SPAIN_HOLIDAYS_MMDD).astype(int)
    return out


def add_lag_features(
    df: pd.DataFrame,
    target: str = "load_mw_clean",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Add lag features. Assumes df is sorted by time within country."""
    lags = lags or [1, 24, 48, 168]
    out = df.copy()
    for lag in lags:
        out[f"{target}_lag_{lag}"] = out[target].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    target: str = "load_mw_clean",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Rolling mean/std — computed over past values only (shift(1))."""
    windows = windows or [24, 168]
    out = df.copy()
    shifted = out[target].shift(1)
    for w in windows:
        out[f"{target}_roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        out[f"{target}_roll_std_{w}"] = shifted.rolling(w, min_periods=1).std()
    return out
