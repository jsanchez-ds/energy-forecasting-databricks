"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_load_df() -> pd.DataFrame:
    """Two weeks of synthetic hourly load with realistic diurnal + weekly seasonality."""
    rng = np.random.default_rng(42)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    timestamps = pd.date_range(start, periods=24 * 14, freq="h", tz="UTC")
    hours = timestamps.hour
    dows = timestamps.dayofweek

    base = 25_000
    diurnal = 8_000 * np.sin(2 * np.pi * (hours - 6) / 24)
    weekly = -3_000 * (dows >= 5).astype(int)
    noise = rng.normal(0, 500, size=len(timestamps))
    load = base + diurnal + weekly + noise

    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "country": "ES",
            "kind": "actual",
            "load_mw": load,
            "load_mw_clean": load,
            "is_outlier": False,
            "quality_flag": "ok",
            "ingested_at": datetime.now(tz=timezone.utc),
        }
    )


@pytest.fixture
def sample_gold_df(sample_load_df: pd.DataFrame) -> pd.DataFrame:
    """Gold-shaped features for ML tests."""
    from src.features.calendar import (
        add_calendar_features,
        add_lag_features,
        add_rolling_features,
    )

    df = sample_load_df.sort_values("timestamp_utc").reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = df.dropna(
        subset=[c for c in df.columns if c.startswith("load_mw_clean_lag_")]
    ).reset_index(drop=True)
    return df
