"""
Forecasting models — LightGBM (gradient boosting, handles exogenous features)
and Prophet (classical seasonal decomposition). Both produce 24h-ahead forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


FEATURE_COLS = [
    "hour", "day_of_week", "day_of_month", "month", "quarter", "week_of_year",
    "is_weekend", "is_holiday_es",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "load_mw_clean_lag_1", "load_mw_clean_lag_24", "load_mw_clean_lag_48",
    "load_mw_clean_lag_168",
    "load_mw_clean_roll_mean_24", "load_mw_clean_roll_std_24",
    "load_mw_clean_roll_mean_168", "load_mw_clean_roll_std_168",
]
TARGET_COL = "load_mw_clean"


@dataclass
class ForecastMetrics:
    mape: float
    rmse: float
    mae: float

    def to_dict(self) -> dict[str, float]:
        return {"mape": self.mape, "rmse": self.rmse, "mae": self.mae}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ForecastMetrics:
    mape = float(mean_absolute_percentage_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return ForecastMetrics(mape=mape, rmse=rmse, mae=mae)


def time_split(
    df: pd.DataFrame, test_days: int = 30, ts_col: str = "timestamp_utc"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(ts_col)
    cutoff = df[ts_col].max() - pd.Timedelta(days=test_days)
    train = df[df[ts_col] <= cutoff].copy()
    test = df[df[ts_col] > cutoff].copy()
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LightGBMForecaster:
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "objective": "regression",
            "metric": "mape",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 64,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": 42,
            "verbose": -1,
        }
    )
    model: lgb.LGBMRegressor | None = None

    def fit(self, df_train: pd.DataFrame) -> "LightGBMForecaster":
        X = df_train[FEATURE_COLS]
        y = df_train[TARGET_COL]
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        return self.model.predict(df[FEATURE_COLS])

    def evaluate(self, df_test: pd.DataFrame) -> ForecastMetrics:
        y_pred = self.predict(df_test)
        return compute_metrics(df_test[TARGET_COL].values, y_pred)

    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        return pd.DataFrame(
            {"feature": FEATURE_COLS, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Prophet
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ProphetForecaster:
    params: dict[str, Any] = field(
        default_factory=lambda: {
            "daily_seasonality": True,
            "weekly_seasonality": True,
            "yearly_seasonality": True,
        }
    )
    model: Any = None

    def fit(self, df_train: pd.DataFrame) -> "ProphetForecaster":
        from prophet import Prophet

        prophet_df = df_train[["timestamp_utc", TARGET_COL]].rename(
            columns={"timestamp_utc": "ds", TARGET_COL: "y"}
        )
        prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)
        self.model = Prophet(**self.params)
        self.model.fit(prophet_df)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        future = df[["timestamp_utc"]].rename(columns={"timestamp_utc": "ds"})
        future["ds"] = future["ds"].dt.tz_localize(None)
        return self.model.predict(future)["yhat"].values

    def evaluate(self, df_test: pd.DataFrame) -> ForecastMetrics:
        y_pred = self.predict(df_test)
        return compute_metrics(df_test[TARGET_COL].values, y_pred)
