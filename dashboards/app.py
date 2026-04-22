"""
Streamlit dashboard — executive view of forecasts + anomalies.

Reads Gold Delta table directly (with pandas/pyarrow) and calls the FastAPI
service for live predictions.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Energy Forecasting | Executive Dashboard",
    page_icon="⚡",
    layout="wide",
)

GOLD_PATH = Path(os.getenv("GOLD_PATH", "./data/gold")) / "load_features"
API_URL = os.getenv("API_URL", "http://localhost:8000")


@st.cache_data(ttl=300)
def load_gold() -> pd.DataFrame:
    """Load Gold features from local Delta table via pandas+pyarrow."""
    try:
        from deltalake import DeltaTable

        return DeltaTable(str(GOLD_PATH)).to_pandas()
    except Exception:
        # Fallback: try parquet directly (for dev without delta-rs installed)
        return pd.read_parquet(GOLD_PATH)


# ── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("⚡ Energy Forecasting")
st.sidebar.caption("Real-time demand forecast + anomaly detection")

try:
    df = load_gold()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
except Exception as exc:
    st.error(f"Could not load Gold table at `{GOLD_PATH}`: {exc}")
    st.stop()

available_regions = sorted(df["country"].unique().tolist())
country = st.sidebar.selectbox("Region", options=available_regions, index=0)
horizon = st.sidebar.slider("Forecast horizon (hours)", 1, 168, 24)
df = df[df["country"] == country].sort_values("timestamp_utc")

# ── Header KPIs ─────────────────────────────────────────────────────────────
st.title("Energy Demand — Forecast & Anomalies")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Records", f"{len(df):,}")
col2.metric("Avg load (MW)", f"{df['load_mw_clean'].mean():,.0f}")
col3.metric("Peak load (MW)", f"{df['load_mw_clean'].max():,.0f}")
col4.metric(
    "Coverage",
    f"{df['timestamp_utc'].min().strftime('%Y-%m-%d')} → {df['timestamp_utc'].max().strftime('%Y-%m-%d')}",
)

# ── Demand chart ────────────────────────────────────────────────────────────
st.subheader(f"Demand — last 30 days ({country})")
recent = df.tail(24 * 30)

fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
fig.add_trace(
    go.Scatter(
        x=recent["timestamp_utc"],
        y=recent["load_mw_clean"],
        name="Actual load (MW)",
        mode="lines",
        line={"width": 1.5},
    )
)
fig.add_trace(
    go.Scatter(
        x=recent["timestamp_utc"],
        y=recent["load_mw_clean_roll_mean_168"],
        name="7-day rolling mean",
        mode="lines",
        line={"dash": "dash", "width": 1.2},
    )
)
fig.update_layout(
    height=420, hovermode="x unified", margin={"t": 20, "b": 10, "l": 10, "r": 10}
)
st.plotly_chart(fig, use_container_width=True)

# ── Daily profile ───────────────────────────────────────────────────────────
st.subheader("Hourly profile (median ± IQR)")
profile = (
    df.groupby("hour")["load_mw_clean"]
    .agg(["median", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    .rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75"})
    .reset_index()
)
fig2 = go.Figure()
fig2.add_trace(
    go.Scatter(
        x=profile["hour"], y=profile["q75"], fill=None, mode="lines",
        line={"width": 0}, showlegend=False,
    )
)
fig2.add_trace(
    go.Scatter(
        x=profile["hour"], y=profile["q25"], fill="tonexty", mode="lines",
        line={"width": 0}, name="IQR", fillcolor="rgba(0,120,255,0.15)",
    )
)
fig2.add_trace(
    go.Scatter(
        x=profile["hour"], y=profile["median"], mode="lines+markers",
        name="Median", line={"width": 2.5},
    )
)
fig2.update_layout(
    height=360, xaxis_title="Hour of day (UTC)", yaxis_title="Load (MW)",
    margin={"t": 20, "b": 10, "l": 10, "r": 10},
)
st.plotly_chart(fig2, use_container_width=True)

# ── Model status ────────────────────────────────────────────────────────────
st.subheader("Model status")
try:
    import httpx

    info = httpx.get(f"{API_URL}/model-info", timeout=3).json()
    st.json(info)
except Exception as exc:
    st.warning(f"API not reachable at `{API_URL}`: {exc}")

st.caption(
    "Built with Databricks · Delta Lake · MLflow · FastAPI · Streamlit · "
    "data from ENTSO-E Transparency Platform."
)
