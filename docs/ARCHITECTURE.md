# Architecture

## Why Medallion?

The Bronze → Silver → Gold pattern (popularised by Databricks) gives us:

- **Bronze** — an *immutable, append-only* record of raw source data. Invaluable for audit and for reprocessing when upstream schema/quality rules change.
- **Silver** — cleaned, deduplicated, quality-flagged. One row per logical fact. Business users query this for exploratory work.
- **Gold** — purpose-built marts. Here we pre-compute ML features (lags, rolling stats, cyclic encodings) so training becomes a simple `read → fit`.

All three layers are **Delta Lake** tables, so we get ACID transactions, time-travel, schema evolution, and `MERGE` upserts out of the box.

## Why both LightGBM and Prophet?

They have complementary failure modes:

| | LightGBM | Prophet |
|---|---|---|
| Handles exogenous features | ✅ (lags, calendar, weather) | Limited |
| Uncertainty intervals | Requires quantile regression | Built-in |
| Interpretability | SHAP | Additive components |
| Extrapolation | Weak | Good (trend-aware) |

We train both, log both to MLflow, and the registry promotes whichever has the lower MAPE on the 30-day test window. This is basically "model selection as code".

## Why Isolation Forest for anomalies?

Energy load anomalies include both *point* anomalies (sensor glitches, storms) and *contextual* anomalies (demand that would be normal at noon but is abnormal at 3 AM). Isolation Forest handles both by partitioning feature space and measuring how easily a point gets isolated. It also scales fine to millions of rows and doesn't need labelled data.

## Why MLflow Registry + aliases?

Aliases (`@staging`, `@production`) decouple the deployment target from a specific version. When the API reloads `models:/energy-demand-forecaster@staging`, it picks up whichever version the registry currently points to — so a promote = one CLI command, no code change, no redeploy.

## Serving choices

- **FastAPI + Uvicorn** — async, Pydantic-typed, near-zero-overhead over a synchronous `model.predict()`.
- **Prometheus metrics** at `/metrics` → scrapable by any observability stack.
- **Dockerised** with a multi-stage build, non-root user, and a `HEALTHCHECK`.
- **Streamlit** for the dashboard — cheap, fast to iterate on, talks directly to the API for model info and to the Gold Delta table for historical context.

## Trade-offs / limitations

- **Databricks Community Edition** doesn't expose Unity Catalog or Delta Live Tables. The `notebooks/` assume mounted DBFS paths (`/mnt/energy/...`) — adjust to your workspace.
- **ENTSO-E** throttles to ~400 requests/min. For large backfills, batch by month.
- **Prophet** is occasionally flaky on Windows due to its `cmdstanpy` backend — LightGBM is the fallback default.
