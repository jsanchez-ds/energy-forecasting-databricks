# ⚡ Real-Time Energy Demand Forecasting & Anomaly Detection

> **Production-grade MLOps pipeline on Databricks with Medallion architecture, MLflow registry, and drift monitoring.**

End-to-end data platform that ingests electricity-demand data from [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/), transforms it through a **Bronze → Silver → Gold** (Medallion) Delta Lake architecture, trains forecasting (LightGBM + Prophet) and anomaly-detection (Isolation Forest) models, tracks them in MLflow, and serves predictions through a FastAPI microservice with a Streamlit dashboard on top.

---

## 🏗️ Architecture

```
┌────────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   ENTSO-E API      │────▶│  Bronze (Delta)  │────▶│  Silver (Delta)  │
│  (15-min intervals)│     │   raw, append    │     │  cleaned, typed  │
└────────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                             │
                                                             ▼
        ┌─────────────────────────┐              ┌───────────────────────┐
        │  Streamlit Dashboard    │◀─────────────│   Gold (Delta)        │
        │  (forecast + anomalies) │              │   features + targets  │
        └────────────┬────────────┘              └──────────┬────────────┘
                     │                                      │
                     │ HTTPS                                ▼
                     ▼                         ┌───────────────────────┐
        ┌─────────────────────────┐            │   MLflow Tracking     │
        │    FastAPI Service      │◀───────────│   + Model Registry    │
        │  (Dockerized, /predict) │            └───────────┬───────────┘
        └─────────────────────────┘                        │
                     ▲                                     ▼
                     │                         ┌───────────────────────┐
                     └─────────────────────────│  Evidently AI         │
                                               │  (drift monitoring)   │
                                               └───────────────────────┘
```

---

## 🎯 What this project proves

| Capability | Evidence |
|---|---|
| **Cloud data engineering** | Medallion architecture (Bronze/Silver/Gold) with Delta Lake ACID tables |
| **Streaming-ready ingestion** | Idempotent incremental loads, schema evolution, watermarking |
| **Production ML** | MLflow experiments + Model Registry with staging/production aliases |
| **Multi-model serving** | Forecasting (LightGBM, Prophet) + Anomaly detection (Isolation Forest) in one pipeline |
| **MLOps rigor** | Unit tests, CI/CD, Docker, drift monitoring with Evidently |
| **Explainability** | SHAP values exposed via API |
| **Observability** | Prometheus metrics, structured logging, drift alerts |

---

## 📂 Project structure

```
.
├── src/
│   ├── ingestion/        # ENTSO-E API client, bronze loaders
│   ├── transformations/  # Bronze → Silver → Gold pipelines (PySpark)
│   ├── features/         # Feature engineering (calendar, lags, weather joins)
│   ├── models/           # Forecasting + anomaly detection
│   ├── serving/          # FastAPI app
│   ├── monitoring/       # Evidently drift reports
│   └── utils/            # Logging, config, spark session
├── notebooks/            # Databricks-exportable .py notebooks (Medallion)
├── dashboards/           # Streamlit app
├── tests/                # pytest suite
├── docker/               # Dockerfile + compose
├── configs/              # YAML configs per environment
├── scripts/              # Bootstrap / helper scripts
├── .github/workflows/    # CI/CD pipelines
└── docs/                 # Architecture decisions, diagrams
```

---

## 🚀 Quickstart

### 1. Requirements

- Python 3.11+
- Java 11 (for local PySpark)
- A **free ENTSO-E API token** → request at <https://transparency.entsoe.eu/> (Account → "Web API Security Token")
- Docker (optional, for serving)

### 2. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # fill in ENTSOE_API_TOKEN
```

### 3. Run the pipeline locally

```bash
# 1. Ingest raw data → Bronze
python -m src.ingestion.run_bronze --start 2024-01-01 --end 2024-12-31

# 2. Bronze → Silver
python -m src.transformations.run_silver

# 3. Silver → Gold (features)
python -m src.features.run_gold

# 4. Train forecasting + anomaly models (logs to MLflow)
python -m src.models.train_all

# 5. Launch API
uvicorn src.serving.api:app --reload --port 8000

# 6. Launch dashboard
streamlit run dashboards/app.py
```

### 4. Databricks deployment

Notebooks in `notebooks/` use the Databricks `# COMMAND ----------` format — import directly into a Databricks workspace (works with Community Edition).

---

## 🧪 Testing

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## 📊 Results — 2 years of CAISO data (2024-04-08 → 2026-04-22)

**Dataset:** 17,854 hourly observations of California grid demand pulled from EIA Open Data.
**Test window:** 30 days holdout (718 hours).

### Forecasting — MAPE on 30-day test set (lower is better)

| Model                       | Family            | MAPE      | RMSE (MW) | MAE (MW) | Notes |
|-----------------------------|-------------------|-----------|-----------|----------|-------|
| **LightGBM** (winner)       | Gradient boosting | **1.81%** | **700**   | **533**  | Promoted to `@staging` |
| LSTM (PyTorch, 168h window) | Deep learning     | 2.84%     | 1,053     | 817      | 2-layer, 64 hidden, 30 epochs CPU |
| Prophet                     | Bayesian stats    | _skipped_ | —         | —        | `cmdstan` needs MinGW on Windows |

LightGBM wins on this benchmark by a comfortable margin — as expected for well-engineered tabular features on hourly data. The LSTM is a respectable runner-up and would likely close the gap with hyperparameter tuning (longer sequences, larger hidden size, attention heads).

### Anomaly detection

| Model            | Anomalies found | Rate   | Score range        |
|------------------|-----------------|--------|--------------------|
| Isolation Forest | 179 / 17,854    | 1.00%  | -0.08 → 0.24       |

### Explainability (SHAP on the LightGBM winner)

The LightGBM model ships with SHAP artifacts logged to MLflow (`shap/shap_bar.png`, `shap/shap_beeswarm.png`, `shap/shap_values.csv`) so any consumer of the model can see which features drive predictions.

| Mean |SHAP| (global importance) | Feature-level impact distribution |
|---|---|
| ![shap bar](docs/images/shap_bar.png) | ![shap beeswarm](docs/images/shap_beeswarm.png) |

The top drivers are the recent load lags (`load_mw_clean_lag_1`, `load_mw_clean_lag_24`) and calendar cyclic encodings (`hour_sin`, `hour_cos`), matching the intuition that grid demand is dominated by strong diurnal + weekly seasonality plus persistence.

### LSTM training curve

![lstm training curve](docs/images/lstm_training_curve.png)

Validation loss flattens around epoch 10; early stopping (patience=5) kicks in soon after.

---

## 🏗️ Running on Databricks (Free Edition + Unity Catalog)

The same code runs end-to-end on Databricks Free Edition against a **Unity Catalog** workspace — Bronze/Silver/Gold as Delta tables inside a UC volume, MLflow experiments tracked at workspace level, and the winning model promoted to the Unity Catalog Model Registry with a `@staging` alias.

Benchmark on **400 days of CAISO hourly data** pulled fresh inside Databricks:

| Metric | Local pipeline | Databricks Free Edition |
|---|---|---|
| LightGBM MAPE | **1.81 %** | **1.83 %** |
| LightGBM RMSE (MW) | 700 | 705 |
| Isolation Forest anomaly rate | 1.00 % | 1.01 % |

Near-identical results across environments — the pipeline is reproducible and portable, not environment-coupled.

### Unity Catalog volume (Bronze / Silver / Gold as Delta)

![Unity Catalog volumes](docs/images/databricks_catalog.png)

The Medallion layers live under `workspace.default.energy/` as a UC-managed volume, so they inherit workspace-level governance, lineage and access control — no DBFS root needed.

### MLflow experiment — runs + metrics

![MLflow experiment](docs/images/databricks_experiment.png)

Each training run logs params, metrics and the signed model artifact. The `lightgbm-forecaster` and `isolation-forest-anomaly` runs are compared side by side.

### Unity Catalog Model Registry — `@staging` alias

![Unity Catalog model registry](docs/images/databricks_registry.png)

`workspace.default.energy-demand-forecaster` version 1 carries the `@staging` alias with a full input/output signature, ready to be served via Databricks Model Serving or consumed by the local FastAPI microservice via `models:/...@staging`.

---

## 📸 UI screenshots (local stack)

<!-- User-captured — see docs/images/README.md for the capture checklist. -->

| Streamlit dashboard | MLflow experiments |
|---|---|
| ![dashboard](docs/images/dashboard.png) | ![mlflow](docs/images/mlflow_experiments.png) |
| _17.8k records, California duck curve._ | _LightGBM + LSTM runs side by side with metrics + SHAP artifacts._ |

---

## 📜 License

MIT
