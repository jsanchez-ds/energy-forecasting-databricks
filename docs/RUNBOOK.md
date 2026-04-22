# Runbook

## First-time setup

1. **Get an ENTSO-E token** (free). Register at <https://transparency.entsoe.eu/>, go to **My Account Settings → Web API Security Token → Generate a new token**.

2. **Bootstrap**:
   ```bash
   bash scripts/bootstrap.sh
   # edit .env → paste ENTSOE_API_TOKEN
   ```

3. **Run the full pipeline locally**:
   ```bash
   make pipeline      # bronze → silver → gold → train
   ```

4. **Start the services**:
   ```bash
   make api &         # http://localhost:8000/docs
   make mlflow-ui &   # http://localhost:5000
   make dashboard &   # http://localhost:8501
   ```

## Common operations

| Goal | Command |
|---|---|
| Backfill a specific window | `python -m src.ingestion.run_bronze --start 2023-01-01 --end 2023-12-31` |
| Re-train only | `make train` |
| Check for drift | `make drift` |
| Promote staging → production | `mlflow models update --name energy-demand-forecaster --alias production --version N` |
| Spin up full stack (Docker) | `make docker-up` |

## Troubleshooting

**`ModuleNotFoundError: delta`** → Run `pip install delta-spark==3.2.1` inside the venv. Delta needs to be on the Spark classpath — `src/utils/spark.py` handles this via `configure_spark_with_delta_pip`.

**Prophet fails to fit on Windows** → CmdStan compilation can be finicky. Skip Prophet by removing it from `configs/config.yaml` — LightGBM is the primary model anyway.

**`entsoe.InvalidParameterError`** → Your `TARGET_COUNTRY` code isn't a valid ENTSO-E bidding zone. Check <https://github.com/EnergieID/entsoe-py#areas>.

**API 503 on `/predict`** → Model not loaded. Check `/model-info`. Root cause is usually MLflow registry pointing at a missing version — run `make train` and the alias will be re-set.
