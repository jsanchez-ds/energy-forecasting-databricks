#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# bootstrap.sh — one-shot local setup
# ---------------------------------------------------------------------------
set -euo pipefail

echo "🚀 Bootstrapping energy-forecasting project..."

# 1. Virtualenv
if [[ ! -d .venv ]]; then
  echo "→ Creating .venv (Python 3.11)"
  python3.11 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 2. Dependencies
echo "→ Installing requirements"
pip install --upgrade pip
pip install -r requirements.txt

# 3. .env
if [[ ! -f .env ]]; then
  echo "→ Creating .env from template"
  cp .env.example .env
  echo "⚠️  Edit .env and set ENTSOE_API_TOKEN before running the pipeline."
fi

# 4. Data directories
mkdir -p data/{bronze,silver,gold} mlruns logs

echo "✅ Setup complete. Next:"
echo "    source .venv/bin/activate"
echo "    python -m src.ingestion.run_bronze --start 2024-01-01 --end 2024-12-31"
echo "    python -m src.transformations.run_silver"
echo "    python -m src.features.run_gold"
echo "    python -m src.models.train_all"
