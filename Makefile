# ═══════════════════════════════════════════════════════════════════════════
# Energy Forecasting — developer Makefile
# ═══════════════════════════════════════════════════════════════════════════

.PHONY: help install lint format typecheck test test-cov clean \
        bronze silver gold train drift api dashboard docker-up docker-down

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install dependencies into active virtualenv
	pip install --upgrade pip
	pip install -r requirements.txt

lint:  ## Ruff lint
	ruff check src tests

format:  ## Black + ruff --fix
	black src tests
	ruff check --fix src tests

typecheck:  ## Mypy type-check
	mypy src

test:  ## Run unit tests
	pytest tests/ -v -m "not integration"

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

clean:  ## Remove caches, builds
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

# ── Pipeline ───────────────────────────────────────────────────────────────
bronze:  ## Run Bronze ingestion (last 7 days)
	python -m src.ingestion.run_bronze

silver:  ## Run Silver transformation
	python -m src.transformations.run_silver

gold:  ## Run Gold features
	python -m src.features.run_gold

train:  ## Train models + register to MLflow
	python -m src.models.train_all

drift:  ## Run drift check
	python -m src.monitoring.drift

pipeline:  ## Full pipeline: bronze → silver → gold → train
	$(MAKE) bronze && $(MAKE) silver && $(MAKE) gold && $(MAKE) train

# ── Services ───────────────────────────────────────────────────────────────
api:  ## Run FastAPI locally
	uvicorn src.serving.api:app --reload --host 0.0.0.0 --port 8000

dashboard:  ## Run Streamlit dashboard locally
	streamlit run dashboards/app.py

mlflow-ui:  ## Launch MLflow UI
	mlflow ui --backend-store-uri ./mlruns --port 5000

# ── Docker ─────────────────────────────────────────────────────────────────
docker-up:  ## Start full stack (API + MLflow + Dashboard)
	docker compose -f docker/docker-compose.yml up --build -d

docker-down:  ## Stop stack
	docker compose -f docker/docker-compose.yml down
