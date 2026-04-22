"""Typed configuration loader — reads YAML + .env and exposes a single settings object."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvSettings(BaseSettings):
    """Secrets and environment-specific values loaded from `.env`."""

    entsoe_api_token: str = Field(..., description="ENTSO-E Transparency Platform token")
    bronze_path: str = "./data/bronze"
    silver_path: str = "./data/silver"
    gold_path: str = "./data/gold"
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "energy-forecasting"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    target_country: str = "ES"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def load_yaml_config(path: Path | str = CONFIG_PATH) -> dict[str, Any]:
    """Load the main YAML configuration."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_env() -> EnvSettings:
    """Return cached environment settings."""
    return EnvSettings()  # type: ignore[call-arg]
