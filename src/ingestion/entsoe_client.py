"""
ENTSO-E Transparency Platform client.

Wraps `entsoe-py` with retries, pagination safety and a pandas-friendly output.
Used by the Bronze ingestion layer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pandas as pd
from entsoe import EntsoePandasClient
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logging import get_logger

log = get_logger(__name__)


class EntsoeClient:
    """Thin wrapper over `EntsoePandasClient` with retry + structured logging."""

    def __init__(self, api_token: str, country_code: str = "ES") -> None:
        if not api_token or api_token == "your_token_here":
            raise ValueError(
                "ENTSO-E API token missing. Request one at "
                "https://transparency.entsoe.eu/ (Account → Web API Security Token)."
            )
        self._client = EntsoePandasClient(api_key=api_token)
        self.country_code = country_code

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def fetch_load(
        self,
        start: datetime | str,
        end: datetime | str,
        kind: Literal["actual", "forecast"] = "actual",
    ) -> pd.DataFrame:
        """
        Fetch electricity load (MW) for the configured country.

        Parameters
        ----------
        start, end : datetime | str
            Inclusive range. If str, parsed with pandas.Timestamp.
        kind : {"actual", "forecast"}
            ENTSO-E publishes both; "actual" is the ground truth.

        Returns
        -------
        DataFrame with columns: ['timestamp_utc', 'country', 'load_mw', 'kind'].
        """
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")

        log.info(
            "entsoe.fetch_load.start",
            country=self.country_code,
            start=str(start_ts),
            end=str(end_ts),
            kind=kind,
        )

        if kind == "actual":
            series = self._client.query_load(self.country_code, start=start_ts, end=end_ts)
        else:
            series = self._client.query_load_forecast(
                self.country_code, start=start_ts, end=end_ts
            )

        if isinstance(series, pd.DataFrame):
            # entsoe-py sometimes returns a single-column DataFrame
            series = series.iloc[:, 0]

        df = series.rename("load_mw").reset_index()
        df.columns = ["timestamp_utc", "load_mw"]
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["country"] = self.country_code
        df["kind"] = kind

        log.info(
            "entsoe.fetch_load.done",
            rows=len(df),
            min_ts=str(df["timestamp_utc"].min()) if not df.empty else None,
            max_ts=str(df["timestamp_utc"].max()) if not df.empty else None,
        )
        return df[["timestamp_utc", "country", "load_mw", "kind"]]

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        reraise=True,
    )
    def fetch_generation_by_source(
        self, start: datetime | str, end: datetime | str
    ) -> pd.DataFrame:
        """Generation disaggregated by fuel source (wind, solar, nuclear, ...)."""
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")

        log.info(
            "entsoe.fetch_generation.start",
            country=self.country_code,
            start=str(start_ts),
            end=str(end_ts),
        )
        df = self._client.query_generation(self.country_code, start=start_ts, end=end_ts)
        df = df.reset_index().melt(id_vars=df.index.name or "index",
                                    var_name="source",
                                    value_name="generation_mw")
        df = df.rename(columns={df.columns[0]: "timestamp_utc"})
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["country"] = self.country_code

        log.info("entsoe.fetch_generation.done", rows=len(df))
        return df[["timestamp_utc", "country", "source", "generation_mw"]]
