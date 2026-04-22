"""
EIA Open Data client (https://www.eia.gov/opendata/).

Wraps the EIA v2 REST API (`/electricity/rto/region-data/`) to fetch hourly
electricity demand for US Balancing Authorities (CAISO, ERCOT, PJM, MISO, ...).

Exposes a pandas-friendly output matching the Bronze schema so the rest of the
pipeline is source-agnostic.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.utils.logging import get_logger

log = get_logger(__name__)

# Common Balancing Authorities (respondent codes). Full list:
# https://www.eia.gov/opendata/browser/electricity/rto/region-data
KNOWN_REGIONS = {
    "CAL": "California (CAISO)",
    "CISO": "California ISO",
    "ERCO": "Texas (ERCOT)",
    "MISO": "Midwest (MISO)",
    "PJM": "Mid-Atlantic (PJM)",
    "NYIS": "New York (NYISO)",
    "ISNE": "New England (ISO-NE)",
    "SWPP": "Southwest Power Pool",
    "TEX": "Texas region",
    "NY": "New York region",
    "US48": "US Lower-48 aggregate",
}

# "Type" facet values on the API:
#   D  = demand (actual hourly consumption — use this as target)
#   DF = day-ahead demand forecast
#   NG = net generation
#   TI = total interchange


class EiaClient:
    """Thin typed wrapper over the EIA v2 REST API with tenacity retries."""

    BASE_URL = "https://api.eia.gov/v2"

    def __init__(self, api_key: str, region: str = "CAL", timeout: float = 30.0) -> None:
        if not api_key or api_key == "your_token_here":
            raise ValueError(
                "EIA API key missing. Request one (free, instant) at "
                "https://www.eia.gov/opendata/register.php"
            )
        if region not in KNOWN_REGIONS:
            log.warning(
                "eia.unknown_region",
                region=region,
                hint=f"Known: {list(KNOWN_REGIONS)[:6]}...",
            )
        self.api_key = api_key
        self.region = region
        self._client = httpx.Client(timeout=timeout)

    # ── Public API ─────────────────────────────────────────────────────────
    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
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
        Fetch hourly electricity demand (MWh) for the configured region.

        EIA v2 caps a single response at 5000 rows, so we paginate by month.
        """
        start_ts = pd.Timestamp(start).tz_convert("UTC") if _is_tz(start) else pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end).tz_convert("UTC") if _is_tz(end) else pd.Timestamp(end, tz="UTC")

        type_code = "D" if kind == "actual" else "DF"

        log.info(
            "eia.fetch_load.start",
            region=self.region,
            start=str(start_ts),
            end=str(end_ts),
            kind=kind,
        )

        frames: list[pd.DataFrame] = []
        # Slide in ~60-day windows to stay well under the 5000-row page cap
        window = timedelta(days=60)
        cursor = start_ts
        while cursor < end_ts:
            window_end = min(cursor + window, end_ts)
            page = self._fetch_page(cursor, window_end, type_code)
            if not page.empty:
                frames.append(page)
            cursor = window_end

        if not frames:
            log.warning("eia.empty_response")
            return pd.DataFrame(columns=["timestamp_utc", "country", "load_mw", "kind"])

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")
        df["country"] = self.region  # reuse the 'country' column for region
        df["kind"] = kind

        log.info(
            "eia.fetch_load.done",
            rows=len(df),
            min_ts=str(df["timestamp_utc"].min()) if not df.empty else None,
            max_ts=str(df["timestamp_utc"].max()) if not df.empty else None,
        )
        return df[["timestamp_utc", "country", "load_mw", "kind"]]

    # ── Internals ──────────────────────────────────────────────────────────
    def _fetch_page(
        self, start: pd.Timestamp, end: pd.Timestamp, type_code: str
    ) -> pd.DataFrame:
        params = {
            "api_key": self.api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": self.region,
            "facets[type][]": type_code,
            "start": start.strftime("%Y-%m-%dT%H"),
            "end": end.strftime("%Y-%m-%dT%H"),
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": "5000",
        }
        url = f"{self.BASE_URL}/electricity/rto/region-data/data/"
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()

        if "response" not in payload or "data" not in payload["response"]:
            log.warning("eia.malformed_response", keys=list(payload.keys()))
            return pd.DataFrame()

        data = payload["response"]["data"]
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df["timestamp_utc"] = pd.to_datetime(df["period"], utc=True)
        df["load_mw"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["timestamp_utc", "load_mw"]].dropna()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EiaClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _is_tz(ts: datetime | str) -> bool:
    if isinstance(ts, str):
        return False
    return ts.tzinfo is not None
