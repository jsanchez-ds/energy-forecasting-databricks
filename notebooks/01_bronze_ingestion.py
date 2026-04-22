# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze Ingestion
# MAGIC
# MAGIC Pulls hourly electricity demand into the Bronze Delta table.
# MAGIC
# MAGIC * **Sources:** EIA Open Data (US) or ENTSO-E Transparency Platform (EU)
# MAGIC * **Output:** `dbfs:/FileStore/energy/bronze/load`, partitioned by `country` / region
# MAGIC * **Idempotent:** re-runs merge on `(timestamp_utc, country, kind)`

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dependencies
# MAGIC
# MAGIC `httpx` is the only non-stock dep on Databricks Runtime 15+ / Free Edition.
# MAGIC For the ENTSO-E path also add `entsoe-py`.

# COMMAND ----------

# MAGIC %pip install httpx==0.27.2 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from typing import Literal

import httpx
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters
# MAGIC
# MAGIC Community Edition does not have secret scopes enabled by default, so we
# MAGIC pass the API key as a widget.

# COMMAND ----------

dbutils.widgets.dropdown("source", "eia", ["eia", "entsoe"])
dbutils.widgets.text("region", "CAL", label="EIA region / ENTSO-E country")
dbutils.widgets.text("backfill_days", "30")
dbutils.widgets.text("api_key", "", label="EIA_API_KEY or ENTSOE_API_TOKEN")
dbutils.widgets.text("base_path", "dbfs:/FileStore/energy", label="Delta base path")

source: Literal["eia", "entsoe"] = dbutils.widgets.get("source")  # type: ignore
region = dbutils.widgets.get("region")
backfill_days = int(dbutils.widgets.get("backfill_days"))
api_key = dbutils.widgets.get("api_key")
base_path = dbutils.widgets.get("base_path").rstrip("/")
BRONZE_PATH = f"{base_path}/bronze/load"

assert api_key, "Paste your API key into the `api_key` widget (top of notebook)."

now_utc = pd.Timestamp.now(tz="UTC")
start_ts = now_utc - pd.Timedelta(days=backfill_days)
end_ts = now_utc

print(f"Source: {source}  Region: {region}  Window: {start_ts} → {end_ts}")
print(f"Writing to: {BRONZE_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch

# COMMAND ----------

def fetch_eia(api_key: str, region: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Paginate the EIA v2 endpoint in 60-day windows."""
    frames = []
    cursor = start
    while cursor < end:
        window_end = min(cursor + pd.Timedelta(days=60), end)
        params = {
            "api_key": api_key,
            "frequency": "hourly",
            "data[0]": "value",
            "facets[respondent][]": region,
            "facets[type][]": "D",
            "start": cursor.strftime("%Y-%m-%dT%H"),
            "end": window_end.strftime("%Y-%m-%dT%H"),
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "length": "5000",
        }
        r = httpx.get(
            "https://api.eia.gov/v2/electricity/rto/region-data/data/",
            params=params, timeout=60,
        )
        r.raise_for_status()
        data = r.json().get("response", {}).get("data", [])
        if data:
            df = pd.DataFrame(data)
            df["timestamp_utc"] = pd.to_datetime(df["period"], utc=True)
            df["load_mw"] = pd.to_numeric(df["value"], errors="coerce").astype("float64")
            frames.append(df[["timestamp_utc", "load_mw"]])
        cursor = window_end
    if not frames:
        return pd.DataFrame(columns=["timestamp_utc", "load_mw"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp_utc").sort_values("timestamp_utc")


def fetch_entsoe(api_key: str, country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    from entsoe import EntsoePandasClient
    client = EntsoePandasClient(api_key=api_key)
    series = client.query_load(country, start=start, end=end)
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    df = series.rename("load_mw").reset_index()
    df.columns = ["timestamp_utc", "load_mw"]
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


if source == "eia":
    pdf = fetch_eia(api_key, region, start_ts, end_ts)
else:
    pdf = fetch_entsoe(api_key, region, start_ts, end_ts)

pdf["country"] = region
pdf["kind"] = "actual"
pdf["ingested_at"] = now_utc
print(f"Fetched {len(pdf):,} rows")
display(pdf.head(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Bronze (Delta, partitioned + merged)

# COMMAND ----------

from delta.tables import DeltaTable

sdf = spark.createDataFrame(pdf)

if DeltaTable.isDeltaTable(spark, BRONZE_PATH):
    (
        DeltaTable.forPath(spark, BRONZE_PATH).alias("t")
        .merge(
            sdf.alias("s"),
            "t.timestamp_utc = s.timestamp_utc AND t.country = s.country AND t.kind = s.kind",
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
else:
    sdf.write.format("delta").partitionBy("country").mode("overwrite").save(BRONZE_PATH)

rows = spark.read.format("delta").load(BRONZE_PATH).count()
print(f"Bronze total rows: {rows:,}")

# COMMAND ----------

display(
    spark.read.format("delta").load(BRONZE_PATH)
    .orderBy("timestamp_utc", ascending=False).limit(10)
)
