# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze Ingestion — ENTSO-E Load Data
# MAGIC
# MAGIC **Layer:** Bronze (raw, append-only Delta)
# MAGIC **Source:** ENTSO-E Transparency Platform — hourly electricity load per country
# MAGIC **Output:** `/mnt/energy/bronze/load` (Delta, partitioned by `country`)
# MAGIC
# MAGIC This notebook is idempotent: re-running over an overlapping window merges on `(timestamp_utc, country, kind)`.

# COMMAND ----------

# MAGIC %pip install entsoe-py==0.6.11 tenacity==9.0.0 structlog==24.4.0 -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from datetime import datetime, timedelta, timezone
import pandas as pd
from entsoe import EntsoePandasClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parameters

# COMMAND ----------

dbutils.widgets.text("country", "ES")
dbutils.widgets.text("backfill_days", "30")
dbutils.widgets.text("api_token_secret_scope", "energy")
dbutils.widgets.text("api_token_secret_key", "entsoe_token")

country = dbutils.widgets.get("country")
backfill_days = int(dbutils.widgets.get("backfill_days"))
api_token = dbutils.secrets.get(
    dbutils.widgets.get("api_token_secret_scope"),
    dbutils.widgets.get("api_token_secret_key"),
)

BRONZE_PATH = f"/mnt/energy/bronze/load"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fetch from ENTSO-E

# COMMAND ----------

now_utc = datetime.now(tz=timezone.utc)
start_ts = pd.Timestamp(now_utc - timedelta(days=backfill_days), tz="UTC")
end_ts = pd.Timestamp(now_utc, tz="UTC")

client = EntsoePandasClient(api_key=api_token)
series = client.query_load(country, start=start_ts, end=end_ts)
if isinstance(series, pd.DataFrame):
    series = series.iloc[:, 0]

pdf = series.rename("load_mw").reset_index()
pdf.columns = ["timestamp_utc", "load_mw"]
pdf["timestamp_utc"] = pd.to_datetime(pdf["timestamp_utc"], utc=True)
pdf["country"] = country
pdf["kind"] = "actual"
pdf["ingested_at"] = now_utc

print(f"Fetched {len(pdf)} rows for {country} ({start_ts} → {end_ts})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Bronze (Delta, partitioned + merged)

# COMMAND ----------

from delta.tables import DeltaTable

sdf = spark.createDataFrame(pdf)

if DeltaTable.isDeltaTable(spark, BRONZE_PATH):
    (
        DeltaTable.forPath(spark, BRONZE_PATH)
        .alias("t")
        .merge(
            sdf.alias("s"),
            "t.timestamp_utc = s.timestamp_utc AND t.country = s.country AND t.kind = s.kind",
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
else:
    (
        sdf.write.format("delta")
        .partitionBy("country")
        .mode("overwrite")
        .save(BRONZE_PATH)
    )

# COMMAND ----------

display(spark.read.format("delta").load(BRONZE_PATH).orderBy("timestamp_utc", ascending=False).limit(10))
