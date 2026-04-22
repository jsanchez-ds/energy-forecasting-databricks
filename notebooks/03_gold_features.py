# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold — ML-Ready Features
# MAGIC
# MAGIC **Layer:** Gold (features + target)
# MAGIC **Input:** `/mnt/energy/silver/load`
# MAGIC **Output:** `/mnt/energy/gold/load_features` (Delta, partitioned by `country`)
# MAGIC
# MAGIC Adds calendar features, cyclic encodings, lags (1, 24, 48, 168h) and rolling statistics (24, 168h).

# COMMAND ----------

import numpy as np
import pandas as pd

SILVER_PATH = "/mnt/energy/silver/load"
GOLD_PATH = "/mnt/energy/gold/load_features"

silver_pdf = (
    spark.read.format("delta").load(SILVER_PATH)
    .filter("kind = 'actual'")
    .toPandas()
)

# COMMAND ----------

SPAIN_HOLIDAYS_MMDD = {"01-01","01-06","05-01","08-15","10-12","11-01","12-06","12-08","12-25"}

def add_calendar(df):
    ts = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["day_of_month"] = ts.dt.day
    df["month"] = ts.dt.month
    df["quarter"] = ts.dt.quarter
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
    df["dow_sin"] = np.sin(2*np.pi*df["day_of_week"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["day_of_week"]/7)
    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["is_holiday_es"] = ts.dt.strftime("%m-%d").isin(SPAIN_HOLIDAYS_MMDD).astype(int)
    return df

def add_lags_rolls(df, target="load_mw_clean"):
    for lag in [1, 24, 48, 168]:
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)
    shifted = df[target].shift(1)
    for w in [24, 168]:
        df[f"{target}_roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
        df[f"{target}_roll_std_{w}"] = shifted.rolling(w, min_periods=1).std()
    return df

# COMMAND ----------

frames = []
for c, g in silver_pdf.groupby("country"):
    g = g.sort_values("timestamp_utc").reset_index(drop=True)
    g = add_calendar(g)
    g = add_lags_rolls(g)
    frames.append(g)

gold_pdf = pd.concat(frames, ignore_index=True)
gold_pdf = gold_pdf.dropna(subset=[c for c in gold_pdf.columns if c.startswith("load_mw_clean_lag_")])
print(f"Gold rows: {len(gold_pdf):,}  cols: {gold_pdf.shape[1]}")

# COMMAND ----------

(
    spark.createDataFrame(gold_pdf)
    .write.format("delta").partitionBy("country").mode("overwrite")
    .option("overwriteSchema", "true")
    .save(GOLD_PATH)
)

display(spark.read.format("delta").load(GOLD_PATH).limit(5))
