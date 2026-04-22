# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver Cleaning
# MAGIC
# MAGIC **Layer:** Silver (cleaned, typed, quality-flagged)
# MAGIC **Input:** `/mnt/energy/bronze/load`
# MAGIC **Output:** `/mnt/energy/silver/load` (Delta, partitioned by `country`)

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F
from delta.tables import DeltaTable

dbutils.widgets.text(
    "base_path",
    "/Volumes/workspace/default/energy",
    label="Delta base path (Unity Catalog volume; override with dbfs:/... on classic DBR)",
)
base_path = dbutils.widgets.get("base_path").rstrip("/")
BRONZE_PATH = f"{base_path}/bronze/load"
SILVER_PATH = f"{base_path}/silver/load"

bronze = spark.read.format("delta").load(BRONZE_PATH)

# COMMAND ----------

# Deduplicate keeping the latest ingestion per (timestamp, country, kind)
w_latest = Window.partitionBy("timestamp_utc", "country", "kind").orderBy(F.col("ingested_at").desc())
dedup = bronze.withColumn("_rn", F.row_number().over(w_latest)).filter("_rn = 1").drop("_rn")

# Outlier detection (country-level IQR bands)
q1, q3 = dedup.approxQuantile("load_mw", [0.25, 0.75], 0.001)
iqr = q3 - q1
lower, upper = q1 - 3 * iqr, q3 + 3 * iqr

enriched = (
    dedup
    .withColumn("is_outlier", (F.col("load_mw") < lower) | (F.col("load_mw") > upper))
    .withColumn(
        "quality_flag",
        F.when(F.col("load_mw").isNull(), "missing")
         .when(F.col("is_outlier"), "outlier")
         .otherwise("ok"),
    )
)

# Forward-fill short gaps (up to 3 hours)
w_ffill = Window.partitionBy("country").orderBy("timestamp_utc").rowsBetween(-3, 0)
silver = enriched.withColumn(
    "load_mw_clean",
    F.coalesce(F.col("load_mw"), F.last("load_mw", ignorenulls=True).over(w_ffill)),
).select(
    "timestamp_utc", "country", "kind",
    "load_mw", "load_mw_clean",
    "is_outlier", "quality_flag", "ingested_at",
)

# COMMAND ----------

if DeltaTable.isDeltaTable(spark, SILVER_PATH):
    (
        DeltaTable.forPath(spark, SILVER_PATH).alias("t")
        .merge(
            silver.alias("s"),
            "t.timestamp_utc = s.timestamp_utc AND t.country = s.country AND t.kind = s.kind",
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
else:
    silver.write.format("delta").partitionBy("country").mode("overwrite").save(SILVER_PATH)

# COMMAND ----------

display(spark.read.format("delta").load(SILVER_PATH).groupBy("country", "quality_flag").count())
