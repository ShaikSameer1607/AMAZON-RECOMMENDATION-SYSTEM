#!/usr/bin/env python3

import logging
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import hash
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():

    logger.info("="*60)
    logger.info("DISTRIBUTED ALS RECOMMENDATION SYSTEM")
    logger.info("="*60)

    spark = (
    	SparkSession.builder
    	.appName("Amazon ALS Recommender")
    	.master("local[*]")
    	.config("spark.driver.memory","8G")
    	.config("spark.executor.memory","8G")
    	.config("spark.sql.shuffle.partitions","200")
    	.config("spark.default.parallelism","200")
    	.getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    logger.info("Spark UI: http://localhost:4040")

    # ------------------------------------------------
    # Load dataset
    # ------------------------------------------------

    path = "hdfs://localhost:9000/dataset/all_csv_files.csv"

    logger.info("Loading dataset from HDFS...")

    df = (
        spark.read
        .option("header","false")
        .option("inferSchema","true")
        .csv(path)
    )

    df = df.toDF("product_id","user_id","rating","timestamp")

    # ------------------------------------------------
    # SAMPLE EARLY (VERY IMPORTANT)
    # ------------------------------------------------

    df = df.sample(fraction=0.01, seed=42)

    logger.info(f"Dataset after sampling: {df.count():,} rows")

    # ------------------------------------------------
    # Cleaning
    # ------------------------------------------------

    logger.info("Cleaning dataset...")

    df = (
        df.dropDuplicates(["user_id","product_id"])
        .filter(col("rating") > 0)
    )

    logger.info(f"Dataset after cleaning: {df.count():,} rows")

    # ------------------------------------------------
    # Convert IDs to numeric using StringIndexer
    # ------------------------------------------------

    logger.info("Creating numeric IDs...")
    
    logger.info("Creating numeric IDs...")

    df = df.withColumn("user_idx", hash(col("user_id")))
    df = df.withColumn("item_idx", hash(col("product_id")))

    df = df.select("user_idx", "item_idx", "rating")



          # ------------------------------------------------
    # Repartition for stability
    # ------------------------------------------------

    df = df.repartition(200).cache()

    df.count()

    # ------------------------------------------------
    # Train/Test split
    # ------------------------------------------------

    train, test = df.randomSplit([0.8,0.2], seed=42)

    logger.info(f"Train rows: {train.count():,}")
    logger.info(f"Test rows: {test.count():,}")

    # ------------------------------------------------
    # ALS Model
    # ------------------------------------------------

    logger.info("Training ALS model...")

    start = datetime.now()

    als = ALS(
        userCol="user_idx",
        itemCol="item_idx",
        ratingCol="rating",
        rank=8,
        maxIter=3,
        regParam=0.1,
        coldStartStrategy="drop",
        nonnegative=True,
        numUserBlocks=20,
        numItemBlocks=20
    )

    model = als.fit(train)

    logger.info(f"Training completed in {datetime.now()-start}")

    # ------------------------------------------------
    # Predictions
    # ------------------------------------------------

    logger.info("Generating predictions...")

    predictions = model.transform(test)

    # ------------------------------------------------
    # Evaluate on SAMPLE to avoid OOM
    # ------------------------------------------------

    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )

    pred_sample = predictions.sample(0.2)

    rmse = evaluator.evaluate(pred_sample)

    logger.info(f"RMSE = {rmse}")

    # ------------------------------------------------
    # Generate recommendations
    # ------------------------------------------------

    logger.info("Generating recommendations...")

    users = df.select("user_idx").distinct().limit(1000)

    # Generate recommendations
    recs = model.recommendForUserSubset(users, 10)

# Reduce output size (prevents driver crash)
    recs = recs.limit(50000)

# Repartition before writing (important)
    recs = recs.repartition(10)

# Save to HDFS
    output = f"hdfs://localhost:9000/recommendations/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    recs.write.mode("overwrite").parquet(output)

    logger.info(f"Recommendations saved successfully to {output}")

    spark.stop()


if __name__ == "__main__":
    main()
