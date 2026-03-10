#!/usr/bin/env python3
import logging, os, sys
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
   logger.info("="*70)
   logger.info("DISTRIBUTED RECOMMENDATION PIPELINE")
   logger.info("="*70)
    
   try:
        os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
        os.environ['SPARK_DRIVER_HOST'] = '127.0.0.1'
        os.environ['SPARK_DRIVER_BIND_ADDRESS'] = '127.0.0.1'
        
        spark = (SparkSession.builder
            .appName("Amazon Recommendation System")
            .master("yarn")
            .config("spark.executor.memory", "4G")
            .config("spark.driver.memory", "2G")
            .config("spark.executor.cores", "2")
            .config("spark.driver.host", "127.0.0.1")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .getOrCreate())
        
        spark.sparkContext.setLogLevel("WARN")
       logger.info(f"Spark UI: http://localhost:4040")
        
       hdfs_path = "hdfs://localhost:9000/dataset/all_csv_files.csv"
       logger.info("\nLoading data from HDFS...")
       df = spark.read.option("header", "false").option("inferSchema", "true").csv(hdfs_path)
       df= df.toDF("product_id", "user_id", "rating", "timestamp")
       logger.info(f"Loaded {df.count():,} records")
        
       logger.info("\nPreprocessing...")
       df = df.dropDuplicates(["user_id", "product_id"])
       df = df.filter(col("rating") > 0)
       df.cache()
       logger.info(f"After preprocessing: {df.count():,}")
        
       logger.info("\nFeature Engineering...")
        user_indexer = StringIndexer(inputCol="user_id", outputCol="user_idx", handleInvalid="skip")
       df = user_indexer.fit(df).transform(df)
        item_indexer = StringIndexer(inputCol="product_id", outputCol="item_idx", handleInvalid="skip")
       df = item_indexer.fit(df).transform(df)
       df= df.select("user_id", "product_id", "rating", "user_idx", "item_idx")
        
       logger.info("\nTrain/Test Split...")
       train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
       logger.info(f"Train: {train_df.count():,}, Test: {test_df.count():,}")
        
       logger.info("\nTraining ALS Model...")
        als = ALS(userCol="user_idx", itemCol="item_idx", ratingCol="rating",
                 coldStartStrategy="drop", nonnegative=True, rank=10, maxIter=15, regParam=0.1)
       start_time = datetime.now()
       model = als.fit(train_df)
       logger.info(f"Model trained in {datetime.now() -start_time}")
        
       logger.info("\nEvaluating...")
       predictions = model.transform(test_df)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
        rmse = evaluator.evaluate(predictions.na.drop())
       logger.info(f"RMSE: {rmse:.4f}")
        
       logger.info("\nGenerating Recommendations...")
        sample_users = df.select("user_idx").distinct().limit(100)
       recommendations = model.recommendForUserSubset(sample_users, 10)
       logger.info(f"Generated for {recommendations.count():,} users")
        
       logger.info("\nSaving to HDFS...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"hdfs://localhost:9000/recommendations/{timestamp}"
       recommendations.write.mode("overwrite").parquet(output_path)
       logger.info(f"Saved: {output_path}")
        
       logger.info("\n" + "="*70)
       logger.info("SUCCESS! PIPELINE COMPLETED")
       logger.info("="*70)
        
       print("\nCheck results: hdfs dfs -ls/recommendations/")
        
        spark.stop()
        
   except Exception as e:
       logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
