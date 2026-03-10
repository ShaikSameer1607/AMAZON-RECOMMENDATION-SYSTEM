#!/usr/bin/env python3
"""
Fast Amazon Recommendation System - Optimized Single Script
Runs the complete pipeline in minimal time with optimized settings.
"""

import logging
import sys
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, mean, stddev, countDistinct, desc
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger= logging.getLogger(__name__)

def main():
    """Fast pipeline execution- optimized for speed."""
    
    start_time = time.time()
   logger.info("="*70)
   logger.info("FAST AMAZON RECOMMENDATION SYSTEM - OPTIMIZED PIPELINE")
   logger.info("="*70)
    
    # Step 1: Initialize Spark with optimized settings
   logger.info("[1/6] Initializing Spark session (optimized)...")
    spark = SparkSession.builder \
        .appName("Amazon Recommendation System - Fast") \
        .config("spark.executor.memory", "2G") \
        .config("spark.driver.memory", "2G") \
        .config("spark.executor.cores", "2") \
        .config("spark.sql.shuffle.partitions", "50") \
        .config("spark.default.parallelism", "4") \
        .config("spark.dynamicAllocation.enabled", "false") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("WARN")
   logger.info(f"✓ Spark initialized (took {time.time()-start_time:.1f}s)")
    
    step1_time = time.time()
    
    # Step 2: Load data from HDFS
    try:
       logger.info("[2/6] Loading data from HDFS...")
        df = spark.read.option("header", "true") \
                    .option("inferSchema", "true") \
                    .csv("hdfs://master:9000/dataset/all_csv_files.csv")
        
       row_count = df.count()
       logger.info(f"✓ Loaded {row_count:,} rows (took {time.time()-step1_time:.1f}s)")
    except Exception as e:
       logger.error(f"❌ Error loading data: {e}")
       logger.error("HINT: Check if HDFS is running and dataset exists")
        spark.stop()
        sys.exit(1)
    
    step2_time = time.time()
    
    # Step 3: Quick preprocessing
   logger.info("[3/6] Preprocessing data...")
    df_clean = df.select('user_id', 'product_id', 'rating') \
                 .dropna() \
                 .dropDuplicates() \
                 .filter((col('rating') >= 0.5) & (col('rating') <= 5.0))
    
    clean_count = df_clean.count()
   logger.info(f"✓ Cleaned data: {clean_count:,} rows ({time.time()-step2_time:.1f}s)")
    
    df_clean.cache()
    
    step3_time = time.time()
    
    # Step 4: Basic EDA
   logger.info("[4/6] Running exploratory analysis...")
    total_users = df_clean.select(countDistinct('user_id')).collect()[0][0]
    total_products = df_clean.select(countDistinct('product_id')).collect()[0][0]
    
    rating_stats = df_clean.select(
        mean('rating').alias('mean'),
        stddev('rating').alias('stddev')
    ).collect()[0]
    
   logger.info(f"  Users: {total_users:,}")
   logger.info(f"  Products: {total_products:,}")
   logger.info(f"  Mean Rating: {rating_stats['mean']:.2f}")
   logger.info(f"✓ EDA complete ({time.time()-step3_time:.1f}s)")
    
    step4_time = time.time()
    
    # Step 5: Train ALS model
   logger.info("[5/6] Training ALS model...")
    
    # Split data
    train_df, test_df = df_clean.randomSplit([0.8, 0.2], seed=42)
   logger.info(f"  Train: {train_df.count():,} | Test: {test_df.count():,}")
    
    # Configure ALS with faster settings
    als = ALS(
        userCol="user_id",
        itemCol="product_id",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
        rank=5,      # Smaller rank for speed
        maxIter=10,  # Fewer iterations
       regParam=0.1
    )
    
   model = als.fit(train_df)
   logger.info(f"✓ Model trained ({time.time()-step4_time:.1f}s)")
    
    step5_time = time.time()
    
    # Step 6: Evaluate
   logger.info("[6/6] Evaluating model...")
    
    predictions = model.transform(test_df)
    valid_preds = predictions.filter(predictions.prediction.isNotNull())
    
    evaluator = RegressionEvaluator(
        metricName="rmse",
        labelCol="rating",
        predictionCol="prediction"
    )
    
    rmse = evaluator.evaluate(valid_preds)
    mae = evaluator.setMetricName("mae").evaluate(valid_preds)
    
   logger.info(f"  RMSE: {rmse:.4f}")
   logger.info(f"  MAE: {mae:.4f}")
   logger.info(f"✓ Evaluation complete ({time.time()-step5_time:.1f}s)")
    
    # Generate sample recommendations
   logger.info("\nGenerating sample recommendations...")
    sample_user= df_clean.select('user_id').distinct().first()['user_id']
    
    user_items = df_clean.filter(df_clean.user_id == sample_user) \
                         .select('product_id').distinct()
    all_items = df_clean.select('product_id').distinct()
    candidate_items = all_items.subtract(user_items)
    
    user_candidates = spark.createDataFrame([(sample_user, row.product_id) 
                                            for row in candidate_items.take(100)],
                                           ['user_id', 'product_id'])
    
   recs = model.transform(user_candidates)
    top_recs = recs.orderBy(desc('prediction')).limit(5)
    
   logger.info(f"\nTop 5 Recommendations for User {sample_user}:")
    for row in top_recs.collect():
       logger.info(f"  Product {row.product_id}: {row.prediction:.2f}")
    
    # Save model
   model_path = "hdfs://master:9000/recommendation_results/models/als_model_fast"
   model.write().overwrite().save(model_path)
   logger.info(f"\n✓ Model saved to: {model_path}")
    
    # Summary
    total_time = time.time() - start_time
   logger.info("\n" + "="*70)
   logger.info("PIPELINE COMPLETE!")
   logger.info("="*70)
   logger.info(f"Total Execution Time: {total_time/60:.1f} minutes")
   logger.info(f"Dataset Size: {clean_count:,} ratings")
   logger.info(f"Users: {total_users:,} | Products: {total_products:,}")
   logger.info(f"Final RMSE: {rmse:.4f} | MAE: {mae:.4f}")
   logger.info("="*70)
    
    spark.stop()
   logger.info("\n✓ Spark session stopped")


if __name__ == "__main__":
    main()
