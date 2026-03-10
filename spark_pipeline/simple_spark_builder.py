"""
Simple Spark Session Utility

Provides a straightforward way to create Spark sessions for local processing
without complex configuration patterns.
"""

import logging
from pyspark.sql import SparkSession

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger= logging.getLogger(__name__)


def get_spark_session(app_name: str = "Amazon Recommendation System") -> SparkSession:
   logger.info("Creating Spark session...")
    
    spark = (SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        .config("spark.driver.host", "localhost")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.memory", "2G")
        .config("spark.executor.memory", "4G")
        .config("spark.executor.cores", "2")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.network.timeout", "86400")
        .config("spark.executor.heartbeatInterval", "60")
        .getOrCreate())
    
    spark.sparkContext.setLogLevel("WARN")
    
   logger.info("Spark session created successfully!")
   logger.info(f"Master: {spark.sparkContext.master}")
   logger.info(f"App Name: {spark.sparkContext.appName}")
    
  return spark


if __name__ == "__main__":
    spark = get_spark_session()
    print(f"Spark Version: {spark.version}")
    spark.stop()