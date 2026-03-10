#!/usr/bin/env python3
"""Production Spark Session Builder"""

import logging
import os
from pyspark.sql import SparkSession
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def get_spark_session():
    """Create Spark session for distributed processing on YARN."""

    # Set networking environment variables
    os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
    os.environ['SPARK_DRIVER_HOST'] = '127.0.0.1'
    os.environ['SPARK_DRIVER_BIND_ADDRESS'] = '127.0.0.1'

    logger.info("Creating Spark session for YARN cluster...")

    spark = (
        SparkSession.builder
        .appName("Amazon Product Recommendation System")
        .master("yarn")
        .config("spark.executor.memory", "4G")
        .config("spark.driver.memory", "2G")
        .config("spark.executor.cores", "2")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.default.parallelism", "8")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.network.timeout", "86400")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    logger.info("=" * 70)
    logger.info("SPARK SESSION CREATED")
    logger.info("=" * 70)
    logger.info(f"Master: {spark.sparkContext.master}")
    logger.info(f"App Name: {spark.sparkContext.appName}")
    logger.info("Spark UI: http://localhost:4040")
    logger.info("=" * 70)

    return spark


if __name__ == "__main__":
    spark = get_spark_session()
    print(f"Spark Version: {spark.version}")
    spark.stop()