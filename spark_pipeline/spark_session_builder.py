"""
Spark Session Builder Utility

Provides optimized Spark session configuration for distributed processing
on Hadoop cluster with HDFS integration.
"""

import logging
from typing import Optional
from pyspark.sql import SparkSession
import yaml
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SparkSessionBuilder:
    """
    Builder class for creating optimized Spark sessions configured for
    large-scale data processing on Hadoop/HDFS clusters.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Spark session builder with configuration.
        
        Args:
            config_path: Path to YAML configuration file. If None, uses default config.
        """
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self.spark_session: Optional[SparkSession] = None
    
    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, '..', 'configs', 'spark_config.yaml')
    
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file not found at {self.config_path}, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Return default Spark configuration."""
        return {
            'spark': {
                'app_name': 'Amazon Product Recommendation System',
                'executor_memory': '4G',
                'driver_memory': '2G',
                'executor_cores': 2,
                'shuffle_partitions': 200,
                'default_parallelism': 8,
                'serializer': 'org.apache.spark.serializer.KryoSerializer',
                'sql_shuffle_partitions': 200,
                'sql_adaptive_enabled': True,
                'dynamic_allocation_enabled': False,
            }
        }
    
    def build(self) -> SparkSession:
        """
        Build and return an optimized Spark session.
        
        Returns:
            SparkSession: Configured Spark session instance
        """
        if self.spark_session is not None:
            logger.info("Returning existing Spark session")
            return self.spark_session
        
        spark_config = self.config.get('spark', {})
        
        # Initialize Spark session builder with explicit local master
        builder = SparkSession.builder \
            .master("local[*]") \
            .appName(spark_config.get('app_name', 'Amazon Recommendation System')) \
            .config("spark.executor.memory", spark_config.get('executor_memory', '4G')) \
            .config("spark.driver.memory", spark_config.get('driver_memory', '2G')) \
            .config("spark.executor.cores", spark_config.get('executor_cores', 2)) \
            .config("spark.sql.shuffle.partitions", spark_config.get('shuffle_partitions', 200)) \
            .config("spark.default.parallelism", spark_config.get('default_parallelism', 8)) \
            .config("spark.serializer", spark_config.get('serializer', 'org.apache.spark.serializer.KryoSerializer')) \
            .config("spark.sql.shuffle.partitions", spark_config.get('sql_shuffle_partitions', 200)) \
            .config("spark.sql.adaptive.enabled", spark_config.get('sql_adaptive_enabled', True)) \
            .config("spark.sql.adaptive.coalescePartitions.enabled", 
                   spark_config.get('sql_adaptive_coalesce_partitions_enabled', True)) \
            .config("spark.memory.fraction", spark_config.get('memory_fraction', 0.6)) \
            .config("spark.memory.storageFraction", spark_config.get('memory_storage_fraction', 0.5)) \
            .config("spark.network.timeout", spark_config.get('network_timeout', 86400)) \
            .config("spark.executor.heartbeatInterval", spark_config.get('executor_heartbeat_interval', 60)) \
            .config("spark.dynamicAllocation.enabled", 
                   spark_config.get('dynamic_allocation_enabled', False)) \
            .config("spark.dynamicAllocation.minExecutors", 
                   spark_config.get('dynamic_allocation_min_executors', 1)) \
            .config("spark.dynamicAllocation.maxExecutors", 
                   spark_config.get('dynamic_allocation_max_executors', 4)) \
            .config("spark.dynamicAllocation.initialExecutors", 
                   spark_config.get('dynamic_allocation_initial_executors', 2)) \
            .config("spark.driver.host", "localhost") \
            .config("spark.driver.bindAddress", "127.0.0.1")
        
        # Add HDFS configurations if present
        hadoop_config = spark_config.get('hadoop_configuration', {})
        for key, value in hadoop_config.items():
            builder.config(f"spark.hadoop.{key}", value)
        
        # Enable Kryo registration if specified
        if spark_config.get('kryo_registration_required', False):
            builder.config("spark.kryo.registrationRequired", "true")
        
        # Build the session
        logger.info("Building Spark session with optimized configuration...")
        self.spark_session = builder.getOrCreate()
        
        # Set log level
        self.spark_session.sparkContext.setLogLevel(
            self.config.get('logging', {}).get('level', 'INFO')
        )
        
        # Log configuration summary
        self._log_configuration_summary()
        
        logger.info("Spark session created successfully")
        return self.spark_session
    
    def _log_configuration_summary(self):
        """Log key Spark configuration parameters."""
        if self.spark_session:
            sc = self.spark_session.sparkContext
            logger.info("=" * 60)
            logger.info("SPARK CONFIGURATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Application Name: {sc.appName}")
            logger.info(f"Master: {sc.master}")
            logger.info(f"Default Parallelism: {sc.defaultParallelism}")
            logger.info(f"Max Result Size: {sc.getConf().get('spark.driver.maxResultSize', '1G')}")
            logger.info(f"Executor Memory: {sc.getConf().get('spark.executor.memory', '4G')}")
            logger.info(f"Driver Memory: {sc.getConf().get('spark.driver.memory', '2G')}")
            logger.info(f"SQL Shuffle Partitions: {sc.getConf().get('spark.sql.shuffle.partitions', '200')}")
            logger.info(f"Dynamic Allocation: {sc.getConf().get('spark.dynamicAllocation.enabled', 'true')}")
            logger.info("=" * 60)
    
    def get_config_value(self, key: str, default=None):
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key (dot-separated for nested keys)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def stop(self):
        """Stop the Spark session if active."""
        if self.spark_session:
            logger.info("Stopping Spark session...")
            self.spark_session.stop()
            self.spark_session = None
            logger.info("Spark session stopped")


def get_spark_session(config_path: Optional[str] = None) -> SparkSession:
    """
    Convenience function to get a Spark session.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        SparkSession: Configured Spark session
    """
    builder = SparkSessionBuilder(config_path)
    return builder.build()


if __name__ == "__main__":
    # Example usage
    spark = get_spark_session()
    print(f"Spark session created: {spark}")
    print(f"Spark version: {spark.version}")
    spark.stop()
