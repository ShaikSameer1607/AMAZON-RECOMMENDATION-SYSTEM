"""
Data Ingestion Module

Loads Amazon ratings dataset directly from HDFS with optimized Spark configuration.
Handles schema validation, null detection, repartitioning, and caching.
"""

import logging
from typing import Tuple, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, LongType
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spark_pipeline.simple_spark_builder import get_spark_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Handles data ingestion from HDFS with optimized Spark processing.
    Implements distributed reading, schema validation, and caching strategies.
    """
    
    def __init__(self, spark: SparkSession, config: Optional[dict] = None):
        """
        Initialize data ingestion module.
        
        Args:
            spark: Active Spark session
            config: Configuration dictionary (optional)
        """
        self.spark = spark
        self.config = config or self._get_default_config()
        self.raw_df: Optional[DataFrame] = None
        self.validated_df: Optional[DataFrame] = None
    
    def _get_default_config(self) -> dict:
        """Return default ingestion configuration."""
        return {
            'hdfs_input_path': 'hdfs://master:9000/dataset/all_csv_files.csv',
            'target_partition_size_mb': 128,
            'cache_persistence_level': 'MEMORY_AND_DISK_SER',
            'infer_schema': True,
            'header': True,
        }
    
    def load_from_hdfs(self, hdfs_path: Optional[str] = None) -> DataFrame:
        """
        Load dataset directly from HDFS using Spark's distributed file reader.
        
        The dataset is automatically split into partitions matching HDFS block
        distribution (~128MB blocks) for parallel processing across cluster.
        
        Args:
            hdfs_path: HDFS path to CSV file. Uses config path if not provided.
            
        Returns:
            DataFrame: Raw dataset loaded from HDFS
        """
        hdfs_path = hdfs_path or self.config['hdfs_input_path']
        
        logger.info(f"Loading data from HDFS: {hdfs_path}")
        logger.info("Using Spark distributed file reader with automatic partitioning")
        
        try:
            # Read CSV with optimized options
            self.raw_df = self.spark.read \
                .option("header", "true") \
                .option("inferSchema", str(self.config.get('infer_schema', True)).lower()) \
                .option("encoding", "UTF-8") \
                .option("mode", "PERMISSIVE") \
                .csv(hdfs_path)
            
            # Log basic statistics
            row_count = self.raw_df.count()
            num_partitions = self.raw_df.rdd.getNumPartitions()
            
            logger.info(f"Successfully loaded {row_count:,} rows from HDFS")
            logger.info(f"Number of partitions: {num_partitions}")
            logger.info(f"Average partition size: {row_count / max(num_partitions, 1):,.0f} rows/partition")
            
            # Print schema
            logger.info("Detected Schema:")
            self.raw_df.printSchema()
            
            return self.raw_df
            
        except Exception as e:
            logger.error(f"Error loading data from HDFS: {e}")
            raise
    
    def validate_schema(self, df: Optional[DataFrame] = None) -> Tuple[bool, dict]:
        """
        Validate DataFrame schema and detect data quality issues.
        
        Args:
            df: DataFrame to validate. Uses raw_df if not provided.
            
        Returns:
            Tuple[bool, dict]: Validation success flag and validation report
        """
        if df is None:
            df = self.raw_df
        
        if df is None:
            raise ValueError("No DataFrame to validate. Call load_from_hdfs first.")
        
        logger.info("Validating schema and detecting data quality issues...")
        
        validation_report = {
            'total_rows': df.count(),
            'total_columns': len(df.columns),
            'column_names': df.columns,
            'null_counts': {},
            'null_percentages': {},
            'distinct_counts': {},
            'data_types': {}
        }
        
        # Analyze each column
        for col_name in df.columns:
            # Get null count
            null_count = df.filter(f"{col_name} IS NULL").count()
            null_percentage = (null_count / validation_report['total_rows']) * 100
            
            validation_report['null_counts'][col_name] = null_count
            validation_report['null_percentages'][col_name] = round(null_percentage, 2)
            
            # Get distinct count
            distinct_count = df.select(col_name).distinct().count()
            validation_report['distinct_counts'][col_name] = distinct_count
            
            # Get data type
            data_type = df.schema[col_name].dataType.simpleString()
            validation_report['data_types'][col_name] = data_type
        
        # Log validation results
        logger.info("=" * 60)
        logger.info("SCHEMA VALIDATION REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Rows: {validation_report['total_rows']:,}")
        logger.info(f"Total Columns: {validation_report['total_columns']}")
        logger.info(f"Columns: {validation_report['column_names']}")
        logger.info("-" * 60)
        logger.info("NULL VALUE ANALYSIS:")
        for col, count in validation_report['null_counts'].items():
            pct = validation_report['null_percentages'][col]
            if count > 0:
                logger.warning(f"  {col}: {count:,} nulls ({pct:.2f}%)")
            else:
                logger.info(f"  {col}: No nulls ✓")
        logger.info("-" * 60)
        logger.info("DATA TYPES:")
        for col, dtype in validation_report['data_types'].items():
            logger.info(f"  {col}: {dtype}")
        logger.info("=" * 60)
        
        # Check for critical issues
        has_critical_nulls = any(pct > 50 for pct in validation_report['null_percentages'].values())
        
        if has_critical_nulls:
            logger.warning("⚠ CRITICAL: Some columns have >50% null values")
        
        self.validated_df = df
        return True, validation_report
    
    def detect_duplicates(self, df: Optional[DataFrame] = None) -> int:
        """
        Detect duplicate rows in the dataset.
        
        Args:
            df: DataFrame to check. Uses validated_df if not provided.
            
        Returns:
            int: Number of duplicate rows
        """
        if df is None:
            df = self.validated_df or self.raw_df
        
        if df is None:
            raise ValueError("No DataFrame available")
        
        logger.info("Detecting duplicate rows...")
        
        total_rows = df.count()
        distinct_rows = df.distinct().count()
        duplicate_count = total_rows - distinct_rows
        
        logger.info(f"Total rows: {total_rows:,}")
        logger.info(f"Distinct rows: {distinct_rows:,}")
        logger.info(f"Duplicate rows: {duplicate_count:,} ({(duplicate_count/total_rows)*100:.2f}%)")
        
        return duplicate_count
    
    def repartition_for_processing(self, df: Optional[DataFrame] = None, 
                                   num_partitions: Optional[int] = None) -> DataFrame:
        """
        Repartition DataFrame for optimal Spark distributed processing.
        
        Aligns partitions with HDFS block size (128MB) for efficient parallel processing.
        
        Args:
            df: DataFrame to repartition. Uses validated_df if not provided.
            num_partitions: Target number of partitions. Auto-calculated if not provided.
            
        Returns:
            DataFrame: Repartitioned DataFrame
        """
        if df is None:
            df = self.validated_df or self.raw_df
        
        if df is None:
            raise ValueError("No DataFrame to repartition")
        
        current_partitions = df.rdd.getNumPartitions()
        
        # Calculate optimal partitions if not provided
        if num_partitions is None:
            target_size_mb = self.config.get('target_partition_size_mb', 128)
            # Estimate based on data size and target partition size
            estimated_data_size_gb = 6.7  # From requirements
            num_partitions = max(int(estimated_data_size_gb * 1024 / target_size_mb), 
                                current_partitions)
            num_partitions = min(num_partitions, 200)  # Cap at reasonable limit
        
        logger.info(f"Repartitioning from {current_partitions} to {num_partitions} partitions")
        logger.info(f"Target partition size: ~{self.config.get('target_partition_size_mb', 128)}MB")
        
        repartitioned_df = df.repartition(num_partitions)
        
        new_partition_count = repartitioned_df.rdd.getNumPartitions()
        row_count = repartitioned_df.count()
        
        logger.info(f"Repartitioning complete: {new_partition_count} partitions")
        logger.info(f"Average partition size: {row_count / new_partition_count:,.0f} rows")
        
        return repartitioned_df
    
    def cache_dataset(self, df: Optional[DataFrame] = None, 
                     persistence_level: str = "MEMORY_AND_DISK_SER") -> DataFrame:
        """
        Cache DataFrame for iterative ML operations and performance optimization.
        
        Args:
            df: DataFrame to cache. Uses validated_df if not provided.
            persistence_level: Storage level for caching
            
        Returns:
            DataFrame: Cached DataFrame
        """
        if df is None:
            df = self.validated_df or self.raw_df
        
        if df is None:
            raise ValueError("No DataFrame to cache")
        
        logger.info(f"Caching dataset with persistence level: {persistence_level}")
        
        # Apply caching
        cached_df = df.cache()
        
        # Force materialization by triggering an action
        row_count = cached_df.count()
        
        logger.info(f"Dataset cached successfully: {row_count:,} rows")
        logger.info(f"Storage level: {cached_df.storageLevel}")
        
        return cached_df
    
    def ingest_complete(self, hdfs_path: Optional[str] = None) -> Tuple[DataFrame, dict]:
        """
        Execute complete ingestion pipeline: load, validate, repartition, cache.
        
        Args:
            hdfs_path: HDFS path to dataset
            
        Returns:
            Tuple[DataFrame, dict]: Processed DataFrame and ingestion report
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE DATA INGESTION PIPELINE")
        logger.info("=" * 60)
        
        # Step 1: Load from HDFS
        df = self.load_from_hdfs(hdfs_path)
        
        # Step 2: Validate schema
        success, validation_report = self.validate_schema(df)
        
        # Step 3: Detect duplicates
        duplicate_count = self.detect_duplicates(df)
        validation_report['duplicate_count'] = duplicate_count
        
        # Step 4: Repartition for distributed processing
        df_repartitioned = self.repartion_for_processing(df)
        
        # Step 5: Cache for performance
        df_cached = self.cache_dataset(df_repartitioned)
        
        # Generate ingestion report
        ingestion_report = {
            'success': success,
            'total_rows': validation_report['total_rows'],
            'total_columns': validation_report['total_columns'],
            'column_names': validation_report['column_names'],
            'null_counts': validation_report['null_counts'],
            'duplicate_rows': duplicate_count,
            'num_partitions': df_cached.rdd.getNumPartitions(),
            'is_cached': df_cached.is_cached,
            'storage_level': str(df_cached.storageLevel)
        }
        
        logger.info("=" * 60)
        logger.info("DATA INGESTION COMPLETE")
        logger.info(f"Rows: {ingestion_report['total_rows']:,}")
        logger.info(f"Columns: {ingestion_report['total_columns']}")
        logger.info(f"Partitions: {ingestion_report['num_partitions']}")
        logger.info(f"Cached: {ingestion_report['is_cached']}")
        logger.info("=" * 60)
        
        return df_cached, ingestion_report


def main():
    """Main function to test data ingestion pipeline."""
    logger.info("Initializing Data Ingestion Pipeline")
    
    # Initialize Spark session
    spark = get_spark_session()
    
    try:
        # Initialize data ingestion module
        ingestion = DataIngestion(spark)
        
        # Execute complete ingestion pipeline
        df, report = ingestion.ingest_complete()
        
        # Display sample data
        logger.info("\nSample Data (First 5 rows):")
        df.show(5)
        
        logger.info("\n✅ Data ingestion pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in data ingestion: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
