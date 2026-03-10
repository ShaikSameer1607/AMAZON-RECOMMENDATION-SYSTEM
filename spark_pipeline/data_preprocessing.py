"""
Data Preprocessing Module

Handles data cleaning, deduplication, type conversion, and filtering
for the Amazon ratings dataset.
"""

import logging
from typing import Tuple, Optional, List
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, count, when, isnan, isnull, row_number
from pyspark.sql.window import Window
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


class DataPreprocessor:
    """
    Handles comprehensive data preprocessing for recommendation system.
    Implements cleaning, deduplication, type conversion, and validation.
    """
    
    def __init__(self, spark: SparkSession, config: Optional[dict] = None):
        """
        Initialize data preprocessor.
        
        Args:
            spark: Active Spark session
            config: Configuration dictionary (optional)
        """
        self.spark = spark
        self.config = config or self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Return default preprocessing configuration."""
        return {
            'required_columns': ['user_id', 'product_id', 'rating'],
            'rating_min': 0.5,
            'rating_max': 5.0,
            'handle_nulls_strategy': 'drop',  # 'drop' or 'impute'
            'remove_duplicates': True,
            'convert_types': True,
        }
    
    def select_required_columns(self, df: DataFrame, 
                                columns: Optional[List[str]] = None) -> DataFrame:
        """
        Select only required columns for the recommendation system.
        
        Args:
            df: Input DataFrame
            columns: List of column names to select
            
        Returns:
            DataFrame: DataFrame with selected columns
        """
        columns = columns or self.config['required_columns']
        
        logger.info(f"Selecting required columns: {columns}")
        
        # Check if all required columns exist
        available_columns = df.columns
        missing_columns = [col for col in columns if col not in available_columns]
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            # Use available columns only
            columns = [col for col in columns if col in available_columns]
        
        # Select columns (add timestamp if available)
        if 'timestamp' in available_columns and 'timestamp' not in columns:
            columns.append('timestamp')
            logger.info("Including 'timestamp' column")
        
        selected_df = df.select(*columns)
        
        logger.info(f"Selected {len(columns)} columns: {columns}")
        logger.info(f"DataFrame shape: {selected_df.count():,} rows x {len(columns)} columns")
        
        return selected_df
    
    def handle_null_values(self, df: DataFrame, 
                          strategy: Optional[str] = None) -> Tuple[DataFrame, dict]:
        """
        Handle null values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling nulls ('drop' or 'impute')
            
        Returns:
            Tuple[DataFrame, dict]: Cleaned DataFrame and null handling report
        """
        strategy = strategy or self.config.get('handle_nulls_strategy', 'drop')
        
        logger.info(f"Handling null values with strategy: {strategy}")
        
        original_count = df.count()
        null_report = {
            'original_row_count': original_count,
            'strategy': strategy,
            'null_counts_before': {},
            'rows_affected': 0,
            'final_row_count': 0
        }
        
        # Count nulls per column
        for col_name in df.columns:
            null_count = df.filter(col(col_name).isNull()).count()
            null_report['null_counts_before'][col_name] = null_count
            if null_count > 0:
                logger.warning(f"Column '{col_name}' has {null_count:,} null values")
        
        if strategy == 'drop':
            # Drop rows with any null values
            cleaned_df = df.dropna()
            final_count = cleaned_df.count()
            null_report['rows_affected'] = original_count - final_count
            
            logger.info(f"Dropped {null_report['rows_affected']:,} rows with null values")
            logger.info(f"Remaining rows: {final_count:,}")
            
        elif strategy == 'impute':
            logger.info("Imputation strategy selected")
            # For recommendation systems, we typically drop nulls in rating columns
            # as imputation can introduce bias
            cleaned_df = df.dropna(subset=['user_id', 'product_id', 'rating'])
            final_count = cleaned_df.count()
            null_report['rows_affected'] = original_count - final_count
            
            logger.info(f"Dropped {null_report['rows_affected']:,} rows with nulls in key columns")
            logger.info(f"Remaining rows: {final_count:,}")
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'drop' or 'impute'.")
        
        null_report['final_row_count'] = final_count
        
        return cleaned_df, null_report
    
    def remove_duplicates(self, df: DataFrame) -> Tuple[DataFrame, int]:
        """
        Remove duplicate rows from the dataset.
        
        For recommendation systems, we keep the latest rating if a user has
        rated the same product multiple times.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple[DataFrame, int]: Deduplicated DataFrame and duplicate count
        """
        logger.info("Removing duplicate rows...")
        
        original_count = df.count()
        
        # Check if timestamp column exists for keeping latest ratings
        if 'timestamp' in df.columns:
            logger.info("Using timestamp to keep latest ratings for duplicates")
            
            # Define window specification
            window_spec = Window.partitionBy('user_id', 'product_id').orderBy(col('timestamp').desc())
            
            # Keep only the latest rating per user-product pair
            deduplicated_df = df.withColumn('row_num', row_number().over(window_spec)) \
                                .filter(col('row_num') == 1) \
                                .drop('row_num')
        else:
            logger.info("No timestamp column found, removing exact duplicates")
            deduplicated_df = df.dropDuplicates()
        
        final_count = deduplicated_df.count()
        duplicate_count = original_count - final_count
        
        logger.info(f"Original rows: {original_count:,}")
        logger.info(f"After deduplication: {final_count:,}")
        logger.info(f"Duplicates removed: {duplicate_count:,} ({(duplicate_count/original_count)*100:.2f}%)")
        
        return deduplicated_df, duplicate_count
    
    def convert_data_types(self, df: DataFrame) -> DataFrame:
        """
        Convert columns to optimal data types for ML processing.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: DataFrame with converted types
        """
        logger.info("Converting data types for ML optimization...")
        
        converted_df = df
        
        # Convert user_id and product_id to IntegerType if they're strings
        for col_name in ['user_id', 'product_id']:
            if col_name in df.columns:
                current_type = df.schema[col_name].dataType.simpleString()
                
                if current_type in ['string', 'bigint']:
                    logger.info(f"Converting {col_name} from {current_type} to integer")
                    try:
                        converted_df = converted_df.withColumn(col_name, col(col_name).cast('int'))
                    except Exception as e:
                        logger.warning(f"Could not convert {col_name}: {e}")
        
        # Ensure rating is FloatType
        if 'rating' in df.columns:
            current_type = df.schema['rating'].dataType.simpleString()
            if current_type != 'float':
                logger.info(f"Converting rating from {current_type} to float")
                converted_df = converted_df.withColumn('rating', col('rating').cast('float'))
        
        # Log final schema
        logger.info("Final schema after type conversion:")
        converted_df.printSchema()
        
        return converted_df
    
    def filter_invalid_ratings(self, df: DataFrame, 
                               min_rating: Optional[float] = None,
                               max_rating: Optional[float] = None) -> Tuple[DataFrame, int]:
        """
        Filter out invalid ratings outside the valid range.
        
        Args:
            df: Input DataFrame
            min_rating: Minimum valid rating (default: 0.5)
            max_rating: Maximum valid rating (default: 5.0)
            
        Returns:
            Tuple[DataFrame, int]: Filtered DataFrame and count of filtered rows
        """
        min_rating = min_rating or self.config.get('rating_min', 0.5)
        max_rating = max_rating or self.config.get('rating_max', 5.0)
        
        logger.info(f"Filtering ratings outside range [{min_rating}, {max_rating}]")
        
        original_count = df.count()
        
        # Filter ratings within valid range
        filtered_df = df.filter(
            (col('rating') >= min_rating) & 
            (col('rating') <= max_rating)
        )
        
        final_count = filtered_df.count()
        filtered_count = original_count - final_count
        
        logger.info(f"Original rows: {original_count:,}")
        logger.info(f"After filtering: {final_count:,}")
        logger.info(f"Invalid ratings filtered: {filtered_count:,} ({(filtered_count/original_count)*100:.2f}%)")
        
        return filtered_df, filtered_count
    
    def preprocess_complete(self, df: DataFrame) -> Tuple[DataFrame, dict]:
        """
        Execute complete preprocessing pipeline.
        
        Pipeline steps:
        1. Select required columns
        2. Handle null values
        3. Remove duplicates
        4. Convert data types
        5. Filter invalid ratings
        
        Args:
            df: Input raw DataFrame
            
        Returns:
            Tuple[DataFrame, dict]: Preprocessed DataFrame and preprocessing report
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE DATA PREPROCESSING PIPELINE")
        logger.info("=" * 60)
        
        preprocessing_report = {
            'initial_row_count': df.count(),
            'steps_completed': [],
            'row_counts': {}
        }
        
        # Step 1: Select required columns
        df = self.select_required_columns(df)
        preprocessing_report['steps_completed'].append('column_selection')
        preprocessing_report['row_counts']['after_column_selection'] = df.count()
        
        # Step 2: Handle null values
        df, null_report = self.handle_null_values(df)
        preprocessing_report['steps_completed'].append('null_handling')
        preprocessing_report['null_report'] = null_report
        preprocessing_report['row_counts']['after_null_handling'] = df.count()
        
        # Step 3: Remove duplicates
        df, duplicate_count = self.remove_duplicates(df)
        preprocessing_report['steps_completed'].append('deduplication')
        preprocessing_report['duplicate_count'] = duplicate_count
        preprocessing_report['row_counts']['after_deduplication'] = df.count()
        
        # Step 4: Convert data types
        df = self.convert_data_types(df)
        preprocessing_report['steps_completed'].append('type_conversion')
        preprocessing_report['row_counts']['after_type_conversion'] = df.count()
        
        # Step 5: Filter invalid ratings
        df, filtered_count = self.filter_invalid_ratings(df)
        preprocessing_report['steps_completed'].append('rating_filtering')
        preprocessing_report['invalid_ratings_filtered'] = filtered_count
        preprocessing_report['row_counts']['after_rating_filtering'] = df.count()
        
        # Final statistics
        preprocessing_report['final_row_count'] = df.count()
        preprocessing_report['total_rows_removed'] = preprocessing_report['initial_row_count'] - preprocessing_report['final_row_count']
        preprocessing_report['removal_percentage'] = round(
            (preprocessing_report['total_rows_removed'] / preprocessing_report['initial_row_count']) * 100, 2
        )
        
        logger.info("=" * 60)
        logger.info("DATA PREPROCESSING COMPLETE")
        logger.info(f"Initial rows: {preprocessing_report['initial_row_count']:,}")
        logger.info(f"Final rows: {preprocessing_report['final_row_count']:,}")
        logger.info(f"Total rows removed: {preprocessing_report['total_rows_removed']:,} ({preprocessing_report['removal_percentage']}%)")
        logger.info("=" * 60)
        
        return df, preprocessing_report


def main():
    """Main function to test data preprocessing pipeline."""
    logger.info("Initializing Data Preprocessing Pipeline")
    
    # Initialize Spark session
    spark = get_spark_session()
    
    try:
        # Import data ingestion for testing
        from spark_pipeline.data_ingestion import DataIngestion
        
        # Initialize ingestion and preprocessing modules
        ingestion = DataIngestion(spark)
        preprocessor = DataPreprocessor(spark)
        
        # Load and ingest data
        logger.info("Loading data for preprocessing test...")
        df_raw, _ = ingestion.ingest_complete()
        
        # Execute complete preprocessing pipeline
        df_cleaned, report = preprocessor.preprocess_complete(df_raw)
        
        # Display sample data
        logger.info("\nSample Cleaned Data (First 5 rows):")
        df_cleaned.show(5)
        
        logger.info("\n✅ Data preprocessing pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error in data preprocessing: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
