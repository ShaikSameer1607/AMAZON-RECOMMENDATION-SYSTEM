"""
Feature Engineering Module

Creates and transforms features required for the recommendation system
including user activity scores, product popularity, and interaction patterns.
"""

import logging
from typing import Tuple, Optional, Dict
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, count, mean, stddev, sum as sum_col, \
    countDistinct, when, rank, dense_rank, row_number, broadcast
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


class FeatureEngineer:
    """
    Handles feature engineering for collaborative filtering recommendation system.
    Creates user-based, item-based, and interaction-based features.
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize feature engineer.
        
        Args:
            spark: Active Spark session
        """
        self.spark = spark
        self.feature_stats: Dict = {}
    
    def compute_user_activity_features(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Compute user activity features including engagement metrics.
        
        Features created:
        - user_activity_score: Total number of ratings by user
        - user_avg_rating: Average rating given by user
        - user_rating_stddev: Standard deviation of user's ratings
        - user_unique_products: Number of unique products rated
        
        Args:
            df: Input DataFrame with user_id, product_id, rating
            
        Returns:
            Tuple[DataFrame, DataFrame]: Original DF + user features DF
        """
        logger.info("Computing user activity features...")
        
        user_features = df.groupBy('user_id') \
            .agg(
                count('rating').alias('user_activity_score'),
                round(mean('rating'), 4).alias('user_avg_rating'),
                round(stddev('rating'), 4).alias('user_rating_stddev'),
                countDistinct('product_id').alias('user_unique_products')
            )
        
        # Fill null stddev values (users with only 1 rating)
        user_features = user_features.fillna({'user_rating_stddev': 0.0})
        
        logger.info(f"Computed features for {user_features.count():,} users")
        logger.info("User Activity Features Sample:")
        user_features.show(5)
        
        return df, user_features
    
    def compute_product_popularity_features(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Compute product popularity features.
        
        Features created:
        - product_popularity: Total number of ratings for product
        - product_avg_rating: Average rating of product
        - product_rating_stddev: Standard deviation of product ratings
        - product_min_rating: Minimum rating received
        - product_max_rating: Maximum rating received
        
        Args:
            df: Input DataFrame with user_id, product_id, rating
            
        Returns:
            Tuple[DataFrame, DataFrame]: Original DF + product features DF
        """
        logger.info("Computing product popularity features...")
        
        product_features = df.groupBy('product_id') \
            .agg(
                count('rating').alias('product_popularity'),
                round(mean('rating'), 4).alias('product_avg_rating'),
                round(stddev('rating'), 4).alias('product_rating_stddev'),
                min(col('rating')).alias('product_min_rating'),
                max(col('rating')).alias('product_max_rating')
            )
        
        # Fill null stddev values
        product_features = product_features.fillna({'product_rating_stddev': 0.0})
        
        logger.info(f"Computed features for {product_features.count():,} products")
        logger.info("Product Popularity Features Sample:")
        product_features.show(5)
        
        return df, product_features
    
    def add_user_features_to_dataframe(self, df: DataFrame, 
                                       user_features: DataFrame) -> DataFrame:
        """
        Join user features back to main DataFrame.
        
        Args:
            df: Original DataFrame
            user_features: User feature DataFrame
            
        Returns:
            DataFrame: DataFrame with user features added
        """
        logger.info("Adding user features to main DataFrame...")
        
        # Use broadcast join for small user_features table
        df_enriched = df.join(
            broadcast(user_features),
            on='user_id',
            how='left'
        )
        
        logger.info(f"Enriched DataFrame shape: {df_enriched.count():,} rows")
        return df_enriched
    
    def add_product_features_to_dataframe(self, df: DataFrame, 
                                          product_features: DataFrame) -> DataFrame:
        """
        Join product features back to main DataFrame.
        
        Args:
            df: Original DataFrame
            product_features: Product feature DataFrame
            
        Returns:
            DataFrame: DataFrame with product features added
        """
        logger.info("Adding product features to main DataFrame...")
        
        # Use broadcast join for small product_features table
        df_enriched = df.join(
            broadcast(product_features),
            on='product_id',
            how='left'
        )
        
        logger.info(f"Enriched DataFrame shape: {df_enriched.count():,} rows")
        return df_enriched
    
    def compute_interaction_frequency(self, df: DataFrame) -> DataFrame:
        """
        Compute interaction frequency features.
        
        This creates a normalized score representing how frequently a user
        interacts with products relative to other users.
        
        Args:
            df: Input DataFrame with user features
            
        Returns:
            DataFrame: DataFrame with interaction frequency
        """
        logger.info("Computing interaction frequency features...")
        
        if 'user_activity_score' not in df.columns:
            logger.warning("user_activity_score not found. Computing first...")
            _, user_features = self.compute_user_activity_features(df)
            df = self.add_user_features_to_dataframe(df, user_features)
        
        # Compute global statistics for normalization
        stats = df.agg(
            mean('user_activity_score').alias('mean_activity'),
            stddev('user_activity_score').alias('stddev_activity')
        ).collect()[0]
        
        mean_activity = stats['mean_activity'] or 0
        stddev_activity = stats['stddev_activity'] or 1
        
        # Z-score normalization for interaction frequency
        df_with_freq = df.withColumn(
            'interaction_frequency',
            (col('user_activity_score') - mean_activity) / max(stddev_activity, 1)
        )
        
        logger.info("Interaction frequency computed (Z-score normalized)")
        return df_with_freq
    
    def normalize_ratings(self, df: DataFrame, method: str = 'zscore') -> DataFrame:
        """
        Normalize ratings using z-score or min-max normalization.
        
        Args:
            df: Input DataFrame
            method: Normalization method ('zscore' or 'minmax')
            
        Returns:
            DataFrame: DataFrame with normalized ratings
        """
        logger.info(f"Normalizing ratings using {method} method...")
        
        if method == 'zscore':
            # Z-score normalization
            stats = df.agg(
                mean('rating').alias('mean_rating'),
                stddev('rating').alias('stddev_rating')
            ).collect()[0]
            
            mean_rating = stats['mean_rating'] or 0
            stddev_rating = stats['stddev_rating'] or 1
            
            df_normalized = df.withColumn(
                'normalized_rating',
                (col('rating') - mean_rating) / max(stddev_rating, 1)
            )
            
            logger.info(f"Z-score normalization: mean={mean_rating:.4f}, std={stddev_rating:.4f}")
            
        elif method == 'minmax':
            # Min-max normalization to [0, 1]
            min_max = df.agg(
                min(col('rating')).alias('min_rating'),
                max(col('rating')).alias('max_rating')
            ).collect()[0]
            
            min_rating = min_max['min_rating'] or 0
            max_rating = max_max['max_rating'] or 5
            
            df_normalized = df.withColumn(
                'normalized_rating',
                (col('rating') - min_rating) / max((max_rating - min_rating), 1)
            )
            
            logger.info(f"Min-Max normalization: min={min_rating}, max={max_rating}")
        else:
            logger.warning(f"Unknown normalization method: {method}. Skipping.")
            return df
        
        return df_normalized
    
    def create_user_product_matrix_features(self, df: DataFrame) -> DataFrame:
        """
        Create features based on user-product interaction matrix properties.
        
        Features:
        - user_rank: Rank of user by activity
        - product_rank: Rank of product by popularity
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: DataFrame with rank features
        """
        logger.info("Creating user-product matrix rank features...")
        
        # User activity rank
        user_window = Window.orderBy(col('user_activity_score').desc())
        df_ranked = df.withColumn('user_rank', dense_rank().over(user_window))
        
        # Product popularity rank
        product_window = Window.orderBy(col('product_popularity').desc())
        df_ranked = df_ranked.withColumn('product_rank', dense_rank().over(product_window))
        
        logger.info("Rank features created successfully")
        return df_ranked
    
    def prepare_als_training_data(self, df: DataFrame) -> DataFrame:
        """
        Prepare final training data for ALS model.
        
        Ensures columns are properly typed and ordered for ALS:
        - user_id: Integer
        - product_id: Integer
        - rating: Float
        
        Args:
            df: Input DataFrame with all features
            
        Returns:
            DataFrame: Cleaned DataFrame ready for ALS training
        """
        logger.info("Preparing ALS training data...")
        
        # Select and order essential columns
        als_df = df.select(
            col('user_id').cast('int').alias('user_id'),
            col('product_id').cast('int').alias('product_id'),
            col('rating').cast('float').alias('rating')
        )
        
        # Cache for performance
        als_df.cache()
        
        # Force materialization
        count = als_df.count()
        logger.info(f"ALS training data prepared: {count:,} samples")
        logger.info("Schema:")
        als_df.printSchema()
        
        return als_df
    
    def engineer_features_complete(self, df: DataFrame) -> Tuple[DataFrame, Dict]:
        """
        Execute complete feature engineering pipeline.
        
        Pipeline:
        1. Compute user activity features
        2. Compute product popularity features
        3. Join features to main DataFrame
        4. Compute interaction frequency
        5. Create rank features
        6. Prepare ALS training data
        
        Args:
            df: Preprocessed input DataFrame
            
        Returns:
            Tuple[DataFrame, Dict]: Feature-enriched DataFrame and feature report
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 60)
        
        feature_report = {
            'initial_row_count': df.count(),
            'features_created': [],
            'statistics': {}
        }
        
        # Step 1: Compute user activity features
        df, user_features = self.compute_user_activity_features(df)
        feature_report['features_created'].append('user_activity_features')
        feature_report['num_users'] = user_features.count()
        
        # Step 2: Compute product popularity features
        df, product_features = self.compute_product_popularity_features(df)
        feature_report['features_created'].append('product_popularity_features')
        feature_report['num_products'] = product_features.count()
        
        # Step 3: Join features to main DataFrame
        df = self.add_user_features_to_dataframe(df, user_features)
        df = self.add_product_features_to_dataframe(df, product_features)
        feature_report['features_created'].append('joined_user_product_features')
        
        # Step 4: Compute interaction frequency
        df = self.compute_interaction_frequency(df)
        feature_report['features_created'].append('interaction_frequency')
        
        # Step 5: Create rank features
        df = self.create_user_product_matrix_features(df)
        feature_report['features_created'].append('rank_features')
        
        # Step 6: Prepare ALS training data
        als_df = self.prepare_als_training_data(df)
        
        # Store statistics
        feature_report['final_row_count'] = als_df.count()
        feature_report['feature_columns'] = als_df.columns
        feature_report['statistics']['user_features'] = user_features
        feature_report['statistics']['product_features'] = product_features
        
        logger.info("=" * 60)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info(f"Features created: {feature_report['features_created']}")
        logger.info(f"Final dataset: {feature_report['final_row_count']:,} rows")
        logger.info("=" * 60)
        
        return als_df, feature_report


def main():
    """Main function to test feature engineering pipeline."""
    logger.info("Initializing Feature Engineering Pipeline")
    
    # Initialize Spark session
    spark = get_spark_session()
    
    try:
        # Import modules for testing
        from spark_pipeline.data_ingestion import DataIngestion
        from spark_pipeline.data_preprocessing import DataPreprocessor
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data for feature engineering test...")
        ingestion = DataIngestion(spark)
        df_raw, _ = ingestion.ingest_complete()
        
        preprocessor = DataPreprocessor(spark)
        df_cleaned, _ = preprocessor.preprocess_complete(df_raw)
        
        # Run feature engineering
        feature_engineer = FeatureEngineer(spark)
        df_features, report = feature_engineer.engineer_features_complete(df_cleaned)
        
        logger.info("\n✅ Feature engineering completed successfully!")
        logger.info(f"\nFeature Summary:")
        logger.info(f"  - Users: {report['num_users']:,}")
        logger.info(f"  - Products: {report['num_products']:,}")
        logger.info(f"  - Total Ratings: {report['final_row_count']:,}")
        logger.info(f"  - Features Created: {report['features_created']}")
        
        # Display sample with features
        logger.info("\nSample Data with Features:")
        df_features.show(5)
        
    except Exception as e:
        logger.error(f"❌ Error in feature engineering: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
