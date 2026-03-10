"""
Exploratory Data Analysis Module

Performs distributed EDA computations on the Amazon ratings dataset
using Spark transformations and aggregations.
"""

import logging
from typing import Dict, Optional, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, count, mean, stddev, min as min_col, max as max_col, \
    countDistinct, desc, asc, grouping_id
import sys
import os
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spark_pipeline.simple_spark_builder import get_spark_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExploratoryDataAnalysis:
    """
    Performs comprehensive exploratory data analysis using Spark's
    distributed computing capabilities.
    """
    
    def __init__(self, spark: SparkSession):
        """
        Initialize EDA module.
        
        Args:
            spark: Active Spark session
        """
        self.spark = spark
        self.eda_results: Dict[str, Any] = {}
    
    def compute_basic_statistics(self, df: DataFrame) -> Dict[str, Any]:
        """
        Compute basic statistics for the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            dict: Basic statistics
        """
        logger.info("Computing basic statistics...")
        
        stats = {
            'total_users': df.select(countDistinct('user_id')).collect()[0][0],
            'total_products': df.select(countDistinct('product_id')).collect()[0][0],
            'total_ratings': df.count(),
            'avg_ratings_per_user': 0,
            'avg_ratings_per_product': 0,
            'sparsity': 0
        }
        
        # Calculate averages
        stats['avg_ratings_per_user'] = round(stats['total_ratings'] / max(stats['total_users'], 1), 2)
        stats['avg_ratings_per_product'] = round(stats['total_ratings'] / max(stats['total_products'], 1), 2)
        
        # Calculate matrix sparsity
        possible_ratings = stats['total_users'] * stats['total_products']
        stats['sparsity'] = round((1 - stats['total_ratings'] / max(possible_ratings, 1)) * 100, 4)
        
        logger.info("=" * 60)
        logger.info("BASIC STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total Users: {stats['total_users']:,}")
        logger.info(f"Total Products: {stats['total_products']:,}")
        logger.info(f"Total Ratings: {stats['total_ratings']:,}")
        logger.info(f"Avg Ratings per User: {stats['avg_ratings_per_user']}")
        logger.info(f"Avg Ratings per Product: {stats['avg_ratings_per_product']}")
        logger.info(f"Matrix Sparsity: {stats['sparsity']}%")
        logger.info("=" * 60)
        
        self.eda_results['basic_statistics'] = stats
        return stats
    
    def compute_rating_distribution(self, df: DataFrame) -> Dict[str, Any]:
        """
        Compute detailed rating distribution statistics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            dict: Rating distribution statistics
        """
        logger.info("Computing rating distribution...")
        
        # Compute rating statistics
        rating_stats = df.select(
            count('rating').alias('count'),
            mean('rating').alias('mean'),
            stddev('rating').alias('stddev'),
            min_col('rating').alias('min'),
            max_col('rating').alias('max')
        ).collect()[0]
        
        distribution = {
            'count': rating_stats['count'],
            'mean': round(rating_stats['mean'], 4) if rating_stats['mean'] else 0,
            'stddev': round(rating_stats['stddev'], 4) if rating_stats['stddev'] else 0,
            'min': rating_stats['min'],
            'max': rating_stats['max'],
            'median': self._compute_median(df, 'rating')
        }
        
        # Compute rating value counts (distribution by rating value)
        rating_counts = df.groupBy('rating').count().orderBy('rating').collect()
        distribution['rating_value_counts'] = {
            row['rating']: row['count'] for row in rating_counts
        }
        
        logger.info("=" * 60)
        logger.info("RATING DISTRIBUTION")
        logger.info("=" * 60)
        logger.info(f"Mean Rating: {distribution['mean']}")
        logger.info(f"Std Deviation: {distribution['stddev']}")
        logger.info(f"Min Rating: {distribution['min']}")
        logger.info(f"Max Rating: {distribution['max']}")
        logger.info(f"Median Rating: {distribution['median']}")
        logger.info("=" * 60)
        
        self.eda_results['rating_distribution'] = distribution
        return distribution
    
    def _compute_median(self, df: DataFrame, column: str) -> float:
        """
        Compute median of a column using approximate quantile.
        
        Args:
            df: Input DataFrame
            column: Column name
            
        Returns:
            float: Median value
        """
        try:
            median = df.approxQuantile(column, [0.5], 0.01)[0]
            return round(median, 4)
        except Exception as e:
            logger.warning(f"Could not compute median: {e}")
            return 0.0
    
    def get_top_rated_products(self, df: DataFrame, top_n: int = 20) -> DataFrame:
        """
        Get top N rated products by average rating and review count.
        
        Args:
            df: Input DataFrame
            top_n: Number of top products to return
            
        Returns:
            DataFrame: Top rated products
        """
        logger.info(f"Computing top {top_n} rated products...")
        
        top_products = df.groupBy('product_id') \
            .agg(
                count('rating').alias('review_count'),
                round(mean('rating'), 4).alias('avg_rating')
            ) \
            .filter(col('review_count') >= 5) \
            .orderBy(desc('avg_rating'), desc('review_count')) \
            .limit(top_n)
        
        logger.info(f"Top {top_n} products identified")
        top_products.show(truncate=False)
        
        self.eda_results['top_rated_products'] = top_products
        return top_products
    
    def get_most_active_users(self, df: DataFrame, top_n: int = 20) -> DataFrame:
        """
        Get top N most active users by number of ratings.
        
        Args:
            df: Input DataFrame
            top_n: Number of top users to return
            
        Returns:
            DataFrame: Most active users
        """
        logger.info(f"Computing top {top_n} most active users...")
        
        top_users = df.groupBy('user_id') \
            .agg(
                count('rating').alias('rating_count'),
                round(mean('rating'), 4).alias('avg_rating'),
                round(stddev('rating'), 4).alias('rating_stddev')
            ) \
            .orderBy(desc('rating_count')) \
            .limit(top_n)
        
        logger.info(f"Top {top_n} active users identified")
        top_users.show(truncate=False)
        
        self.eda_results['most_active_users'] = top_users
        return top_users
    
    def compute_product_popularity(self, df: DataFrame) -> DataFrame:
        """
        Compute product popularity metrics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: Product popularity metrics
        """
        logger.info("Computing product popularity metrics...")
        
        popularity_df = df.groupBy('product_id') \
            .agg(
                count('rating').alias('popularity_score'),
                round(mean('rating'), 4).alias('avg_rating'),
                round(stddev('rating'), 4).alias('rating_stddev'),
                min_col('rating').alias('min_rating'),
                max_col('rating').alias('max_rating')
            ) \
            .orderBy(desc('popularity_score'))
        
        logger.info(f"Computed popularity for {popularity_df.count():,} products")
        
        self.eda_results['product_popularity'] = popularity_df
        return popularity_df
    
    def compute_user_activity(self, df: DataFrame) -> DataFrame:
        """
        Compute user activity metrics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame: User activity metrics
        """
        logger.info("Computing user activity metrics...")
        
        activity_df = df.groupBy('user_id') \
            .agg(
                count('rating').alias('activity_score'),
                round(mean('rating'), 4).alias('avg_rating'),
                round(stddev('rating'), 4).alias('rating_stddev'),
                countDistinct('product_id').alias('unique_products')
            ) \
            .orderBy(desc('activity_score'))
        
        logger.info(f"Computed activity for {activity_df.count():,} users")
        
        self.eda_results['user_activity'] = activity_df
        return activity_df
    
    def analyze_temporal_patterns(self, df: DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze temporal patterns if timestamp column exists.
        
        Args:
            df: Input DataFrame
            
        Returns:
            dict: Temporal analysis results or None if no timestamp
        """
        if 'timestamp' not in df.columns:
            logger.info("No timestamp column found, skipping temporal analysis")
            return None
        
        logger.info("Analyzing temporal patterns...")
        
        from pyspark.sql.functions import year, month, from_unixtime
        
        try:
            # Extract year and month from timestamp
            df_with_time = df.withColumn('datetime', from_unixtime(col('timestamp'))) \
                             .withColumn('year', year('datetime')) \
                             .withColumn('month', month('datetime'))
            
            # Ratings per year
            ratings_per_year = df_with_time.groupBy('year') \
                .agg(count('rating').alias('rating_count')) \
                .orderBy('year') \
                .collect()
            
            temporal_stats = {
                'ratings_per_year': {row['year']: row['rating_count'] for row in ratings_per_year},
                'earliest_year': min(row['year'] for row in ratings_per_year) if ratings_per_year else None,
                'latest_year': max(row['year'] for row in ratings_per_year) if ratings_per_year else None
            }
            
            logger.info("=" * 60)
            logger.info("TEMPORAL PATTERNS")
            logger.info("=" * 60)
            logger.info(f"Time Range: {temporal_stats['earliest_year']} - {temporal_stats['latest_year']}")
            logger.info("Ratings per Year:")
            for year, count in temporal_stats['ratings_per_year'].items():
                logger.info(f"  {year}: {count:,}")
            logger.info("=" * 60)
            
            self.eda_results['temporal_patterns'] = temporal_stats
            return temporal_stats
            
        except Exception as e:
            logger.warning(f"Error in temporal analysis: {e}")
            return None
    
    def save_results_to_hdfs(self, output_path: str = "hdfs://master:9000/recommendation_results/eda_results"):
        """
        Save EDA results to HDFS in distributed format.
        
        Args:
            output_path: HDFS output path
        """
        logger.info(f"Saving EDA results to HDFS: {output_path}")
        
        try:
            # Save DataFrames to HDFS
            if 'top_rated_products' in self.eda_results:
                self.eda_results['top_rated_products'] \
                    .coalesce(1) \
                    .write.mode('overwrite') \
                    .csv(f"{output_path}/top_rated_products")
            
            if 'most_active_users' in self.eda_results:
                self.eda_results['most_active_users'] \
                    .coalesce(1) \
                    .write.mode('overwrite') \
                    .csv(f"{output_path}/most_active_users")
            
            if 'product_popularity' in self.eda_results:
                self.eda_results['product_popularity'] \
                    .write.mode('overwrite') \
                    .csv(f"{output_path}/product_popularity")
            
            if 'user_activity' in self.eda_results:
                self.eda_results['user_activity'] \
                    .write.mode('overwrite') \
                    .csv(f"{output_path}/user_activity")
            
            # Save summary statistics as JSON
            summary_stats = {
                'basic_statistics': self.eda_results.get('basic_statistics', {}),
                'rating_distribution': self.eda_results.get('rating_distribution', {})
            }
            
            # Note: For HDFS write, you'd use hadoop FileSystem API
            # For now, we'll log the summary
            logger.info("EDA Summary Statistics:")
            logger.info(json.dumps(summary_stats, indent=2))
            
            logger.info("EDA results saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving EDA results: {e}")
            raise
    
    def run_complete_eda(self, df: DataFrame, 
                        save_to_hdfs: bool = True) -> Dict[str, Any]:
        """
        Run complete exploratory data analysis pipeline.
        
        Args:
            df: Input DataFrame
            save_to_hdfs: Whether to save results to HDFS
            
        Returns:
            dict: Complete EDA results
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE EXPLORATORY DATA ANALYSIS")
        logger.info("=" * 60)
        
        # 1. Basic Statistics
        self.compute_basic_statistics(df)
        
        # 2. Rating Distribution
        self.compute_rating_distribution(df)
        
        # 3. Top Rated Products
        self.get_top_rated_products(df)
        
        # 4. Most Active Users
        self.get_most_active_users(df)
        
        # 5. Product Popularity
        self.compute_product_popularity(df)
        
        # 6. User Activity
        self.compute_user_activity(df)
        
        # 7. Temporal Patterns (if available)
        self.analyze_temporal_patterns(df)
        
        # 8. Save results
        if save_to_hdfs:
            self.save_results_to_hdfs()
        
        logger.info("=" * 60)
        logger.info("EXPLORATORY DATA ANALYSIS COMPLETE")
        logger.info("=" * 60)
        
        return self.eda_results


def main():
    """Main function to test EDA pipeline."""
    logger.info("Initializing Exploratory Data Analysis Pipeline")
    
    # Initialize Spark session
    spark = get_spark_session()
    
    try:
        # Import modules for testing
        from spark_pipeline.data_ingestion import DataIngestion
        from spark_pipeline.data_preprocessing import DataPreprocessor
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data for EDA test...")
        ingestion = DataIngestion(spark)
        df_raw, _ = ingestion.ingest_complete()
        
        preprocessor = DataPreprocessor(spark)
        df_cleaned, _ = preprocessor.preprocess_complete(df_raw)
        
        # Run EDA
        eda = ExploratoryDataAnalysis(spark)
        results = eda.run_complete_eda(df_cleaned, save_to_hdfs=True)
        
        logger.info("\n✅ Exploratory data analysis completed successfully!")
        logger.info(f"\nKey Insights:")
        logger.info(f"  - Total Users: {results['basic_statistics']['total_users']:,}")
        logger.info(f"  - Total Products: {results['basic_statistics']['total_products']:,}")
        logger.info(f"  - Mean Rating: {results['rating_distribution']['mean']}")
        logger.info(f"  - Matrix Sparsity: {results['basic_statistics']['sparsity']}%")
        
    except Exception as e:
        logger.error(f"❌ Error in EDA: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
