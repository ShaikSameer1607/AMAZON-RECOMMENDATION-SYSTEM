"""
Model Evaluation Module

Comprehensive evaluation of recommendation system using multiple metrics:
RMSE, Precision@K, Recall@K, and Coverage.
"""

import logging
from typing import Dict, Tuple, Optional, List
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, count, countDistinct, avg, rank
from pyspark.sql.window import Window
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


class ModelEvaluator:
    """
    Comprehensive evaluator for recommendation systems supporting
    multiple evaluation metrics and benchmarking.
    """
    
    def __init__(self, spark: SparkSession, k_value: int = 10):
        """
        Initialize model evaluator.
        
        Args:
            spark: Active Spark session
            k_value: K value for Precision@K and Recall@K (default: 10)
        """
        self.spark = spark
        self.k_value = k_value
        self.evaluation_results: Dict[str, float] = {}
    
    def compute_rmse(self, predictions: DataFrame) -> float:
        """
        Compute Root Mean Square Error (RMSE).
        
        Measures the average squared difference between predicted and actual ratings.
        Lower is better.
        
        Args:
            predictions: DataFrame with 'rating' and 'prediction' columns
            
        Returns:
            float: RMSE value
        """
        logger.info("Computing RMSE...")
        
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        )
        
        rmse = evaluator.evaluate(predictions)
        
        logger.info(f"RMSE: {rmse:.4f}")
        self.evaluation_results['rmse'] = round(rmse, 4)
        
        return rmse
    
    def compute_mae(self, predictions: DataFrame) -> float:
        """
        Compute Mean Absolute Error (MAE).
        
        Measures the average absolute difference between predicted and actual ratings.
        Lower is better.
        
        Args:
            predictions: DataFrame with 'rating' and 'prediction' columns
            
        Returns:
            float: MAE value
        """
        logger.info("Computing MAE...")
        
        evaluator = RegressionEvaluator(
            metricName="mae",
            labelCol="rating",
            predictionCol="prediction"
        )
        
        mae = evaluator.evaluate(predictions)
        
        logger.info(f"MAE: {mae:.4f}")
        self.evaluation_results['mae'] = round(mae, 4)
        
        return mae
    
    def compute_r2(self, predictions: DataFrame) -> float:
        """
        Compute R-squared (coefficient of determination).
        
        Measures how well predictions approximate actual values.
        Higher is better (max 1.0).
        
        Args:
            predictions: DataFrame with 'rating' and 'prediction' columns
            
        Returns:
            float: R-squared value
        """
        logger.info("Computing R²...")
        
        evaluator = RegressionEvaluator(
            metricName="r2",
            labelCol="rating",
            predictionCol="prediction"
        )
        
        r2 = evaluator.evaluate(predictions)
        
        logger.info(f"R²: {r2:.4f}")
        self.evaluation_results['r2'] = round(r2, 4)
        
        return r2
    
    def compute_precision_at_k(self, model: ALSModel, test_df: DataFrame,
                               user_products: DataFrame, k: Optional[int] = None) -> float:
        """
        Compute Precision@K for top-K recommendations.
        
        Precision@K measures the fraction of recommended items that are relevant.
        
        Args:
            model: Trained ALS model
            test_df: Test dataset
            user_products: DataFrame with all user-product pairs
            k: Number of recommendations to consider
            
        Returns:
            float: Precision@K value
        """
        k = k or self.k_value
        logger.info(f"Computing Precision@{k}...")
        
        try:
            # Get top-K recommendations for each user
            top_k_recs = self._get_top_k_recommendations(model, user_products, k)
            
            # Count relevant items in top-K (items user actually rated highly)
            # Consider rating >= 4.0 as "relevant"
            high_rated_test = test_df.filter(col('rating') >= 4.0)
            
            # Join recommendations with test data to find matches
            matches = top_k_recs.join(
                high_rated_test,
                on=['user_id', 'product_id'],
                how='inner'
            )
            
            match_count = matches.count()
            total_recs = top_k_recs.count()
            
            precision = match_count / max(total_recs, 1)
            
            logger.info(f"Precision@{k}: {precision:.4f}")
            logger.info(f"  - Total recommendations: {total_recs:,}")
            logger.info(f"  - Relevant matches: {match_count:,}")
            
            self.evaluation_results[f'precision_at_{k}'] = round(precision, 4)
            
            return precision
            
        except Exception as e:
            logger.error(f"Error computing Precision@K: {e}")
            return 0.0
    
    def compute_recall_at_k(self, model: ALSModel, test_df: DataFrame,
                           user_products: DataFrame, k: Optional[int] = None) -> float:
        """
        Compute Recall@K for top-K recommendations.
        
        Recall@K measures the fraction of relevant items that are recommended.
        
        Args:
            model: Trained ALS model
            test_df: Test dataset
            user_products: DataFrame with all user-product pairs
            k: Number of recommendations to consider
            
        Returns:
            float: Recall@K value
        """
        k = k or self.k_value
        logger.info(f"Computing Recall@{k}...")
        
        try:
            # Get top-K recommendations for each user
            top_k_recs = self._get_top_k_recommendations(model, user_products, k)
            
            # Items user actually rated highly in test set
            high_rated_test = test_df.filter(col('rating') >= 4.0)
            
            # Find which relevant items were recommended
            matches = top_k_recs.join(
                high_rated_test,
                on=['user_id', 'product_id'],
                how='inner'
            )
            
            match_count = matches.count()
            total_relevant = high_rated_test.count()
            
            recall = match_count / max(total_relevant, 1)
            
            logger.info(f"Recall@{k}: {recall:.4f}")
            logger.info(f"  - Relevant items in test: {total_relevant:,}")
            logger.info(f"  - Retrieved relevant items: {match_count:,}")
            
            self.evaluation_results[f'recall_at_{k}'] = round(recall, 4)
            
            return recall
            
        except Exception as e:
            logger.error(f"Error computing Recall@K: {e}")
            return 0.0
    
    def _get_top_k_recommendations(self, model: ALSModel, 
                                   user_products: DataFrame, 
                                   k: int) -> DataFrame:
        """
        Get top-K recommendations for each user.
        
        Args:
            model: Trained ALS model
            user_products: DataFrame with user-product pairs
            k: Number of recommendations per user
            
        Returns:
            DataFrame: Top-K recommendations per user
        """
        # Generate all predictions
        predictions = model.transform(user_products)
        
        # Filter valid predictions and rank by prediction score
        window_spec = Window.partitionBy('user_id').orderBy(col('prediction').desc())
        
        top_k = predictions.filter(col('prediction').isNotNull()) \
            .withColumn('rank', rank().over(window_spec)) \
            .filter(col('rank') <= k) \
            .select('user_id', 'product_id', 'prediction')
        
        return top_k
    
    def compute_coverage(self, model: ALSModel, all_products: DataFrame,
                        user_products: DataFrame) -> float:
        """
        Compute catalog coverage.
        
        Measures the percentage of items that the model can recommend.
        Higher coverage indicates better exploration of item catalog.
        
        Args:
            model: Trained ALS model
            all_products: DataFrame with all products
            user_products: DataFrame with user-product pairs
            
        Returns:
            float: Coverage percentage
        """
        logger.info("Computing catalog coverage...")
        
        try:
            # Get all unique products in recommendations
            predictions = model.transform(user_products)
            recommended_products = predictions.select('product_id').distinct()
            
            total_products = all_products.select('product_id').distinct().count()
            recommended_count = recommended_products.count()
            
            coverage = (recommended_count / max(total_products, 1)) * 100
            
            logger.info(f"Coverage: {coverage:.2f}%")
            logger.info(f"  - Total products: {total_products:,}")
            logger.info(f"  - Recommended products: {recommended_count:,}")
            
            self.evaluation_results['coverage_percent'] = round(coverage, 2)
            
            return coverage
            
        except Exception as e:
            logger.error(f"Error computing coverage: {e}")
            return 0.0
    
    def compute_diversity(self, recommendations: DataFrame) -> float:
        """
        Compute recommendation diversity (intra-list similarity).
        
        Measures how diverse the recommendations are for each user.
        Higher diversity indicates more varied recommendations.
        
        Note: Simplified implementation using product popularity variance.
        
        Args:
            recommendations: Recommendations DataFrame
            
        Returns:
            float: Diversity score
        """
        logger.info("Computing recommendation diversity...")
        
        try:
            # Use prediction standard deviation as proxy for diversity
            from pyspark.sql.functions import stddev
            
            diversity = recommendations.agg(stddev('prediction')).collect()[0][0]
            
            if diversity is None:
                diversity = 0.0
            
            logger.info(f"Diversity Score: {diversity:.4f}")
            self.evaluation_results['diversity'] = round(diversity, 4)
            
            return diversity
            
        except Exception as e:
            logger.error(f"Error computing diversity: {e}")
            return 0.0
    
    def generate_evaluation_report(self, model: ALSModel, 
                                  train_df: DataFrame,
                                  test_df: DataFrame,
                                  save_path: Optional[str] = None) -> Dict[str, any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model: Trained ALS model
            train_df: Training data
            test_df: Test data
            save_path: Path to save report (optional)
            
        Returns:
            dict: Complete evaluation report
        """
        logger.info("=" * 60)
        logger.info("GENERATING COMPREHENSIVE EVALUATION REPORT")
        logger.info("=" * 60)
        
        report = {
            'dataset_info': {
                'train_size': train_df.count(),
                'test_size': test_df.count()
            },
            'metrics': {}
        }
        
        # Generate predictions on test set
        logger.info("Generating predictions on test set...")
        predictions = model.transform(test_df)
        valid_predictions = predictions.filter(col('prediction').isNotNull())
        
        # Rating-based metrics
        report['metrics']['rmse'] = self.compute_rmse(valid_predictions)
        report['metrics']['mae'] = self.compute_mae(valid_predictions)
        report['metrics']['r2'] = self.compute_r2(valid_predictions)
        
        # Ranking-based metrics
        # Create user-product matrix for recommendations
        user_products = self._create_user_product_matrix(train_df, test_df)
        
        report['metrics']['precision_at_10'] = self.compute_precision_at_k(
            model, test_df, user_products, k=10
        )
        report['metrics']['recall_at_10'] = self.compute_recall_at_k(
            model, test_df, user_products, k=10
        )
        
        # Coverage
        all_products = train_df.select('product_id').distinct()
        report['metrics']['coverage'] = self.compute_coverage(
            model, all_products, user_products
        )
        
        # Diversity
        report['metrics']['diversity'] = self.compute_diversity(valid_predictions)
        
        # Summary
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        for metric, value in report['metrics'].items():
            logger.info(f"{metric.upper()}: {value}")
        logger.info("=" * 60)
        
        # Save report if path provided
        if save_path:
            self._save_report(report, save_path)
        
        return report
    
    def _create_user_product_matrix(self, train_df: DataFrame, 
                                   test_df: DataFrame) -> DataFrame:
        """
        Create user-product interaction matrix for evaluation.
        
        Args:
            train_df: Training data
            test_df: Test data
            
        Returns:
            DataFrame: User-product pairs
        """
        # Get all users and products
        users = train_df.select('user_id').distinct()
        products = train_df.select('product_id').distinct()
        
        # Create cross join (all possible pairs)
        user_products = users.crossJoin(products)
        
        # Remove already seen items
        seen_items = train_df.select('user_id', 'product_id')
        user_products = user_products.join(seen_items, on=['user_id', 'product_id'], how='left_anti')
        
        logger.info(f"Created user-product matrix: {user_products.count():,} pairs")
        
        return user_products
    
    def _save_report(self, report: Dict, save_path: str):
        """
        Save evaluation report to file.
        
        Args:
            report: Evaluation report dictionary
            save_path: Path to save report
        """
        try:
            # Convert to JSON-serializable format
            serializable_report = json.dumps(report, indent=2, default=str)
            
            logger.info(f"Saving evaluation report to: {save_path}")
            
            # For HDFS, you would use hadoop FileSystem API
            # For now, save locally
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path.replace('hdfs://master:9000/', '/home/sameer/BDA_PROJECT_NEW/amazon-recommendation-system/outputs/'), 'w') as f:
                f.write(serializable_report)
            
            logger.info("✅ Report saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving report: {e}")


def main():
    """Main function to test evaluation pipeline."""
    logger.info("Initializing Model Evaluation Pipeline")
    
    # Initialize Spark session
    spark = get_spark_session()
    
    try:
        # Import modules for testing
        from spark_pipeline.data_ingestion import DataIngestion
        from spark_pipeline.data_preprocessing import DataPreprocessor
        from spark_pipeline.feature_engineering import FeatureEngineer
        from spark_pipeline.recommendation_model import RecommendationModelTrainer
        
        # Load and prepare data
        logger.info("Loading and preparing data for evaluation...")
        ingestion = DataIngestion(spark)
        df_raw, _ = ingestion.ingest_complete()
        
        preprocessor = DataPreprocessor(spark)
        df_cleaned, _ = preprocessor.preprocess_complete(df_raw)
        
        feature_engineer = FeatureEngineer(spark)
        df_features, _ = feature_engineer.engineer_features_complete(df_cleaned)
        
        # Train model
        logger.info("Training model for evaluation...")
        trainer = RecommendationModelTrainer(spark)
        model, training_report = trainer.train_complete_pipeline(df_features)
        
        # Split for evaluation
        train_df, test_df = trainer.split_train_test(df_features)
        
        # Evaluate model
        evaluator = ModelEvaluator(spark, k_value=10)
        evaluation_report = evaluator.generate_evaluation_report(
            model=model,
            train_df=train_df,
            test_df=test_df,
            save_path="hdfs://master:9000/recommendation_results/evaluation_report.json"
        )
        
        logger.info("\n✅ Model evaluation completed successfully!")
        logger.info(f"\nFinal Metrics:")
        for metric, value in evaluation_report['metrics'].items():
            logger.info(f"  {metric.upper()}: {value}")
        
    except Exception as e:
        logger.error(f"❌ Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
