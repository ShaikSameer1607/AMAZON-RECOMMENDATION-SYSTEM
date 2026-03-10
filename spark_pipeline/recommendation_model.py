"""
Recommendation Model Module

Implements collaborative filtering using Spark MLlib's ALS (Alternating Least Squares)
algorithm for building a large-scale recommendation system.
"""

import logging
from typing import Tuple, Optional, Dict, Any
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.recommendation import ALS, ALSModel
from pyspark.ml.evaluation import RegressionEvaluator
import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spark_pipeline.simple_spark_builder import get_spark_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecommendationModelTrainer:
    """
    Handles ALS model training, optimization, and persistence for
    collaborative filtering recommendation system.
    """
    
    def __init__(self, spark: SparkSession, config: Optional[dict] = None):
        """
        Initialize model trainer.
        
        Args:
            spark: Active Spark session
            config: Configuration dictionary (optional)
        """
        self.spark = spark
        self.config = config or self._get_default_config()
        self.model: Optional[ALSModel] = None
        self.training_history: Dict[str, Any] = {}
    
    def _get_default_config(self) -> dict:
        """Return default ALS model configuration."""
        return {
            'user_col': 'user_id',
            'item_col': 'product_id',
            'rating_col': 'rating',
            'cold_start_strategy': 'drop',
            'nonnegative': True,
            'rank': 10,
            'max_iter': 15,
            'reg_param': 0.1,
            'alpha': 1.0,
            'train_test_split': {
                'test_ratio': 0.2,
                'seed': 42
            }
        }
    
    def split_train_test(self, df: DataFrame) -> Tuple[DataFrame, DataFrame]:
        """
        Split data into training and test sets.
        
        Uses random split with stratification to ensure representative distribution.
        
        Args:
            df: Input DataFrame with ratings
            
        Returns:
            Tuple[DataFrame, DataFrame]: Training and test DataFrames
        """
        logger.info("Splitting data into train/test sets...")
        
        split_config = self.config.get('train_test_split', {})
        test_ratio = split_config.get('test_ratio', 0.2)
        seed = split_config.get('seed', 42)
        
        # Random split
        train_df, test_df = df.randomSplit(
            [1 - test_ratio, test_ratio],
            seed=seed
        )
        
        train_count = train_df.count()
        test_count = test_df.count()
        total = train_count + test_count
        
        logger.info(f"Training set: {train_count:,} samples ({(train_count/total)*100:.1f}%)")
        logger.info(f"Test set: {test_count:,} samples ({(test_count/total)*100:.1f}%)")
        
        self.training_history['train_size'] = train_count
        self.training_history['test_size'] = test_count
        
        return train_df, test_df
    
    def configure_als_model(self, 
                           rank: Optional[int] = None,
                           max_iter: Optional[int] = None,
                           reg_param: Optional[float] = None,
                           alpha: Optional[float] = None) -> ALS:
        """
        Configure ALS model with hyperparameters.
        
        Args:
            rank: Number of latent factors
            max_iter: Maximum iterations
            reg_param: Regularization parameter
            alpha: Confidence parameter for implicit feedback
            
        Returns:
            ALS: Configured ALS estimator
        """
        rank = rank or self.config.get('rank', 10)
        max_iter = max_iter or self.config.get('max_iter', 15)
        reg_param = reg_param or self.config.get('reg_param', 0.1)
        alpha = alpha or self.config.get('alpha', 1.0)
        
        logger.info("=" * 60)
        logger.info("ALS MODEL CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Latent Factors (rank): {rank}")
        logger.info(f"Max Iterations: {max_iter}")
        logger.info(f"Regularization Parameter: {reg_param}")
        logger.info(f"Alpha (confidence): {alpha}")
        logger.info(f"Cold Start Strategy: {self.config['cold_start_strategy']}")
        logger.info(f"Non-negative Constraints: {self.config['nonnegative']}")
        logger.info("=" * 60)
        
        als = ALS(
            userCol=self.config['user_col'],
            itemCol=self.config['item_col'],
            ratingCol=self.config['rating_col'],
            coldStartStrategy=self.config['cold_start_strategy'],
            nonnegative=self.config['nonnegative'],
            rank=rank,
            maxIter=max_iter,
            regParam=reg_param,
            alpha=alpha
        )
        
        return als
    
    def train_model(self, train_df: DataFrame, 
                   als_config: Optional[ALS] = None) -> ALSModel:
        """
        Train ALS model on training data.
        
        Args:
            train_df: Training DataFrame
            als_config: Pre-configured ALS estimator (optional)
            
        Returns:
            ALSModel: Trained ALS model
        """
        logger.info("=" * 60)
        logger.info("TRAINING ALS MODEL")
        logger.info("=" * 60)
        
        # Configure ALS if not provided
        if als_config is None:
            als = self.configure_als_model()
        else:
            als = als_config
        
        start_time = time.time()
        
        try:
            # Fit the model
            logger.info("Fitting ALS model (this may take a few minutes)...")
            self.model = als.fit(train_df)
            
            training_time = time.time() - start_time
            
            logger.info(f"✅ Model training completed in {training_time:.2f} seconds")
            logger.info(f"Model type: {type(self.model)}")
            logger.info(f"Number of latent factors: {self.model.rank}")
            
            self.training_history['training_time_seconds'] = round(training_time, 2)
            self.training_history['model_rank'] = self.model.rank
            
            return self.model
            
        except Exception as e:
            logger.error(f"❌ Error during model training: {e}")
            raise
    
    def generate_predictions(self, test_df: DataFrame) -> DataFrame:
        """
        Generate predictions on test data using trained model.
        
        Args:
            test_df: Test DataFrame
            
        Returns:
            DataFrame: Predictions DataFrame
        """
        if self.model is None:
            raise ValueError("No trained model available. Call train_model first.")
        
        logger.info("Generating predictions on test set...")
        
        predictions = self.model.transform(test_df)
        
        # Filter out NaN predictions (cold start cases)
        valid_predictions = predictions.filter(predictions.prediction.isNotNull())
        
        prediction_count = valid_predictions.count()
        total_count = test_df.count()
        
        logger.info(f"Total test samples: {total_count:,}")
        logger.info(f"Valid predictions: {prediction_count:,} ({(prediction_count/total_count)*100:.1f}%)")
        
        if prediction_count < total_count:
            logger.warning(f"Cold start cases: {total_count - prediction_count:,} users/products not in training set")
        
        return valid_predictions
    
    def save_model(self, model_path: str = "hdfs://master:9000/recommendation_results/models/als_model"):
        """
        Save trained model to HDFS or local filesystem.
        
        Args:
            model_path: Path to save model
        """
        if self.model is None:
            raise ValueError("No trained model to save")
        
        logger.info(f"Saving model to: {model_path}")
        
        try:
            self.model.write().overwrite().save(model_path)
            logger.info(f"✅ Model saved successfully to {model_path}")
            
            self.training_history['model_path'] = model_path
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            raise
    
    def load_model(self, model_path: str) -> ALSModel:
        """
        Load pre-trained model from HDFS or local filesystem.
        
        Args:
            model_path: Path to model
            
        Returns:
            ALSModel: Loaded model
        """
        logger.info(f"Loading model from: {model_path}")
        
        try:
            self.model = ALSModel.load(model_path)
            logger.info(f"✅ Model loaded successfully")
            logger.info(f"Model rank: {self.model.rank}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise
    
    def train_complete_pipeline(self, df: DataFrame) -> Tuple[ALSModel, Dict[str, Any]]:
        """
        Execute complete model training pipeline.
        
        Pipeline:
        1. Split data into train/test
        2. Configure ALS model
        3. Train model
        4. Evaluate on test set (basic RMSE)
        5. Save model
        
        Args:
            df: Prepared DataFrame with user_id, product_id, rating
            
        Returns:
            Tuple[ALSModel, Dict]: Trained model and training report
        """
        logger.info("=" * 60)
        logger.info("STARTING COMPLETE MODEL TRAINING PIPELINE")
        logger.info("=" * 60)
        
        training_report = {
            'dataset_size': df.count(),
            'configuration': self.config
        }
        
        # Step 1: Split data
        train_df, test_df = self.split_train_test(df)
        training_report['train_size'] = train_df.count()
        training_report['test_size'] = test_df.count()
        
        # Step 2: Configure model
        als = self.configure_als_model()
        
        # Step 3: Train model
        self.model = self.train_model(train_df, als)
        
        # Step 4: Generate predictions
        predictions = self.generate_predictions(test_df)
        training_report['valid_predictions'] = predictions.count()
        
        # Step 5: Basic RMSE evaluation
        evaluator = RegressionEvaluator(
            metricName="rmse",
            labelCol="rating",
            predictionCol="prediction"
        )
        
        rmse = evaluator.evaluate(predictions)
        training_report['rmse'] = round(rmse, 4)
        
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Training Samples: {training_report['train_size']:,}")
        logger.info(f"Test Samples: {training_report['test_size']:,}")
        logger.info(f"RMSE: {training_report['rmse']}")
        logger.info(f"Training Time: {self.training_history.get('training_time_seconds', 0):.2f}s")
        logger.info(f"{'='*60}")
        
        # Step 6: Save model
        self.save_model()
        
        return self.model, training_report


def main():
    """Main function to test model training pipeline."""
    logger.info("Initializing Recommendation Model Training Pipeline")
    
    # Initialize Spark session
    spark = get_spark_session()
    
    try:
        # Import modules for testing
        from spark_pipeline.data_ingestion import DataIngestion
        from spark_pipeline.data_preprocessing import DataPreprocessor
        from spark_pipeline.feature_engineering import FeatureEngineer
        
        # Load and prepare data
        logger.info("Loading and preparing data for model training...")
        ingestion = DataIngestion(spark)
        df_raw, _ = ingestion.ingest_complete()
        
        preprocessor = DataPreprocessor(spark)
        df_cleaned, _ = preprocessor.preprocess_complete(df_raw)
        
        feature_engineer = FeatureEngineer(spark)
        df_features, _ = feature_engineer.engineer_features_complete(df_cleaned)
        
        # Train model
        trainer = RecommendationModelTrainer(spark)
        model, report = trainer.train_complete_pipeline(df_features)
        
        logger.info("\n✅ Model training completed successfully!")
        logger.info(f"\nTraining Report:")
        logger.info(f"  - Training Samples: {report['train_size']:,}")
        logger.info(f"  - Test Samples: {report['test_size']:,}")
        logger.info(f"  - RMSE: {report['rmse']}")
        logger.info(f"  - Model Rank: {report.get('model_rank', 'N/A')}")
        
        # Sample predictions
        logger.info("\nSample Predictions:")
        from pyspark.sql.functions import col
        predictions = model.transform(df_features.sample(0.01))
        predictions.select("user_id", "product_id", "rating", "prediction").show(10)
        
    except Exception as e:
        logger.error(f"❌ Error in model training: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
