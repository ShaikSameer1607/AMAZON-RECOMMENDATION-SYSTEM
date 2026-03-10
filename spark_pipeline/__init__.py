"""
Amazon Product Recommendation System - Spark Pipeline

A production-quality big data recommendation system using Apache Spark MLlib
and Hadoop HDFS for distributed processing of large-scale Amazon ratings data.
"""

from spark_pipeline.spark_session_builder import SparkSessionBuilder, get_spark_session
from spark_pipeline.data_ingestion import DataIngestion
from spark_pipeline.data_preprocessing import DataPreprocessor
from spark_pipeline.exploratory_analysis import ExploratoryDataAnalysis
from spark_pipeline.feature_engineering import FeatureEngineer
from spark_pipeline.recommendation_model import RecommendationModelTrainer
from spark_pipeline.evaluation import ModelEvaluator
from spark_pipeline.visualization import RecommendationVisualizer

__version__ = "1.0.0"
__author__ = "Big Data Engineering Team"

__all__ = [
    'SparkSessionBuilder',
    'get_spark_session',
    'DataIngestion',
    'DataPreprocessor',
    'ExploratoryDataAnalysis',
    'FeatureEngineer',
    'RecommendationModelTrainer',
    'ModelEvaluator',
    'RecommendationVisualizer'
]
