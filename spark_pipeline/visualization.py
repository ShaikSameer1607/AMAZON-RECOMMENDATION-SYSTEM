"""
Visualization Module

Creates comprehensive visual analytics for the recommendation system
using matplotlib, seaborn, and plotly.
"""

import logging
from typing import Dict, Optional, List, Any
from pyspark.sql import DataFrame
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RecommendationVisualizer:
    """
    Creates comprehensive visualizations for recommendation system analysis
    including rating distributions, user activity, product popularity, and model performance.
    """
    
    def __init__(self, output_dir: str = "/home/sameer/BDA_PROJECT_NEW/amazon-recommendation-system/outputs/visualization"):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        self.plots_created: List[str] = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def _convert_spark_to_pandas(self, df: DataFrame, sample_limit: int = 100000) -> pd.DataFrame:
        """
        Convert Spark DataFrame to Pandas for visualization.
        
        Args:
            df: Spark DataFrame
            sample_limit: Maximum rows to sample
            
        Returns:
            pd.DataFrame: Pandas DataFrame
        """
        count = df.count()
        if count > sample_limit:
            logger.info(f"Sampling {sample_limit:,} rows from {count:,} total rows")
            pdf = df.sample(sample_limit / count).toPandas()
        else:
            pdf = df.toPandas()
        
        return pdf
    
    def plot_rating_distribution(self, df: DataFrame, 
                                figsize: tuple = (12, 6)) -> str:
        """
        Plot rating distribution histogram with KDE.
        
        Args:
            df: DataFrame with 'rating' column
            figsize: Figure size
            
        Returns:
            str: Path to saved plot
        """
        logger.info("Creating rating distribution plot...")
        
        pdf = self._convert_spark_to_pandas(df)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Histogram with KDE
        sns.histplot(data=pdf, x='rating', kde=True, bins=20, 
                    color='skyblue', edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('Rating', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of User Ratings\nAmazon Product Dataset', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Add statistics text box
        stats_text = f"Mean: {pdf['rating'].mean():.2f}\n"
        stats_text += f"Std: {pdf['rating'].std():.2f}\n"
        stats_text += f"Min: {pdf['rating'].min():.1f}\n"
        stats_text += f"Max: {pdf['rating'].max():.1f}"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'rating_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plots_created.append(output_path)
        logger.info(f"Saved: {output_path}")
        
        return output_path
    
    def plot_top_products(self, top_products_df: DataFrame, 
                         top_n: int = 20,
                         figsize: tuple = (14, 8)) -> str:
        """
        Plot top N products by average rating.
        
        Args:
            top_products_df: DataFrame with product_id, avg_rating, review_count
            top_n: Number of top products to show
            figsize: Figure size
            
        Returns:
            str: Path to saved plot
        """
        logger.info(f"Creating top {top_n} products plot...")
        
        pdf = self._convert_spark_to_pandas(top_products_df.limit(top_n))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Sort by average rating
        pdf = pdf.sort_values('avg_rating', ascending=False)
        
        # Create bar plot
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(pdf)))
        bars = ax.barh(range(len(pdf)), pdf['avg_rating'], color=colors, 
                      edgecolor='black', linewidth=0.5)
        
        # Customize y-axis
        ax.set_yticks(range(len(pdf)))
        ax.set_yticklabels([f"Product {pid}" for pid in pdf['product_id']], 
                          fontsize=9)
        ax.invert_yaxis()
        
        ax.set_xlabel('Average Rating', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Rated Products\nBy Average Customer Rating', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(pdf.iterrows()):
            ax.text(row['avg_rating'] + 0.05, i, f"{row['avg_rating']:.2f}", 
                   va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'top_products.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plots_created.append(output_path)
        logger.info(f"Saved: {output_path}")
        
        return output_path
    
    def plot_user_activity(self, user_activity_df: DataFrame,
                          figsize: tuple = (12, 6)) -> str:
        """
        Plot user activity distribution.
        
        Args:
            user_activity_df: DataFrame with user activity metrics
            figsize: Figure size
            
        Returns:
            str: Path to saved plot
        """
        logger.info("Creating user activity distribution plot...")
        
        pdf = self._convert_spark_to_pandas(user_activity_df)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left plot: Activity score distribution
        sns.histplot(data=pdf, x='activity_score', kde=True, ax=ax1, 
                    color='coral', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Number of Ratings', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('User Activity Distribution\n(Ratings per User)', 
                     fontsize=12, fontweight='bold')
        
        # Right plot: Average rating distribution
        sns.histplot(data=pdf, x='avg_rating', kde=True, ax=ax2, 
                    color='teal', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Average Rating', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Distribution of User Average Ratings', 
                     fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'user_activity.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plots_created.append(output_path)
        logger.info(f"Saved: {output_path}")
        
        return output_path
    
    def plot_interaction_heatmap(self, df: DataFrame, 
                                sample_users: int = 50,
                                sample_products: int = 50,
                                figsize: tuple = (14, 12)) -> str:
        """
        Plot heatmap of user-product interactions (sampled).
        
        Args:
            df: DataFrame with user_id, product_id, rating
            sample_users: Number of users to sample
            sample_products: Number of products to sample
            figsize: Figure size
            
        Returns:
            str: Path to saved plot
        """
        logger.info("Creating interaction heatmap...")
        
        # Sample data
        pdf = self._convert_spark_to_pandas(df, sample_limit=10000)
        
        # Get top users and products by activity
        top_users = pdf.groupby('user_id').size().nlargest(sample_users).index
        top_products = pdf.groupby('product_id').size().nlargest(sample_products).index
        
        # Filter and pivot
        sampled = pdf[(pdf['user_id'].isin(top_users)) & 
                     (pdf['product_id'].isin(top_products))]
        
        pivot = sampled.pivot_table(values='rating', index='user_id', 
                                   columns='product_id', aggfunc='mean')
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(pivot, cmap='YlOrRd', cbar_kws={'label': 'Rating'}, 
                   ax=ax, vmin=0.5, vmax=5.0)
        
        ax.set_xlabel('Product ID', fontsize=12, fontweight='bold')
        ax.set_ylabel('User ID', fontsize=12, fontweight='bold')
        ax.set_title('User-Product Interaction Heatmap\nSample of Top Users and Products', 
                    fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'interaction_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plots_created.append(output_path)
        logger.info(f"Saved: {output_path}")
        
        return output_path
    
    def plot_evaluation_metrics(self, evaluation_results: Dict[str, float],
                               figsize: tuple = (12, 6)) -> str:
        """
        Plot model evaluation metrics comparison.
        
        Args:
            evaluation_results: Dictionary of metric names and values
            figsize: Figure size
            
        Returns:
            str: Path to saved plot
        """
        logger.info("Creating evaluation metrics plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Filter metrics
        error_metrics = {k: v for k, v in evaluation_results.items() 
                        if k in ['rmse', 'mae']}
        ranking_metrics = {k: v for k, v in evaluation_results.items() 
                          if 'precision' in k or 'recall' in k}
        
        # Left: Error metrics
        if error_metrics:
            colors1 = ['crimson', 'darkred']
            bars1 = ax1.bar(range(len(error_metrics)), list(error_metrics.values()), 
                           color=colors1, edgecolor='black', alpha=0.8)
            ax1.set_xticks(range(len(error_metrics)))
            ax1.set_xticklabels([k.upper() for k in error_metrics.keys()], 
                               fontweight='bold')
            ax1.set_ylabel('Error Value', fontsize=11, fontweight='bold')
            ax1.set_title('Prediction Error Metrics\n(Lower is Better)', 
                         fontsize=12, fontweight='bold')
            
            # Add value labels
            for bar, val in zip(bars1, error_metrics.values()):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.3f}', ha='center', va='bottom', 
                        fontsize=10, fontweight='bold')
        
        # Right: Ranking metrics
        if ranking_metrics:
            colors2 = plt.cm.Set2(np.linspace(0.1, 0.9, len(ranking_metrics)))
            bars2 = ax2.bar(range(len(ranking_metrics)), list(ranking_metrics.values()), 
                           color=colors2, edgecolor='black', alpha=0.8)
            ax2.set_xticks(range(len(ranking_metrics)))
            ax2.set_xticklabels([k.replace('_', '@').upper() for k in ranking_metrics.keys()], 
                               fontweight='bold', rotation=45, ha='right')
            ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
            ax2.set_title('Ranking Quality Metrics\n(Higher is Better)', 
                         fontsize=12, fontweight='bold')
            ax2.set_ylim(0, max(ranking_metrics.values()) * 1.2)
            
            # Add value labels
            for bar, val in zip(bars2, ranking_metrics.values()):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{val:.3f}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'evaluation_metrics.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plots_created.append(output_path)
        logger.info(f"Saved: {output_path}")
        
        return output_path
    
    def plot_recommendation_score_distribution(self, recommendations_df: DataFrame,
                                              figsize: tuple = (12, 6)) -> str:
        """
        Plot distribution of recommendation scores/predictions.
        
        Args:
            recommendations_df: DataFrame with prediction scores
            figsize: Figure size
            
        Returns:
            str: Path to saved plot
        """
        logger.info("Creating recommendation score distribution plot...")
        
        pdf = self._convert_spark_to_pandas(recommendations_df)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.histplot(data=pdf, x='prediction', kde=True, bins=30, 
                    color='mediumorchid', edgecolor='black', alpha=0.7)
        
        ax.set_xlabel('Predicted Rating Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Recommendation Scores\nModel Predictions', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Add mean line
        mean_pred = pdf['prediction'].mean()
        ax.axvline(mean_pred, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_pred:.2f}')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(self.output_dir, 'recommendation_scores.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plots_created.append(output_path)
        logger.info(f"Saved: {output_path}")
        
        return output_path
    
    def create_all_visualizations(self, data_dict: Dict[str, DataFrame], 
                                 evaluation_results: Optional[Dict] = None) -> List[str]:
        """
        Create complete suite of visualizations.
        
        Args:
            data_dict: Dictionary of DataFrames for different plots
            evaluation_results: Optional evaluation metrics dictionary
            
        Returns:
            List[str]: Paths to all created plots
        """
        logger.info("=" * 60)
        logger.info("CREATING COMPLETE VISUALIZATION SUITE")
        logger.info("=" * 60)
        
        plots = []
        
        # 1. Rating Distribution
        if 'ratings_df' in data_dict:
            plots.append(self.plot_rating_distribution(data_dict['ratings_df']))
        
        # 2. Top Products
        if 'top_products_df' in data_dict:
            plots.append(self.plot_top_products(data_dict['top_products_df']))
        
        # 3. User Activity
        if 'user_activity_df' in data_dict:
            plots.append(self.plot_user_activity(data_dict['user_activity_df']))
        
        # 4. Interaction Heatmap
        if 'ratings_df' in data_dict:
            plots.append(self.plot_interaction_heatmap(data_dict['ratings_df']))
        
        # 5. Recommendation Scores
        if 'recommendations_df' in data_dict:
            plots.append(self.plot_recommendation_score_distribution(
                data_dict['recommendations_df']
            ))
        
        # 6. Evaluation Metrics
        if evaluation_results:
            plots.append(self.plot_evaluation_metrics(evaluation_results))
        
        logger.info(f"\n✅ Created {len(plots)} visualizations")
        logger.info("Plots saved:")
        for plot_path in plots:
            logger.info(f"  - {plot_path}")
        logger.info("=" * 60)
        
        return plots


def main():
    """Main function to test visualization pipeline."""
    logger.info("Initializing Visualization Pipeline")
    
    try:
        # Import modules for testing
        from spark_pipeline.simple_spark_builder import get_spark_session
        from spark_pipeline.data_ingestion import DataIngestion
        from spark_pipeline.data_preprocessing import DataPreprocessor
        from spark_pipeline.exploratory_analysis import ExploratoryDataAnalysis
        from spark_pipeline.feature_engineering import FeatureEngineer
        from spark_pipeline.recommendation_model import RecommendationModelTrainer
        
        # Initialize Spark
        spark = get_spark_session()
        
        # Load and prepare data
        logger.info("Loading data for visualization...")
        ingestion = DataIngestion(spark)
        df_raw, _ = ingestion.ingest_complete()
        
        preprocessor = DataPreprocessor(spark)
        df_cleaned, _ = preprocessor.preprocess_complete(df_raw)
        
        # Run EDA to get analysis DataFrames
        eda = ExploratoryDataAnalysis(spark)
        eda.run_complete_eda(df_cleaned, save_to_hdfs=False)
        
        feature_engineer = FeatureEngineer(spark)
        df_features, _ = feature_engineer.engineer_features_complete(df_cleaned)
        
        # Train model and get recommendations
        trainer = RecommendationModelTrainer(spark)
        model, _ = trainer.train_complete_pipeline(df_features)
        
        # Generate sample recommendations
        recommendations = model.transform(df_features.sample(0.1))
        
        # Prepare data dictionary
        data_dict = {
            'ratings_df': df_cleaned,
            'top_products_df': eda.eda_results.get('top_rated_products', df_cleaned),
            'user_activity_df': eda.eda_results.get('user_activity', df_cleaned),
            'recommendations_df': recommendations
        }
        
        # Create visualizations
        visualizer = RecommendationVisualizer()
        plots = visualizer.create_all_visualizations(
            data_dict=data_dict,
            evaluation_results={'rmse': 0.85, 'mae': 0.65, 'precision_at_10': 0.15, 'recall_at_10': 0.08}
        )
        
        logger.info("\n✅ Visualization pipeline completed successfully!")
        
        spark.stop()
        
    except Exception as e:
        logger.error(f"❌ Error in visualization: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
