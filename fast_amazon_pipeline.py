#!/usr/bin/env python3
import logging, pandas as pd, numpy as np, subprocess
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
   print("="*60)
   print("FAST AMAZON RECOMMENDATION PIPELINE")
   print("="*60)
   start_time = datetime.now()
   try:
       logger.info("Loading data from HDFS...")
       temp_file = "/tmp/amazon_sample.csv"
       hdfs_path = "hdfs://localhost:9000/dataset/all_csv_files.csv"
       result = subprocess.run(["hdfs", "dfs", "-text", hdfs_path],capture_output=True,text=True)
        if result.returncode == 0:
            lines = result.stdout.split(chr(10))[:10001]
            with open(temp_file, "w") as f:
                f.write(chr(10).join(lines))
           data = pd.read_csv(temp_file)
           logger.info(f"Loaded {len(data)} rows")
       logger.info("Preprocessing...")
       data = data.drop_duplicates()
        num_cols = data.select_dtypes(include=[np.number]).columns
       data[num_cols] = data[num_cols].fillna(0)
        if "rating" in data.columns:
           data = data[data["rating"] > 0]
       logger.info("Training popularity model...")
       product_stats = data.groupby("product_id").agg({"rating": ["mean", "count"]}).reset_index()
       product_stats.columns = ["product_id", "avg_rating", "num_ratings"]
       product_stats["score"] = product_stats["avg_rating"] * np.log(product_stats["num_ratings"] + 1)
       product_stats = product_stats.sort_values("score", ascending=False)
       logger.info("Generating recommendations...")
        top_products = product_stats.head(10)["product_id"].tolist()
       recommendations = []
        unique_users = data["user_id"].unique()[:100]
       for user_id in unique_users:
           for rank, product_id in enumerate(top_products, 1):
               recommendations.append({"user_id": int(user_id), "product_id": product_id, "rank": rank})
       recs_df = pd.DataFrame(recommendations)
       logger.info(f"Generated {len(recs_df)} recommendations")
       logger.info("Saving results...")
        output_base = Path("/home/sameer/BDA_PROJECT_NEW/amazon-recommendation-system/outputs")
       recs_dir = output_base / "recommendation_results"
       recs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       recs_path = recs_dir / f"fast_recommendations_{timestamp}.csv"
       recs_df.to_csv(recs_path, index=False)
       elapsed = datetime.now() - start_time
       print(chr(10) + "="*60)
       print("PIPELINE COMPLETED SUCCESSFULLY!")
       print(f"Total time: {elapsed}")
       print(f"Output: {recs_path}")
       print("="*60)
    except Exception as e:
       logger.error(f"Pipeline failed: {e}", exc_info=True)
       print(f"Pipeline failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
