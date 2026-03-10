# Quick Start Guide - Amazon Product Recommendation System

## 🚀 5-Minute Setup

### Prerequisites Check

```bash
# Verify Hadoop is running
hdfs dfs -ls /

# Verify Spark is available
spark-submit --version

# Verify Python packages
python3 -c "import pyspark; print(pyspark.__version__)"
```

### Installation & Execution

```bash
# Navigate to project
cd /home/sameer/BDA_PROJECT_NEW/amazon-recommendation-system

# Setup HDFS directories
bash scripts/hdfs_setup.sh

# Run complete pipeline (15-30 minutes depending on cluster)
bash scripts/run_pipeline.sh
```

That's it! The pipeline will execute all 7 steps automatically.

---

## 📊 What You'll Get

After pipeline execution:

### Models (HDFS)
```
hdfs://master:9000/recommendation_results/models/als_model/
```

### Recommendations (HDFS)
```
hdfs://master:9000/recommendation_results/recommendations/
```

### Visualizations (Local)
```
/home/sameer/BDA_PROJECT_NEW/amazon-recommendation-system/outputs/visualization/
├── rating_distribution.png
├── top_products.png
├── user_activity.png
├── interaction_heatmap.png
├── recommendation_scores.png
└── evaluation_metrics.png
```

---

## 🔧 Common Tasks

### Run Specific Pipeline Step

```bash
# Data ingestion only
python3 spark_pipeline/data_ingestion.py

# Model training only
python3 spark_pipeline/recommendation_model.py

# Evaluation only
python3 spark_pipeline/evaluation.py
```

### Use Jupyter Notebook

```bash
# Launch Jupyter
jupyter notebook notebooks/exploration.ipynb
```

### Check Results in HDFS

```bash
# List model files
hdfs dfs -ls hdfs://master:9000/recommendation_results/models/

# View recommendations
hdfs dfs -cat hdfs://master:9000/recommendation_results/recommendations/part-*.csv | head
```

---

## ⚙️ Configuration Quick Reference

### Change Model Parameters

Edit `configs/spark_config.yaml`:

```yaml
model:
  als:
    rank: 20          # More latent factors
    max_iter: 25      # More iterations
   reg_param: 0.05   # Less regularization
```

### Adjust Spark Resources

Edit `configs/spark_config.yaml`:

```yaml
spark:
  executor_memory: "8G"   # More memory
  executor_cores: 4       # More cores per executor
  shuffle_partitions: 400 # More parallelism
```

---

## 🐛 Troubleshooting

### Problem: Memory Errors

**Solution**: Increase memory in `spark_config.yaml`
```yaml
spark:
  executor_memory: "8G"
  driver_memory: "4G"
```

### Problem: Slow Performance

**Solutions**:
1. Check cluster resources: `yarn application -list`
2. Monitor Spark UI: `http://master:4040`
3. Increase parallelism: `shuffle_partitions: 400`

### Problem: Dataset Not Found

**Solution**: Upload dataset to HDFS
```bash
hdfs dfs -put /path/to/all_csv_files.csv /dataset/
```

---

## 📈 Understanding Output

### Evaluation Metrics

- **RMSE < 1.0**: Good prediction accuracy
- **Precision@10 > 0.1**: Decent recommendation quality
- **Recall@10 > 0.05**: Reasonable coverage
- **Coverage > 50%**: Model recommends most products

### Visualization Files

1. **rating_distribution.png**: Overall rating histogram
2. **top_products.png**: Best-rated products bar chart
3. **user_activity.png**: User engagement distribution
4. **interaction_heatmap.png**: User-product matrix sample
5. **recommendation_scores.png**: Prediction score distribution
6. **evaluation_metrics.png**: Model performance summary

---

## 🎯 Next Steps

### Customize for Your Use Case

1. **Add more features**: Edit `feature_engineering.py`
2. **Try different algorithms**: Modify `recommendation_model.py`
3. **Hyperparameter tuning**: Add grid search in training
4. **Real-time recommendations**: Export model for serving

### Production Deployment

1. Set up scheduled pipeline runs (cron/Airflow)
2. Monitor model performance over time
3. Implement A/B testing framework
4. Add model retraining triggers

---

## 📞 Support

For detailed documentation, see [README.md](README.md)

For technical details, check module docstrings:
```python
from spark_pipeline.data_ingestion import DataIngestion
help(DataIngestion)
```

---

**Happy Recommending! 🎉**
