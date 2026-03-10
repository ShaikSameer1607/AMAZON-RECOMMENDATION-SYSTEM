# Amazon Product Recommendation System - Production Pipeline

## 🎯 System Overview

Professional distributed recommendation system using **Apache Spark MLlib** and **Hadoop HDFS** for large-scale collaborative filtering.

### Architecture

```
Amazon Dataset (9GB)
    ↓
HDFS Distributed Storage
    ↓
Spark DataFrame (Distributed Read)
    ↓
Data Preprocessing (Distributed)
    ↓
Feature Engineering (StringIndexer)
    ↓
ALS Collaborative Filtering Model
    ↓
Model Evaluation (RMSE)
    ↓
Top-N Recommendations
    ↓
HDFS Output (Parquet + CSV)
```

---

## 📋 Prerequisites

### Required Software

- **Java**: OpenJDK 11 or higher
- **Hadoop**: 3.x with HDFS running
- **Spark**: 3.x with PySpark
- **Python**: 3.9 or 3.10 (**PySpark incompatible with Python3.13**)

### Environment Setup

```bash
# Option 1: Use conda environment (RECOMMENDED)
conda create -n spark_env python=3.10 -y
conda activate spark_env
pip install pyspark pyyaml pandas numpy

# Option 2: Use system Python3.9/3.10
python3.10 -m venv venv
source venv/bin/activate
pip install pyspark pyyaml
```

---

## 🚀 Quick Start

### 1. Verify HDFS is Running

```bash
jps | grep -E "NameNode|DataNode"
# Should show NameNode, DataNode, SecondaryNameNode
```

### 2. Verify Dataset in HDFS

```bash
hdfs dfs -ls/dataset/all_csv_files.csv
# Should show ~9GB file
```

### 3. Run the Distributed Pipeline

```bash
cd /home/sameer/BDA_PROJECT_NEW/amazon-recommendation-system

# Activate correct Python environment
conda activate spark_env  # or use python3.10

# Execute pipeline
python3 spark_pipeline/als_recommendation.py
```

---

## 📊 Pipeline Execution Steps

The pipeline automatically executes these steps:

1. **Load Data from HDFS** - Distributed CSV read
2. **Preprocessing** - Remove duplicates, filter invalid ratings
3. **Feature Engineering** - StringIndexer for user/item IDs
4. **Train/Test Split** - 80/20 random split
5. **ALS Model Training** - Collaborative filtering
6. **Model Evaluation** - RMSE calculation
7. **Generate Recommendations** - Top-10 per user
8. **Save to HDFS** - Parquet + CSV formats

---

## 📁 Output Locations

### HDFS Output

```bash
# View recommendations
hdfs dfs -ls/recommendations/

# View specific output
hdfs dfs -cat /recommendations/csv_YYYYMMDD_HHMMSS/part-00000*.csv | head -20

# View model
hdfs dfs -ls /models/als_model/
```

### Spark UI

Access at: **http://localhost:4040**

Shows:
- DAG visualization
- Stage details
- Task distribution
- Executor metrics

---

## ⚙️ Configuration

### Spark Configuration (spark_config.yaml)

```yaml
spark:
  app_name: "Amazon Product Recommendation System"
  executor_memory: "4G"
  driver_memory: "2G"
  executor_cores: 2
  shuffle_partitions: 200
  default_parallelism: 8
  
model:
  als:
   rank: 10
    max_iter: 15
   reg_param: 0.1
    alpha: 1.0
    cold_start_strategy: "drop"
```

### ALS Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `rank` | 10 | Number of latent factors |
| `maxIter` | 15 | Maximum iterations |
| `regParam` | 0.1 | Regularization parameter |
| `alpha` | 1.0 | Confidence parameter |
| `coldStartStrategy` | "drop" | Handle cold start|

---

## 🔍 Verification Commands

### Check HDFS Output

```bash
# List all recommendations
hdfs dfs -ls /recommendations/

# Count total recommendations
hdfs dfs -cat /recommendations/csv_*/part-00000*.csv | wc -l

# Sample recommendations
hdfs dfs -cat /recommendations/csv_*/part-00000*.csv | head -20
```

### Check Spark Jobs

```bash
# View Spark UI
firefox http://localhost:4040

# Or via command line (YARN)
yarn application -list
```

### View Logs

```bash
# Pipeline logs
tail -f outputs/logs/pipeline.log

# Spark driver logs
ls -lh $SPARK_HOME/logs/
```

---

## 🛠️ Troubleshooting

### Issue: PySpark + Python3.13 Incompatibility

**Problem**: `TypeError: 'JavaPackage' object is not callable`

**Solution**:
```bash
# Create Python3.10 environment
conda create -n spark_env python=3.10 -y
conda activate spark_env
pip install pyspark pyyaml
```

### Issue: HDFS Safe Mode

**Problem**: Cannot write to HDFS

**Solution**:
```bash
hdfs dfsadmin -safemode leave
```

### Issue: Spark UI Not Accessible

**Problem**: Port 4040 not responding

**Solution**:
- Ensure Spark context is active (don't close script immediately)
- Check if port is in use: `netstat -tulpn | grep 4040`
- Next job will use port 4041, 4042, etc.

---

## 📈 Performance Metrics

### Expected Performance (Single Node)

| Metric | Value |
|--------|-------|
| Data Loading | ~30 seconds |
| Preprocessing | ~15 seconds |
| Model Training | ~60-90 seconds |
| Recommendations | ~20 seconds |
| **Total Time** | **~2-3 minutes** |

### Scalability

- **Current**: Single-node Hadoop (~9GB data)
- **Scalable to**: Multi-node cluster (TB-scale data)
- **No code changes required** - Spark handles distribution automatically

---

## 🎓 Key Concepts Demonstrated

✅ **HDFS Distributed Storage**
- Data stored across HDFS blocks
- Fault-tolerant storage

✅ **Spark Distributed Processing**
- In-memory computation
- Lazy evaluation
- DAG execution

✅ **Spark MLlib ALS**
- Collaborative filtering
- Matrix factorization
- Latent factor models

✅ **Feature Engineering**
- StringIndexer for categorical encoding
- Train/test splitting
- Cross-validation ready

✅ **Model Evaluation**
- RMSE metric
- Prediction accuracy
- Model comparison

✅ **Production Pipeline**
- End-to-end automation
- Error handling
- Logging
- HDFS I/O

---

## 📝 Code Structure

```
spark_pipeline/
├── als_recommendation.py      # Main pipeline (distributed)
├── production_spark_builder.py # Spark session builder
└── configs/
    └── spark_config.yaml      # Configuration file
```

---

## 🔗 Additional Resources

- [Spark MLlib ALS Documentation](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)
- [HDFS Architecture](https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html)
- [Spark Tuning Guide](https://spark.apache.org/docs/latest/tuning.html)

---

## 👨‍💻 Author

Big Data Engineering Team  
Production Recommendation System v2.0

---

## 📄 License

Educational/Academic Use
