# Amazon Product Recommendation System

A **production-quality big data recommendation system** built with Apache Spark MLlib and Hadoop HDFS for distributed processing of large-scale Amazon ratings data.

## 🏗️ Architecture Overview

```
Amazon Ratings Dataset (6.7GB in HDFS)
            ↓
    HDFS Distributed Storage
    (128MB Block Distribution)
            ↓
      Apache Spark ETL
    (Distributed Processing)
            ↓
  Data Cleaning & Feature Engineering
            ↓
     Spark MLlib ALS Training
            ↓
   Collaborative Filtering Model
            ↓
    Evaluation & Metrics
            ↓
   Visualization & Insights
```

## 🎯 Features

### Core Capabilities

- **Distributed Data Processing**: Reads data directly from HDFS with automatic partitioning across cluster nodes
- **Collaborative Filtering**: ALS (Alternating Least Squares) algorithm for personalized recommendations
- **Comprehensive EDA**: Distributed exploratory data analysis with statistical insights
- **Feature Engineering**: Automated user activity, product popularity, and interaction features
- **Model Evaluation**: RMSE, Precision@K, Recall@K, Coverage metrics
- **Visual Analytics**: Professional plots and insights using matplotlib/seaborn

### Performance Optimizations

- HDFS block-aware partitioning (~128MB blocks)
- DataFrame caching for iterative ML operations
- Broadcast joins for small tables
- Kryo serialization for efficiency
- Adaptive query execution
- Dynamic resource allocation

## 📁 Project Structure

```
amazon-recommendation-system/
│
├── spark_pipeline/
│   ├── __init__.py                    # Package initialization
│   ├── spark_session_builder.py       # Spark session configuration
│   ├── data_ingestion.py              # HDFS data loading & validation
│   ├── data_preprocessing.py          # Cleaning & deduplication
│   ├── exploratory_analysis.py        # Distributed EDA computations
│   ├── feature_engineering.py         # Feature generation for ML
│   ├── recommendation_model.py        # ALS model training
│   ├── evaluation.py                  # RMSE, Precision@K, Recall@K
│   └── visualization.py               # Matplotlib/Seaborn plots
│
├── notebooks/
│   └── exploration.ipynb              # Interactive analysis (optional)
│
├── configs/
│   └── spark_config.yaml              # Spark configuration parameters
│
├── outputs/
│   ├── eda_results/                   # Exploratory analysis outputs
│   ├── models/                        # Trained model artifacts
│   ├── recommendation_results/        # Final recommendations
│   └── visualization/                 # PNG plots and charts
│
├── scripts/
│   ├── hdfs_setup.sh                 # HDFS directory setup
│   └── run_pipeline.sh               # Master execution script
│
└── README.md                          # This file
```

## 🚀 Quick Start

### Prerequisites

Ensure the following are installed and running:

- Apache Hadoop (HDFS) - Running on master:9000
- Apache Spark - With PySpark support
- Python 3.7+
- Required Python packages: `pyspark`, `pandas`, `matplotlib`, `seaborn`, `numpy`

### Installation

1. **Clone or navigate to project directory**:
```bash
cd /home/sameer/BDA_PROJECT_NEW/amazon-recommendation-system
```

2. **Verify HDFS dataset exists**:
```bash
hdfs dfs -ls hdfs://master:9000/dataset/all_csv_files.csv
```

3. **Setup HDFS directories**:
```bash
bash scripts/hdfs_setup.sh
```

### Execution

#### Option 1: Run Complete Pipeline (Recommended)

Execute all pipeline steps automatically:

```bash
bash scripts/run_pipeline.sh
```

This will run:
1. HDFS setup
2. Data ingestion
3. Preprocessing
4. EDA
5. Feature engineering
6. Model training
7. Evaluation
8. Visualization

#### Option 2: Run Individual Modules

Execute pipeline components separately:

```bash
# Step 1: Data Ingestion
python3 spark_pipeline/data_ingestion.py

# Step 2: Preprocessing
python3 spark_pipeline/data_preprocessing.py

# Step 3: EDA
python3 spark_pipeline/exploratory_analysis.py

# Step 4: Feature Engineering
python3 spark_pipeline/feature_engineering.py

# Step 5: Model Training
python3 spark_pipeline/recommendation_model.py

# Step 6: Evaluation
python3 spark_pipeline/evaluation.py

# Step 7: Visualization
python3 spark_pipeline/visualization.py
```

## 📊 Pipeline Details

### Step 1: Data Ingestion (`data_ingestion.py`)

Loads the 6.7GB Amazon ratings dataset from HDFS with distributed processing.

**Key Operations**:
- Direct HDFS CSV reading with Spark
- Schema validation and inference
- Null value detection
- Automatic repartitioning to match HDFS blocks
- Dataset caching for performance

**Output**: Validated DataFrame ready for preprocessing

```python
df = spark.read.option("header","true") \
               .option("inferSchema","true") \
               .csv("hdfs://master:9000/dataset/all_csv_files.csv")
```

### Step 2: Data Preprocessing (`data_preprocessing.py`)

Cleans and prepares data for ML processing.

**Key Operations**:
- Column selection (user_id, product_id, rating)
- Null value handling (drop strategy)
- Duplicate removal (keep latest rating)
- Type conversion (int/float optimization)
- Invalid rating filtering (0.5-5.0 range)

**Output**: Clean DataFrame with validated ratings

### Step 3: Exploratory Data Analysis (`exploratory_analysis.py`)

Performs distributed statistical analysis.

**Computations**:
- Total users and products
- Rating distribution (mean, std, min, max)
- Top 20 rated products
- Most active users
- Product popularity metrics
- User activity scores
- Temporal patterns (if timestamp available)

**Output**: Statistical summaries saved to HDFS

### Step 4: Feature Engineering (`feature_engineering.py`)

Creates ML-ready features.

**Features Created**:
- `user_activity_score`: Number of ratings per user
- `user_avg_rating`: Average rating given by user
- `product_popularity`: Number of ratings per product
- `product_avg_rating`: Average rating of product
- `interaction_frequency`: Z-score normalized activity
- `user_rank`, `product_rank`: Density-based ranks

**Output**: Enriched DataFrame with engineered features

### Step 5: Model Training (`recommendation_model.py`)

Trains ALS collaborative filtering model.

**Configuration**:
```python
als = ALS(
    userCol="user_id",
    itemCol="product_id", 
    ratingCol="rating",
    coldStartStrategy="drop",
    nonnegative=True,
    rank=10,
    maxIter=15,
    regParam=0.1
)
```

**Process**:
- 80/20 train/test split
- Distributed ALS training
- Model persistence to HDFS

**Output**: Trained ALS model saved to HDFS

### Step 6: Evaluation (`evaluation.py`)

Comprehensive model evaluation.

**Metrics**:
- **RMSE**: Root Mean Square Error (prediction accuracy)
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **Precision@10**: Top-10 recommendation precision
- **Recall@10**: Top-10 recommendation recall
- **Coverage**: Percentage of catalog recommended
- **Diversity**: Recommendation variety

**Output**: Evaluation report with all metrics

### Step 7: Visualization (`visualization.py`)

Creates professional analytical plots.

**Plots Generated**:
1. Rating distribution histogram with KDE
2. Top 20 products bar chart
3. User activity distribution (dual plot)
4. User-product interaction heatmap
5. Recommendation score distribution
6. Evaluation metrics comparison

**Output**: PNG files in `outputs/visualization/`

## ⚙️ Configuration

Edit `configs/spark_config.yaml` to customize:

### Spark Settings

```yaml
spark:
  app_name: "Amazon Product Recommendation System"
  executor_memory: "4G"
  driver_memory: "2G"
  executor_cores: 2
  shuffle_partitions: 200
  default_parallelism: 8
  serializer: "org.apache.spark.serializer.KryoSerializer"
```

### ALS Model Hyperparameters

```yaml
model:
  als:
    rank: 10           # Latent factors
    max_iter: 15       # Iterations
    reg_param: 0.1     # Regularization
    alpha: 1.0         # Confidence weight
```

### Data Paths

```yaml
data:
  hdfs_input_path: "hdfs://master:9000/dataset/all_csv_files.csv"
  hdfs_output_base: "hdfs://master:9000/"
  recommendations_output_path: "hdfs://master:9000/recommendation_results/recommendations"
```

## 📈 Output Locations

After pipeline execution:

### HDFS Outputs

```
hdfs://master:9000/recommendation_results/
├── models/
│   └── als_model/              # Trained ALS model
├── eda_results/
│   ├── top_rated_products/     # Top products CSV
│   ├── most_active_users/      # Active users CSV
│   ├── product_popularity/     # Popularity metrics
│   └── user_activity/          # Activity scores
└── recommendations/
    └── predictions/            # Generated recommendations
```

### Local Outputs

```
/home/sameer/BDA_PROJECT_NEW/amazon-recommendation-system/outputs/
└── visualization/
    ├── rating_distribution.png
    ├── top_products.png
    ├── user_activity.png
    ├── interaction_heatmap.png
    ├── recommendation_scores.png
    └── evaluation_metrics.png
```

## 🔬 Technical Details

### Distributed Processing Strategy

The pipeline leverages HDFS block distribution for parallel processing:

1. **Data Partitioning**: 6.7GB dataset split into ~128MB HDFS blocks
2. **Parallel Reading**: Spark reads blocks simultaneously across workers
3. **Partition Alignment**: Repartitioning matches HDFS block boundaries
4. **In-Memory Caching**: Frequently used DataFrames cached in cluster memory
5. **Lazy Evaluation**: Transformations optimized by Catalyst optimizer

### ALS Algorithm

Alternating Least Squares factorizes the user-item interaction matrix:

```
R ≈ U × V^T
```

Where:
- R: User-Item rating matrix
- U: User latent factor matrix
- V: Item latent factor matrix

**Objective**: Minimize squared error between predicted and actual ratings

### Performance Considerations

**Memory Management**:
- Executor memory: 4GB per node
- Driver memory: 2GB
- Storage fraction: 50% for caching
- Kryo serialization for compact storage

**Parallelism**:
- Default parallelism: Cluster core count
- Shuffle partitions: 200
- Dynamic allocation enabled
- Adaptive query execution

## 🧪 Testing Individual Components

Test data ingestion only:

```bash
python3 spark_pipeline/data_ingestion.py
```

Test model training only (requires preprocessed data):

```bash
python3 spark_pipeline/recommendation_model.py
```

## 📊 Sample Usage in Python

```python
from pyspark.sql import SparkSession
from spark_pipeline.data_ingestion import DataIngestion
from spark_pipeline.recommendation_model import RecommendationModelTrainer

# Initialize Spark
spark = SparkSession.builder.appName("Recommendation System").getOrCreate()

# Load data
ingestion = DataIngestion(spark)
df, report = ingestion.ingest_complete()

# Train model
trainer = RecommendationModelTrainer(spark)
model, training_report = trainer.train_complete_pipeline(df)

# Generate recommendations
user_id = 12345
user_items = df.filter(df.user_id == user_id)
recommendations = model.transform(user_items)
recommendations.show()

spark.stop()
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Dataset not found in HDFS"
```bash
# Solution: Upload dataset to HDFS
hdfs dfs -put /local/path/all_csv_files.csv /dataset/
```

**Issue**: "Out of memory error"
```bash
# Solution: Increase executor memory in spark_config.yaml
spark:
  executor_memory: "8G"
  driver_memory: "4G"
```

**Issue**: "Too many open files"
```bash
# Solution: Increase ulimit
ulimit -n 65536
```

**Issue**: Slow performance
```bash
# Solution: Check cluster resources
yarn application -list
# Monitor Spark UI at http://master:4040
```

## 📝 Logging

All modules use Python logging with INFO level by default.

Logs show:
- Data statistics
- Processing progress
- Model training metrics
- Evaluation results
- File save confirmations

To enable DEBUG logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔐 Security Notes

- HDFS permissions must allow read/write access
- Kerberos authentication if enabled on cluster
- No sensitive data stored in logs
- Model artifacts stored in HDFS with appropriate permissions

## 📄 License

This project is for educational and research purposes.

## 👥 Authors

Big Data Engineering Team

## 📚 References

- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [Spark MLlib Guide](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)
- [ALS Paper](https://doi.org/10.1109/ICDM.2008.22)
- [Hadoop HDFS Architecture](https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-hdfs/HdfsDesign.html)

## 🎓 Educational Value

This project demonstrates:
- Big data pipeline architecture
- Distributed computing with Spark
- HDFS integration patterns
- Collaborative filtering implementation
- Production-quality code organization
- Data engineering best practices

---

**For detailed technical documentation, see individual module docstrings and comments.**
