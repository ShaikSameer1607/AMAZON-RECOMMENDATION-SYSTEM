# Pipeline Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AMAZON PRODUCT RECOMMENDATION SYSTEM                  │
│                         Big Data Pipeline Architecture                   │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA LAYER (HDFS)                              │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │  hdfs://master:9000/dataset/all_csv_files.csv             │         │
│  │  Size: 6.7GB | Blocks: ~128MB each | Replication: 3x      │         │
│  └────────────────────────────────────────────────────────────┘         │
│                            ↓                                             │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │  HDFS Distributed Block Storage                            │         │
│  │  [Block 1] [Block 2] [Block 3] ... [Block N]              │         │
│  │     ↓        ↓        ↓               ↓                    │         │
│  │  Worker 1  Worker 2  Worker 3     Worker N                 │         │
│  └────────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      COMPUTE LAYER (APACHE SPARK)                        │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │              Spark Driver (Master Node)                  │           │
│  │  - SparkSession initialization                           │           │
│  │  - DAG scheduling                                        │           │
│  │  - Task coordination                                     │           │
│  │  - Result aggregation                                    │           │
│  └──────────────────────────────────────────────────────────┘           │
│                            ↓                                             │
│  ┌──────────────────────────────────────────────────────────┐           │
│  │           Spark Executors (Worker Nodes)                 │           │
│  │  ┌──────────┬──────────┬──────────┬──────────┐          │           │
│  │  │ Executor │ Executor │ Executor │ Executor │          │           │
│  │  │    1     │    2     │    3     │    4     │          │           │
│  │  │  Task    │  Task    │  Task    │  Task    │          │           │
│  │  │ Partition│ Partition│ Partition│ Partition│          │           │
│  │  └──────────┴──────────┴──────────┴──────────┘          │           │
│  └──────────────────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE STAGES                                  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────┐             │
│  │ STAGE 1: DATA INGESTION                                │             │
│  │ ─────────────────────────────────────────────────────  │             │
│  │ • Read CSV from HDFS with parallel partitioning        │             │
│  │ • Schema validation and type inference                 │             │
│  │ • Null value detection and reporting                   │             │
│  │ • Repartition to match HDFS block distribution         │             │
│  │ • Cache dataset for iterative processing               │             │
│  │                                                        │             │
│  │ Output: Validated DataFrame (cached in memory)         │             │
│  └────────────────────────────────────────────────────────┘             │
│                            ↓                                             │
│  ┌────────────────────────────────────────────────────────┐             │
│  │ STAGE 2: DATA PREPROCESSING                            │             │
│  │ ─────────────────────────────────────────────────────  │             │
│  │ • Select required columns (user_id, product_id, rating)│             │
│  │ • Drop rows with null values                           │             │
│  │ • Remove duplicate user-product pairs                 │             │
│  │ • Convert data types (int/float optimization)          │             │
│  │ • Filter invalid ratings (0.5 - 5.0 range)             │             │
│  │                                                        │             │
│  │ Output: Clean DataFrame ready for ML                   │             │
│  └────────────────────────────────────────────────────────┘             │
│                            ↓                                             │
│  ┌────────────────────────────────────────────────────────┐             │
│  │ STAGE 3: EXPLORATORY DATA ANALYSIS                     │             │
│  │ ─────────────────────────────────────────────────────  │             │
│  │ • Compute basic statistics (users, products, ratings)  │             │
│  │ • Rating distribution analysis                         │             │
│  │ • Top-N aggregations (products, users)                 │             │
│  │ • Product popularity metrics                           │             │
│  │ • User activity scoring                                │             │
│  │ • Temporal pattern detection (if timestamp available)  │             │
│  │                                                        │             │
│  │ Output: Statistical summaries saved to HDFS            │             │
│  └────────────────────────────────────────────────────────┘             │
│                            ↓                                             │
│  ┌────────────────────────────────────────────────────────┐             │
│  │ STAGE 4: FEATURE ENGINEERING                           │             │
│  │ ─────────────────────────────────────────────────────  │             │
│  │ • Compute user activity features                       │             │
│  │   - user_activity_score                                │             │
│  │   - user_avg_rating                                    │             │
│  │   - user_rating_stddev                                 │             │
│  │ • Compute product popularity features                  │             │
│  │   - product_popularity                                 │             │
│  │   - product_avg_rating                                 │             │
│  │   - product_rating_stddev                              │             │
│  │ • Join features using broadcast optimization           │             │
│  │ • Z-score normalization for interaction frequency      │             │
│  │ • Create rank-based features                           │             │
│  │                                                        │             │
│  │ Output: Enriched DataFrame with ML-ready features      │             │
│  └────────────────────────────────────────────────────────┘             │
│                            ↓                                             │
│  ┌────────────────────────────────────────────────────────┐             │
│  │ STAGE 5: MODEL TRAINING (ALS)                          │             │
│  │ ─────────────────────────────────────────────────────  │             │
│  │ • Split data (80% train, 20% test)                     │             │
│  │ • Configure ALS hyperparameters                       │             │
│  │   - rank: 10 latent factors                           │             │
│  │   - max_iter: 15 iterations                            │             │
│  │   - reg_param: 0.1 regularization                      │             │
│  │ • Train collaborative filtering model                  │             │
│  │ • Handle cold-start cases (drop strategy)              │             │
│  │ • Save trained model to HDFS                           │             │
│  │                                                        │             │
│  │ Output: ALSModel saved to HDFS                         │             │
│  └────────────────────────────────────────────────────────┘             │
│                            ↓                                             │
│  ┌────────────────────────────────────────────────────────┐             │
│  │ STAGE 6: MODEL EVALUATION                              │             │
│  │ ─────────────────────────────────────────────────────  │             │
│  │ • Generate predictions on test set                     │             │
│  │ • Compute error metrics:                               │             │
│  │   - RMSE (Root Mean Square Error)                      │             │
│  │   - MAE (Mean Absolute Error)                          │             │
│  │   - R² (Coefficient of determination)                  │             │
│  │ • Compute ranking metrics:                             │             │
│  │   - Precision@10                                       │             │
│  │   - Recall@10                                          │             │
│  │ • Compute business metrics:                            │             │
│  │   - Coverage percentage                                │             │
│  │   - Diversity score                                    │             │
│  │ • Generate comprehensive evaluation report             │             │
│  │                                                        │             │
│  │ Output: Evaluation metrics and JSON report             │             │
│  └────────────────────────────────────────────────────────┘             │
│                            ↓                                             │
│  ┌────────────────────────────────────────────────────────┐             │
│  │ STAGE 7: VISUALIZATION                                 │             │
│  │ ─────────────────────────────────────────────────────  │             │
│  │ • Rating distribution histogram with KDE               │             │
│  │ • Top 20 products bar chart                            │             │
│  │ • User activity distribution (dual plot)               │             │
│  │ • User-product interaction heatmap                     │             │
│  │ • Recommendation score distribution                    │             │
│  │ • Evaluation metrics comparison                        │             │
│  │                                                        │             │
│  │ Output: PNG files (300 DPI) in outputs/visualization/  │             │
│  └────────────────────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT LAYER                                    │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │  HDFS Outputs                                              │         │
│  │  hdfs://master:9000/recommendation_results/               │         │
│  │  ├── models/als_model/          # Trained model            │         │
│  │  ├── eda_results/               # Statistical summaries    │         │
│  │  └── recommendations/           # Generated predictions    │         │
│  └────────────────────────────────────────────────────────────┘         │
│                            ↓                                             │
│  ┌────────────────────────────────────────────────────────────┐         │
│  │  Local Outputs                                             │         │
│  │  /home/sameer/BDA_PROJECT_NEW/amazon-recommendation-system│         │
│  │  └── outputs/visualization/                               │         │
│  │      ├── rating_distribution.png                          │         │
│  │      ├── top_products.png                                 │         │
│  │      ├── user_activity.png                                │         │
│  │      ├── interaction_heatmap.png                          │         │
│  │      ├── recommendation_scores.png                        │         │
│  │      └── evaluation_metrics.png                           │         │
│  └────────────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Raw Data → Ingestion → Preprocessing → EDA → Features → Training → Evaluation → Visualization
   ↓           ↓            ↓           ↓       ↓          ↓          ↓           ↓
 HDFS      Validate      Clean     Statistics  Engineer  ALS Model  Metrics     Plots
 6.7GB      Schema      Dedup     Aggregations Features  Predict   RMSE/P@K    Matplotlib
```

## Component Interaction

```
┌──────────────────┐
│  spark_config.yaml│
│  (Configuration)  │
└────────┬──────────┘
         │
         ↓
┌──────────────────┐
│ SparkSessionBuilder│
│  (Session Mgmt)   │
└────────┬──────────┘
         │
         ↓
┌──────────────────┐     ┌──────────────────┐
│  DataIngestion   │────▶│ DataPreprocessing│
│  (HDFS Read)     │     │  (Cleaning)       │
└──────────────────┘     └─────────┬────────┘
                                   │
                                   ↓
┌──────────────────┐     ┌──────────────────┐
│  Visualization   │◀────│ ExploratoryAnalysis│
│  (Plotting)      │     │  (Statistics)     │
└──────────────────┘     └─────────┬────────┘
                                   │
                                   ↓
┌──────────────────┐     ┌──────────────────┐
│  Evaluation      │◀────│ FeatureEngineering│
│  (Metrics)       │     │  (ML Features)    │
└──────────────────┘     └─────────┬────────┘
                                   │
                                   ↓
                          ┌──────────────────┐
                          │RecommendationModel│
                          │  (ALS Training)   │
                          └──────────────────┘
```

## Performance Optimization Layers

```
┌─────────────────────────────────────────────────┐
│ Application Layer                               │
│ • DataFrame caching                             │
│ • Broadcast joins for small tables              │
│ • Predicate pushdown                            │
└─────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────┐
│ Execution Layer                                 │
│ • Adaptive query execution                      │
│ • Dynamic partition pruning                     │
│ • Cost-based optimizer                          │
└─────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────┐
│ Serialization Layer                             │
│ • Kryo serialization(50% smaller than Java)    │
│ • Tungsten binary processing                    │
└─────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────┐
│ Memory Layer                                    │
│ • Off-heap memory management                    │
│ • Columnar in-memory representation             │
│ • Code generation for CPU efficiency            │
└─────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────┐
│ Storage Layer                                   │
│ • HDFS block distribution (128MB blocks)        │
│ • Data locality optimization                    │
│ • Rack-aware replica placement                  │
└─────────────────────────────────────────────────┘
```

This architecture ensures:
✅ Scalability to TB-scale datasets
✅ Fault tolerance through HDFS replication
✅ High performance via in-memory computing
✅ Production reliability with error handling
✅ Maintainable and extensible design
