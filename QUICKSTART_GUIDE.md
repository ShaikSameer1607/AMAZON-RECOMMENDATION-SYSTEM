# 🚀 Quick Start Guide - Distributed Recommendation System

## ⚡ 3-Step Setup & Run

### Step 1: Fix Python Version (CRITICAL!)

Your system has **Python3.13** which is **incompatible with PySpark**.

```bash
# Create conda environment with Python3.10
conda create-n spark_env python=3.10 -y

# Activate it
conda activate spark_env

# Install dependencies
pip install pyspark pyyaml pandas numpy scikit-learn
```

### Step 2: Verify Hadoop is Running

```bash
jps | grep -E "NameNode|DataNode"
# Should show both running
```

If not running:
```bash
$HADOOP_HOME/sbin/start-dfs.sh
```

### Step 3: Run the Pipeline

```bash
cd /home/sameer/BDA_PROJECT_NEW/amazon-recommendation-system

# Method 1: Use runner script (easiest)
./RUN_PIPELINE.sh

# Method 2: Run directly
python3 spark_pipeline/als_recommendation.py
```

---

## 📊 What Happens Next

The pipeline will:

1. ✅ Load 9GB Amazon dataset from HDFS (distributed read)
2. ✅ Preprocess data (remove duplicates, filter invalid ratings)
3. ✅ Convert user/product IDs to numeric indices
4. ✅ Split data 80/20 (train/test)
5. ✅ Train ALS collaborative filtering model (~60-90 seconds)
6. ✅ Calculate RMSE evaluation metric
7. ✅ Generate Top-10 recommendations for 100 users
8. ✅ Save results to HDFS

**Total Time**: ~2-3 minutes

---

## 🔍 Monitor Progress

### During Execution

**Open another terminal and check:**

1. **Spark UI** (Visual progress): http://localhost:4040
2. **YARN applications**: `yarn application-list`

### After Completion

```bash
# View output in HDFS
hdfs dfs -ls /recommendations/

# See sample recommendations
hdfs dfs -cat /recommendations/csv_*/part-00000*.csv | head -20

# Count total recommendations
hdfs dfs -cat /recommendations/csv_*/part-00000*.csv | wc -l
```

Expected: ~1,000 recommendations (10 per user × 100 users)

---

## 🎯 Expected Output

Sample recommendations CSV:

```csv
user_id,recommended_items
A1HGH7NF01K3W7,"[{item_idx: 123, rating: 4.8}, ...]"
A3GDQGSD4C0HOB,"[{item_idx: 456, rating: 4.5}, ...]"
...
```

---

## 🛠️ Troubleshooting

### Problem: "Python3.13 detected" error

**Solution**: You skipped Step 1! Create Python3.10 environment:
```bash
conda create -n spark_env python=3.10 -y
conda activate spark_env
```

### Problem: HDFS safe mode

**Solution**:
```bash
hdfs dfsadmin -safemode leave
```

### Problem: No output in HDFS

**Solution**: Check if pipeline completed successfully. Look for:
```
✅ PIPELINE COMPLETED SUCCESSFULLY!
```

---

## 📁 Files Created

After successful run:

```
outputs/recommendation_results/
├── fast_recommendations_*.csv     # From fast pipeline (pandas)
└── (new folder from Spark pipeline)

HDFS:
/recommendations/
├── YYYYMMDD_HHMMSS/               # Parquet format
└── csv_YYYYMMDD_HHMMSS/           # CSV format
```

---

## 🎓 Key Concepts Demonstrated

✅ HDFS distributed storage  
✅ Spark distributed processing  
✅ Spark MLlib ALS algorithm  
✅ Collaborative filtering  
✅ Model evaluation (RMSE)  
✅ Big Data pipeline architecture  

---

## 📞 Commands Cheat Sheet

```bash
# Activate environment
conda activate spark_env

# Run pipeline
./RUN_PIPELINE.sh

# Check Spark UI
firefox http://localhost:4040

# View HDFS output
hdfs dfs -ls /recommendations/
hdfs dfs -cat /recommendations/csv_*/part-00000*.csv | head -20

# View YARN apps
yarn application -list

# Check logs
tail -f outputs/logs/pipeline.log
```

---

## ✅ Success Checklist

- [ ] Python3.10 environment created
- [ ] Dependencies installed (pyspark, pyyaml)
- [ ] Hadoop running (NameNode, DataNode)
- [ ] Dataset in HDFS (/dataset/all_csv_files.csv)
- [ ] Pipeline executed successfully
- [ ] Spark UI accessible (http://localhost:4040)
- [ ] Recommendations saved to HDFS
- [ ] RMSE logged in output

---

## 🎉 You're Done!

Congratulations! You now have a **production-grade distributed recommendation system** running on Apache Spark and Hadoop HDFS!

For detailed documentation, see: `PROFESSIONAL_PIPELINE.md`

---

**Questions?** Check the comprehensive README: `README_SOLUTION.md`
