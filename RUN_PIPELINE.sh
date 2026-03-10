#\!/bin/bash

###############################################################################
# Amazon Product Recommendation System - Distributed Pipeline Runner
# 
# This script executes the professional distributed recommendation pipeline
# using Apache Spark MLlib ALS on Hadoop HDFS.
###############################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "=============================================="
echo "Amazon Product Recommendation System"
echo "Distributed Pipeline (Spark MLlib + HDFS)"
echo "=============================================="
echo ""

# Step 1: Check Python version
echo -e "${BLUE}[1/5] Checking Python environment...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" == "3.13"* ]]; then
    echo -e "${YELLOW}⚠️  WARNING: PySpark is incompatible with Python3.13\!${NC}"
    echo ""
    echo "Please use one of these options:"
    echo ""
    echo "Option 1: Create conda environment with Python3.10"
    echo "  conda create-n spark_env python=3.10 -y"
    echo "  conda activate spark_env"
    echo ""
    echo "Option 2: Use system Python3.10 if available"
    echo "  python3.10 spark_pipeline/als_recommendation.py"
    echo ""
    exit 1
fi

# Step 2: Check HDFS
echo -e "${BLUE}[2/5] Checking HDFS status...${NC}"
jps | grep -q NameNode && echo "✓ NameNode running" || { echo -e "${RED}✗ NameNode not running${NC}"; exit 1; }
jps | grep -q DataNode && echo "✓ DataNode running" || { echo -e "${RED}✗ DataNode not running${NC}"; exit 1; }

# Step 3: Check dataset in HDFS
echo -e "${BLUE}[3/5] Checking dataset in HDFS...${NC}"
if hdfs dfs -test -e /dataset/all_csv_files.csv; then
    SIZE=$(hdfs dfs -stat "%s" /dataset/all_csv_files.csv)
    SIZE_GB=$((SIZE / 1073741824))
    echo "✓ Dataset found: ${SIZE_GB}GB"
else
    echo -e "${RED}✗ Dataset not found in HDFS${NC}"
    exit 1
fi

# Step 4: Run pipeline
echo -e "${BLUE}[4/5] Starting distributed recommendation pipeline...${NC}"
echo ""
echo "This will:"
echo "  1. Load data from HDFS (distributed read)"
echo "  2. Preprocess data (remove duplicates, filter)"
echo "  3. Engineer features (StringIndexer)"
echo "  4. Train ALS collaborative filtering model"
echo "  5. Evaluate model (RMSE)"
echo "  6. Generate Top-10 recommendations per user"
echo "  7. Save results to HDFS"
echo ""
echo "Spark UI will be available at: http://localhost:4040"
echo ""
echo -e "${YELLOW}Starting pipeline...${NC}"
echo ""

cd "$(dirname "$0")"
python3 spark_pipeline/als_recommendation.py

PIPELINE_STATUS=$?

# Step 5: Show results
echo ""
echo -e "${BLUE}[5/5] Pipeline execution complete\!${NC}"
echo ""

if [ $PIPELINE_STATUS -eq 0 ]; then
    echo -e "${GREEN}=============================================="
    echo "✅ PIPELINE COMPLETED SUCCESSFULLY\!"
    echo "==============================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. View Spark UI: http://localhost:4040"
    echo "  2. Check HDFS output:"
    echo "    hdfs dfs -ls /recommendations/"
    echo "    hdfs dfs -cat /recommendations/csv_*/part-00000*.csv | head -20"
    echo "  3. View logs: cat outputs/logs/pipeline.log"
    echo ""
else
    echo -e "${RED}=============================================="
    echo "❌ PIPELINE FAILED"
    echo "==============================================${NC}"
    echo ""
    echo "Check logs for details:"
    echo "  tail -f outputs/logs/pipeline.log"
    echo ""
    exit 1
fi

echo "=============================================="
echo ""
