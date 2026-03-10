#!/bin/bash

###############################################################################
# Main Pipeline Execution Script
# Amazon Product Recommendation System
#
# This script executes the complete recommendation system pipeline:
# 1. Data Ingestion from HDFS
# 2. Data Preprocessing
# 3. Exploratory Data Analysis
# 4. Feature Engineering
# 5. ALS Model Training
# 6. Model Evaluation
# 7. Visualization
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="/home/sameer/BDA_PROJECT_NEW/amazon-recommendation-system"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
PIPELINE_DIR="${PROJECT_DIR}/spark_pipeline"

echo ""
echo "=========================================="
echo "Amazon Product Recommendation System"
echo "Big Data Pipeline Execution"
echo "=========================================="
echo ""

# Step 0: Setup HDFS directories
echo -e "${BLUE}[Step 0/7] Setting up HDFS directories...${NC}"
echo ""
bash ${SCRIPTS_DIR}/hdfs_setup.sh
echo ""

# Step 1: Data Ingestion
echo -e "${BLUE}[Step 1/7] Data Ingestion from HDFS...${NC}"
echo ""
python3 ${PIPELINE_DIR}/data_ingestion.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Error in data ingestion${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Data ingestion complete${NC}"
echo ""

# Step 2: Data Preprocessing
echo -e "${BLUE}[Step 2/7] Data Preprocessing...${NC}"
echo ""
python3 ${PIPELINE_DIR}/data_preprocessing.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Error in data preprocessing${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Data preprocessing complete${NC}"
echo ""

# Step 3: Exploratory Data Analysis
echo -e "${BLUE}[Step 3/7] Exploratory Data Analysis...${NC}"
echo ""
python3 ${PIPELINE_DIR}/exploratory_analysis.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Error in exploratory analysis${NC}"
    exit 1
fi
echo -e "${GREEN}✓ EDA complete${NC}"
echo ""

# Step 4: Feature Engineering
echo -e "${BLUE}[Step 4/7] Feature Engineering...${NC}"
echo ""
python3 ${PIPELINE_DIR}/feature_engineering.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Error in feature engineering${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Feature engineering complete${NC}"
echo ""

# Step 5: Model Training
echo -e "${BLUE}[Step 5/7] ALS Model Training...${NC}"
echo ""
python3 ${PIPELINE_DIR}/recommendation_model.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Error in model training${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Model training complete${NC}"
echo ""

# Step 6: Model Evaluation
echo -e "${BLUE}[Step 6/7] Model Evaluation...${NC}"
echo ""
python3 ${PIPELINE_DIR}/evaluation.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Error in model evaluation${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Model evaluation complete${NC}"
echo ""

# Step 7: Visualization
echo -e "${BLUE}[Step 7/7] Generating Visualizations...${NC}"
echo ""
python3 ${PIPELINE_DIR}/visualization.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Error in visualization${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Visualizations complete${NC}"
echo ""

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}✓ PIPELINE EXECUTION COMPLETE!${NC}"
echo "=========================================="
echo ""
echo "Output Locations:"
echo "  Models: hdfs://master:9000/recommendation_results/models/"
echo "  Recommendations: hdfs://master:9000/recommendation_results/recommendations/"
echo "  EDA Results: hdfs://master:9000/recommendation_results/eda_results/"
echo "  Visualizations: ${PROJECT_DIR}/outputs/visualization/"
echo ""
echo "=========================================="
echo ""
