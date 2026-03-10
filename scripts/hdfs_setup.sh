#!/bin/bash

###############################################################################
# HDFS Directory Setup Script
# Amazon Product Recommendation System
#
# This script creates all necessary HDFS directories for the pipeline
###############################################################################

set -e  # Exit on error

echo "=========================================="
echo "Setting up HDFS Directories"
echo "=========================================="

# Define HDFS base path
HDFS_BASE="hdfs://master:9000/recommendation_results"

# Create directories
echo "Creating HDFS directories..."

# Main output directory
hdfs dfs -mkdir -p ${HDFS_BASE} 2>/dev/null || echo "Base directory exists"

# EDA results directory
hdfs dfs -mkdir -p ${HDFS_BASE}/eda_results 2>/dev/null || echo "EDA directory exists"

# Models directory
hdfs dfs -mkdir -p ${HDFS_BASE}/models 2>/dev/null || echo "Models directory exists"

# Recommendations directory
hdfs dfs -mkdir -p ${HDFS_BASE}/recommendations 2>/dev/null || echo "Recommendations directory exists"

# Visualization outputs (local directory)
VIS_DIR="/home/sameer/BDA_PROJECT_NEW/amazon-recommendation-system/outputs/visualization"
mkdir -p ${VIS_DIR} 2>/dev/null || echo "Visualization directory exists"

# Verify dataset exists in HDFS
DATASET_PATH="hdfs://localhost:9000/dataset/all_csv_files.csv"
echo ""
echo "Checking if dataset exists at: ${DATASET_PATH}"

if hdfs dfs -test -e ${DATASET_PATH}; then
    echo "✓ Dataset found in HDFS"
    DATASET_SIZE=$(hdfs dfs -du -h ${DATASET_PATH} | awk '{print $1}')
    echo "  Size: ${DATASET_SIZE}"
else
    echo "✗ ERROR: Dataset not found at ${DATASET_PATH}"
    echo "Please ensure the dataset is uploaded to HDFS before running the pipeline"
    exit 1
fi

echo ""
echo "=========================================="
echo "HDFS Setup Complete!"
echo "=========================================="
echo ""
echo "Directory Structure:"
echo "  ${HDFS_BASE}/"
echo "  ├── eda_results/"
echo "  ├── models/"
echo "  └── recommendations/"
echo ""
echo "Local Directories:"
echo "  ${VIS_DIR}/"
echo ""
