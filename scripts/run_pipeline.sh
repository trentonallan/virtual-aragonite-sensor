#!/bin/bash

echo "=========================================="
echo "CORAL ARAGONITE SATURATION ML PIPELINE"
echo "=========================================="
echo ""

# Exit on any error
set -e

# Step 1: Process GLODAP data
echo "STEP 1: Processing GLODAP surface data..."
python3 01_download_glodap.py
if [ $? -ne 0 ]; then
    echo "ERROR: GLODAP processing failed"
    exit 1
fi
echo ""

# Step 2: Add bathymetry
echo "STEP 2: Adding bathymetry data..."
python3 02_download_bathymetry.py
if [ $? -ne 0 ]; then
    echo "ERROR: Bathymetry extraction failed"
    exit 1
fi
echo ""

# Step 3: Extract satellite data (this takes ~10-15 minutes)
echo "STEP 3: Extracting satellite data from Google Earth Engine..."
echo "WARNING: This step takes 10-15 minutes"
python3 03_extract_satellite_gee.py
if [ $? -ne 0 ]; then
    echo "ERROR: Satellite extraction failed"
    exit 1
fi
echo ""

# Step 4: Train model
echo "STEP 4: Training neural network..."
python3 04_train_model.py
if [ $? -ne 0 ]; then
    echo "ERROR: Model training failed"
    exit 1
fi
echo ""

echo "=========================================="
echo "✓ PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - learning-data/processed/glodap_surface.csv"
echo "  - learning-data/processed/glodap_with_bathymetry.csv"
echo "  - learning-data/processed/glodap_with_bathymetry_and_satellite_clean.csv"
echo "  - learning-data/processed/complete_dataset.csv"
echo "  - learning-data/models/model_with_bathymetry.pth"
echo "  - learning-data/results/predictions_with_bathymetry.png"
echo ""