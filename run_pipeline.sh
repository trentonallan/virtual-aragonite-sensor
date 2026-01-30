#!/bin/bash

# Virtual Aragonite Sensor - Full Training Pipeline
# Runs all data processing and model training steps

set -e  # Exit on any error

echo "============================================"
echo "Virtual Aragonite Sensor Training Pipeline"
echo "============================================"
echo ""

# Create directories
echo "Creating directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p results

# Step 1: Download GLODAP data
echo ""
echo "Step 1/4: Processing GLODAP ocean chemistry data..."
python scripts/01_download_glodap.py

# Step 2: Add bathymetry
echo ""
echo "Step 2/4: Adding bathymetry data..."
python scripts/02_download_bathymetry.py

# Step 3: Extract satellite data
echo ""
echo "Step 3/4: Extracting satellite features from Google Earth Engine..."
python scripts/03_extract_satellite_gee.py

# Step 4: Train model
echo ""
echo "Step 4/4: Training model and generating visualizations..."
python scripts/04_train_model.py

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "============================================"
echo "Model saved to: models/aragonite_model.pkl"
echo "Visualizations saved to: results/"