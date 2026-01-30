# Virtual Aragonite Sensor: ML-Based Ocean Acidification Monitoring

A Random Forest model that predicts aragonite saturation state (coral reef health indicator) from satellite data, enabling conservation organizations to screen potential restoration sites at zero cost.

## Table of Contents
- [Problem](#problem)
- [Solution](#solution)
- [Results](#results)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Performance Analysis](#performance-analysis)
- [Scientific Context](#scientific-context)
- [References](#references)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Author](#author)

## Problem

Ocean acidification threatens coral reefs by reducing aragonite saturation (Ωarag), but measuring it requires expensive research cruises ($10,000+/day). This creates a critical bottleneck for coral restoration projects that need to identify suitable planting sites.

## Solution

An ML pipeline that predicts Ωarag from freely available satellite data (sea surface temperature, chlorophyll-a), enabling global ocean chemistry screening at zero marginal cost.

## Results
 
- **R² = 0.80** on held-out test data
- **RMSE = 0.39 Ω units** (9% relative error)
- **MAE = 0.23 Ω units** (mean absolute error)
- **Cross-validation**: 0.80 ± 0.04 R² (50-fold repeated CV)

## Methodology

### Data Sources
- **Chemistry**: GLODAP v2.2023 (ocean carbon measurements)
  - 2,140 samples after quality control
  - Temporal coverage: 2000-2021 (21 years)
  - Spatial coverage: 60°S to 60°N
- **Satellite**: MODIS-Aqua (NASA, 4km resolution)
  - Sea Surface Temperature (SST)
  - Chlorophyll-a concentration
- **Bathymetry**: ETOPO1 (seafloor depth)

### Model Architecture

I used a Random Forest (100 trees, max depth 15) because it handles small datasets well and gives built-in feature importance without tuning. The model splits data 60/20/20 for training/validation/test and uses 10-fold cross-validation to verify performance.

The forest approach beat neural networks here because:
- More robust on 2,140 samples
- Built-in feature rankings
- Confidence scores from tree variance

Model includes a `predict_with_confidence()` function that returns both predictions and uncertainty estimates based on inter-tree variance, enabling users to identify low-confidence predictions that may warrant in-situ validation.

### Input Features
1. Sea Surface Temperature (MODIS SST)
2. Chlorophyll-a concentration (MODIS)
3. Salinity (in-situ)
4. Bathymetry (seafloor depth)
5. Latitude
6. Longitude
7. Sample depth

## Project Structure
```
virtual-aragonite-sensor/
├── data/                 # Created by training pipeline*
│   ├── raw/              # Downloaded datasets
│   └── processed/        # Cleaned, merged data
├── scripts/
│   ├── 01_download_glodap.py           # Get ocean chemistry data
│   ├── 02_download_bathymetry.py       # Get bathymetry data
│   ├── 03_extract_satellite_gee.py     # Extract satellite features
│   └── 04_train_model.py               # Train model
├── models/
│   └── aragonite_model.pkl       # Trained model
├── visualizations/
│   ├── cv_distribution.png
│   ├── error_distribution.png
│   ├── feature_importance.png
│   ├── metrics_comparison.png
│   ├── prediction_uncertainty.png
│   ├── residuals.png
│   └── sample_distribution.png
├── .gitignore
├── README.md
├── requirements.txt          
├── run_pipeline.bat      # Windows pipeline runner
└── run_pipeline.sh       # Mac/Linux pipeline runner
```

**Note: data/ directory is not included in the repository. It will be automatically created when running the training pipeline.*

## Usage

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/trentonallan/virtual-aragonite-sensor.git
cd virtual-aragonite-sensor

# Install dependencies
pip install -r requirements.txt

# Authenticate with Google Earth Engine (only needed if retraining from scratch)
earthengine authenticate
```

### Option 1: Use Pre-trained Model

The repository includes a pre-trained model, so you can make predictions immediately without downloading any data:
```python
import pickle
import numpy as np

# Load trained model
with open('models/aragonite_model.pkl', 'rb') as f:
    package = pickle.load(f)

model = package['model']
predict_fn = package['predict_with_confidence']

# Input: [salinity, sst, chlor_a, latitude, longitude, bathymetry_m, depth]
# Note: Salinity is not available from satellites. Must come from in-situ measurements or climatological data (ex: World Ocean Atlas) 
new_data = np.array([[35.0, 26.5, 0.15, 20.5, -155.2, -2500, 5.0]])

# Get prediction with confidence score
predictions, confidence = predict_fn(model, new_data)

print(f"Predicted Ωarag: {predictions[0]:.2f}")
print(f"Confidence: {confidence[0]:.1f}%")
```

### Option 2: Train From Scratch

#### Quick Start - Full Pipeline

Run the complete pipeline to download data and train the model:
```bash
# Mac/Linux
./run_pipeline.sh

# Windows
run_pipeline.bat
```

The pipeline will:
1. Download and process GLODAP ocean chemistry data (~3,000 samples)
2. Add ETOPO1 bathymetry data (~400MB download)
3. Extract MODIS satellite features via Google Earth Engine (10-60 minutes)
4. Train Random Forest model

**Note:** Step 3 (satellite extraction) requires Google Earth Engine authentication and may take 10-60 minutes depending on network speed.

#### Manual Training Pipeline

If you prefer to run steps individually:
```bash
# 1. Process ocean chemistry data
python scripts/01_download_glodap.py

# 2. Add bathymetry data
python scripts/02_download_bathymetry.py

# 3. Extract satellite data (requires GEE authentication)
python scripts/03_extract_satellite_gee.py

# 4. Train model
python scripts/04_train_model.py
```

## Performance Analysis

<img src="visualizations/predictions_vs_actual.png" width="450" align="right">

### Model Metrics

- **Validation R²**: 0.743
- **Test R²**: 0.801
- **Validation RMSE**: 0.443 Ω units
- **Test RMSE**: 0.387 Ω units
- **Test MAE**: 0.233 Ω units
- **Relative error**: 9.0% of mean Ωarag value

<br clear="right"/>




<p align="center">
  <img src="visualizations/metrics_comparison.png" width="800">
</p>


### Cross-Validation Results
10-fold cross-validation with 5 repeats (50 total folds) shows consistent performance:
- **Mean R²**: 0.8025
- **Standard deviation**: 0.0411
- **Range**: 0.6947 - 0.8620
- **95% CI**: [0.7220, 0.8830]

<p align="center">
  <img src="visualizations/cv_distribution.png" width="600">
</p>

The low standard deviation (0.04) indicates consistent performance across different data subsets, confirming model stability.

### Feature Importance
Analysis via Random Forest feature importance reveals:

<p align="center">
  <img src="visualizations/feature_importance.png" width="700">
</p>

1. **Salinity** (38.5% importance) - Proxy for total alkalinity, which buffers against pH changes
2. **SST** (18.2%) - Controls CO₂ solubility in seawater (colder water holds more CO₂)
3. **Chlorophyll-a** (13.2%) - Indicates biological CO₂ uptake by phytoplankton
4. **Latitude** (11.4%) - Captures latitudinal temperature/chemistry gradients
5. **Longitude** (9.7%) - Regional oceanographic patterns
6. **Bathymetry** (7.2%) - Proxies for upwelling and mixing dynamics
7. **Depth** (1.8%) - Minor influence within surface sampling range

These satellite-derived features (SST, chlorophyll) and in-situ measurements (salinity) serve as proxies for the underlying carbonate chemistry that determines aragonite saturation.

Salinity's dominance makes sense because it directly controls carbonate ion concentration, which is what corals need to build their skeletons.

### Error Analysis

<p align="center">
  <img src="visualizations/residuals.png" width="600">
</p>

Residual plot shows random scatter around zero with no systematic bias, confirming model assumptions.

<p align="center">
  <img src="visualizations/error_distribution.png" width="600">
</p>

Error distribution is approximately normal with mean near zero, validating statistical assumptions.

### Practical Accuracy
- Reliably distinguishes excellent sites (Ω > 3.5) from poor ones (Ω < 2.5) - sufficient for initial screening
- RMSE of 0.39 is comparable to measurement uncertainty (~0.2)
- Enables rapid site prioritization before committing to expensive field validation
- 9% error relative to mean values - good enough to eliminate obviously unsuitable locations

**Use case**: Screen 50 potential restoration sites using satellite data → narrow to 5 promising candidates → conduct detailed in-situ measurements at those 5 locations. This inverts the typical workflow where organizations can only afford to measure 5 sites total.

### Limitations
- **Requires salinity data**: Not available from satellites. Users must provide either in-situ measurements or climatological estimates (ex: World Ocean Atlas)
- Only validated on surface waters (0-10m depth)
- Performance may degrade in extreme environments (hypersaline areas, brackish water)
- Training data biased toward well-studied regions

<details>
<summary><b>Additional Visualizations</b></summary>

#### Prediction Uncertainty Analysis
<p align="center">
  <img src="visualizations/prediction_uncertainty.png" width="700">
</p>

#### Global Sample Distribution
<p align="center">
  <img src="visualizations/sample_distribution.png" width="800">
</p>

</details>

## Scientific Context

### Ocean Acidification & Coral Reefs

When atmospheric CO₂ dissolves in seawater, it forms carbonic acid (H₂CO₃), which dissociates and reduces the availability of carbonate ions (CO₃²⁻):
```
CO₂ + H₂O ⇌ H₂CO₃ ⇌ H⁺ + HCO₃⁻ ⇌ 2H⁺ + CO₃²⁻
```

This matters because corals build their skeletons from aragonite (a form of calcium carbonate, CaCO₃), which requires carbonate ions:
```
Ca²⁺ + CO₃²⁻ → CaCO₃ (aragonite)
```

### Aragonite Saturation State (Ωarag)

The aragonite saturation state quantifies whether seawater chemistry favors aragonite formation or dissolution:
```
Ωarag = [Ca²⁺][CO₃²⁻] / Ksp
```

Where:
- **[Ca²⁺]** and **[CO₃²⁻]** are the actual concentrations of calcium and carbonate ions in seawater
- **Ksp** is the solubility product constant - the theoretical ion concentration product at which aragonite would be in equilibrium (neither forming nor dissolving)

**Interpreting Ω:**
- **Ω > 1**: Seawater is *supersaturated* - aragonite formation is thermodynamically favorable
- **Ω = 1**: Seawater is *saturated* - equilibrium (no net formation or dissolution)
- **Ω < 1**: Seawater is *undersaturated* - existing aragonite will dissolve

Ksp increases with depth (higher pressure) and decreases with temperature. This is why cold, deep water dissolves aragonite more easily than warm surface water.

**Biological Thresholds:**
- **Ω > 3.5**: Excellent coral growth and calcification
- **Ω = 3.0-3.5**: Good conditions, healthy reefs
- **Ω = 2.5-3.0**: Marginal conditions, reduced growth
- **Ω < 2.5**: Stressed corals, increased mortality risk
- **Ω < 1.0**: Undersaturated - net dissolution of existing structures

### Why This Matters

Ocean pH has dropped by 0.1 units since pre-industrial times (a 30% increase in acidity) due to rising atmospheric CO₂, and projections suggest 50% of coral reefs could be lost by 2050 (IPCC).

Coral restoration organizations typically face a brutal economic trade-off: site assessment requires expensive research cruises ($10k+/day), but they need to evaluate dozens of potential locations before committing $100k-$1M to restoration work.

This tool doesn't replace direct ocean chemistry measurements - those are the gold standard. Instead, it enables applications that weren't economically feasible before:
- **Global screening**: Survey 10× more sites in initial assessment
- **Continuous monitoring**: Track changes over time without repeated cruises  
- **Rapid spatial mapping**: Identify promising regions across entire coastlines

Conservation organizations can now screen potential sites at zero cost, then direct their limited budget toward precise measurements at the most promising locations. In an acidifying ocean where 50% of reefs may be lost by 2050, being able to identify sites where corals are most likely to survive is critical.

## References

1. IPCC (2021) - Climate Change 2021: The Physical Science Basis
2. Feely et al. (2004) - Ocean acidification impact on CaCO₃
3. GLODAP v2.2023 - Global ocean carbon database
4. MODIS-Aqua - NASA ocean color mission
5. ETOPO1 - NOAA global relief model
6. Mucci (1983) - Aragonite solubility in seawater

## Acknowledgments

Built using GLODAP ocean chemistry data (NOAA), MODIS satellite imagery (NASA), and Google Earth Engine's processing infrastructure.

## License

MIT License - Free to use for research and conservation

## Author

**Trenton Allan**  
Northeastern University
B.S. Computer Science (AI concentration) | [allan.tr@northeastern.edu](mailto:allan.tr@northeastern.edu) | [LinkedIn](https://linkedin.com/in/trentonallan)
