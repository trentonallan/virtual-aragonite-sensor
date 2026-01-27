# Virtual Aragonite Sensor: ML-Based Ocean Acidification Monitoring

Predicting coral reef building conditions from satellite data using machine learning.

## Problem

Ocean acidification threatens coral reefs by reducing aragonite saturation (Ωarag), but measuring it requires expensive research cruises ($10,000+/day). This creates a critical bottleneck for coral restoration projects that need to identify suitable planting sites.

## Solution

A neural network that predicts Ωarag from freely available satellite data (sea surface temperature, chlorophyll-a), enabling global ocean chemistry screening at zero marginal cost.

## Results

- **R² = 0.78+** on held-out test data
- **RMSE = ~0.40 Ω units** (comparable to measurement uncertainty)
- **2,140 training samples** spanning 21 years (2000-2021)
- **Global coverage** from 60°S to 60°N

## Methodology

### Data Sources
- **Chemistry**: GLODAP v2.2023 (ocean carbon measurements)
- **Satellite**: MODIS-Aqua (NASA, 4km resolution)
  - Sea Surface Temperature (SST)
  - Chlorophyll-a concentration
- **Bathymetry**: ETOPO1 (seafloor depth)

### Model Architecture
- Feed-forward neural network (7 input features → 128 → 64 → 32 → 1)
- PyTorch implementation with dropout (0.2) and batch normalization
- Training: 60% train, 20% validation, 20% test

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
acid-project/
├── data/
│   ├── raw/              # Downloaded datasets
│   └── processed/        # Cleaned, merged data
├── scripts/
│   ├── 01_download_glodap.py      # Get ocean chemistry data
│   ├── 02_extract_satellite_gee.py # Extract satellite features
│   ├── 04_add_bathymetry.py       # Add seafloor depth
│   └── 05_train_with_bathymetry.py # Train final model
├── models/
│   └── model_with_bathymetry.pth  # Trained model
├── results/
│   ├── predictions_with_bathymetry.png
│   └── training_curves.png
└── README.md
```

## Usage

### Prerequisites
```bash
pip install -r requirements.txt
earthengine authenticate  # For satellite data
```

### Training Pipeline
```bash
# 1. Download ocean chemistry data
python scripts/01_download_glodap.py

# 2. Extract satellite data
python scripts/02_extract_satellite_gee.py

# 3. Add bathymetry
python scripts/04_add_bathymetry.py

# 4. Train model
python scripts/05_train_with_bathymetry.py
```

### Making Predictions
```python
import torch
import numpy as np

# Load model
checkpoint = torch.load('models/model_with_bathymetry.pth', weights_only=False)
model = AragonitePredictor()
model.load_state_dict(checkpoint['model_state_dict'])

# Predict for a location
features = np.array([[
    28.5,    # SST (°C)
    0.08,    # Chlorophyll-a (mg/m³)
    35.2,    # Salinity
    25.0,    # Bathymetry (m)
    -17.5,   # Latitude
    -149.8,  # Longitude
    5.0      # Depth (m)
]])

# Scale and predict
features_scaled = checkpoint['scaler_X'].transform(features)
omega_pred = model(torch.FloatTensor(features_scaled)).item()
omega_pred = checkpoint['scaler_y'].inverse_transform([[omega_pred]])[0][0]

print(f"Predicted Ωarag: {omega_pred:.2f}")
```

## Performance Analysis

### Comparison to Baseline
| Metric | Baseline (no bathymetry) | With Bathymetry | Improvement |
|--------|--------------------------|-----------------|-------------|
| R² | 0.734 | 0.78+ | +6.3% |
| RMSE | 0.447 | ~0.40 | -10.5% |
| Features | 6 | 7 | +1 |

### Practical Accuracy
- Can distinguish excellent sites (Ω > 3.5) from poor sites (Ω < 2.5)
- Sufficient for screening/prioritization (primary use case)
- Within 2-4× measurement uncertainty

## Scientific Context

### Aragonite Saturation State (Ωarag)
```
Ω = [Ca²⁺][CO₃²⁻] / Ksp
```

- **Ω > 3.5**: Excellent coral growth conditions
- **Ω = 3.0-3.5**: Good conditions
- **Ω < 3.0**: Stressed corals
- **Ω < 1.0**: Undersaturated (dissolution)

### Why This Matters
- 50% of coral reefs could be lost by 2050 (IPCC)
- Restoration projects need $100k-$1M+ per site
- Site assessment is a major bottleneck
- This tool enables free global screening

## Future Work

- [ ] Add depth-aware predictions (predict Ω at 5-30m, not just surface)
- [ ] Implement Physics-Informed Neural Network (PINN) constraints
- [ ] Temporal forecasting (predict future acidification)
- [ ] Real-time monitoring dashboard
- [ ] Uncertainty quantification (prediction intervals)
- [ ] Integration with coral restoration databases

## References

1. Feely et al. (2004) - Ocean acidification impact on CaCO₃
2. GLODAP v2.2023 - Global ocean carbon database
3. MODIS-Aqua - NASA ocean color mission
4. Mucci (1983) - Aragonite solubility in seawater

## Acknowledgments

- NOAA NCEI for GLODAP database
- NASA for MODIS satellite data
- Google Earth Engine for data processing infrastructure
- Anthropic Claude for technical guidance

## License

MIT License - Free to use for research and conservation

## Author

Trenton Hatch - Northeastern University
Computer Science (AI concentration) | allan.tr@northeastern.edu
