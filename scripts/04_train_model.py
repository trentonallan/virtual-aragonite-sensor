import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

print("="*60)
print("ARAGONITE SATURATION PREDICTION MODEL")
print("="*60)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\nLoading data...")
df = pd.read_csv("../data/processed/complete_dataset.csv")
print(f"✓ Loaded {len(df):,} samples")

# Define features
FEATURE_COLS = [
    'salinity',
    'modis_sst',
    'modis_chlora',
    'latitude',
    'longitude',
    'bathymetry_m',
    'depth'
]

TARGET_COL = 'aragonite'

# Clean data
df_model = df[FEATURE_COLS + [TARGET_COL]].dropna()
missing = len(df) - len(df_model)

print(f"\nData summary:")
print(f"  Total samples:   {len(df):,}")
print(f"  Clean samples:   {len(df_model):,}")
print(f"  Missing data:    {missing:,} ({100*missing/len(df):.1f}%)")

# ============================================================================
# 2. TRAIN/VAL/TEST SPLIT
# ============================================================================

X = df_model[FEATURE_COLS].values
y = df_model[TARGET_COL].values

# 60% train, 20% validation, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

print("\n" + "="*60)
print("DATA SPLITS")
print("="*60)
print(f"Training:   {len(X_train):,} samples ({100*len(X_train)/len(X):.1f}%)")
print(f"Validation: {len(X_val):,} samples ({100*len(X_val)/len(X):.1f}%)")
print(f"Test:       {len(X_test):,} samples ({100*len(X_test)/len(X):.1f}%)")

# ============================================================================
# 3. TRAIN MODEL
# ============================================================================

print("\n" + "="*60)
print("TRAINING RANDOM FOREST")
print("="*60)

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

print("Training...")
start_time = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_time

print(f"✓ Training complete in {train_time:.2f}s")

# ============================================================================
# 4. EVALUATE ON VALIDATION SET
# ============================================================================

print("\n" + "="*60)
print("VALIDATION SET PERFORMANCE")
print("="*60)

y_val_pred = model.predict(X_val)
val_r2 = r2_score(y_val, y_val_pred)
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
val_mae = mean_absolute_error(y_val, y_val_pred)

print(f"R²:   {val_r2:.4f}")
print(f"RMSE: {val_rmse:.4f} Omega units")
print(f"MAE:  {val_mae:.4f} Omega units")

# ============================================================================
# 5. EVALUATE ON TEST SET
# ============================================================================

print("\n" + "="*60)
print("TEST SET PERFORMANCE (FINAL)")
print("="*60)

y_test_pred = model.predict(X_test)
test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"R²:   {test_r2:.4f}")
print(f"RMSE: {test_rmse:.4f} Omega units")
print(f"MAE:  {test_mae:.4f} Omega units")
print(f"\nRelative error: {100*test_rmse/y_test.mean():.1f}% of mean")

# ============================================================================
# 6. CROSS-VALIDATION
# ============================================================================

print("\n" + "="*60)
print("CROSS-VALIDATION ANALYSIS")
print("="*60)

print("Running 10-fold CV (5 repeats)...")
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)

start_time = time.time()
cv_scores = cross_val_score(
    model,
    np.vstack([X_train, X_val]),
    np.hstack([y_train, y_val]),
    cv=cv,
    scoring='r2',
    n_jobs=-1
)
cv_time = time.time() - start_time

print(f"✓ Complete in {cv_time:.1f}s\n")
print(f"Cross-validation R² scores (50 folds):")
print(f"  Mean:   {cv_scores.mean():.4f}")
print(f"  Std:    {cv_scores.std():.4f}")
print(f"  Min:    {cv_scores.min():.4f}")
print(f"  Max:    {cv_scores.max():.4f}")
ci_lower = cv_scores.mean() - 1.96*cv_scores.std()
ci_upper = cv_scores.mean() + 1.96*cv_scores.std()
print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================

print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature ranking:")
for i, idx in enumerate(indices, 1):
    bar_length = int(importances[idx] * 50)
    bar = '█' * bar_length
    print(f"  {i}. {FEATURE_COLS[idx]:15s} {importances[idx]:.4f}  {bar}")

# Feature importance plot
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(FEATURE_COLS)))
ax.barh(range(len(FEATURE_COLS)), importances[indices], color=colors)
ax.set_yticks(range(len(FEATURE_COLS)))
ax.set_yticklabels([FEATURE_COLS[i] for i in indices])
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance for Aragonite Saturation Prediction',
             fontsize=14, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('../results/feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: results/feature_importance.png")

# ============================================================================
# 8. PREDICTION VISUALIZATIONS
# ============================================================================

print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

fig = plt.figure(figsize=(16, 5))
gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

# Plot 1: Predictions scatter
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test, y_test_pred, alpha=0.6, s=40, color='steelblue',
            edgecolors='navy', linewidth=0.5)

min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val],
         'r--', lw=2.5, label='Perfect prediction', zorder=10)

ax1.set_xlabel('Actual Omega-arag', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Omega-arag', fontsize=12, fontweight='bold')
ax1.set_title(f'Model Predictions\nR² = {test_r2:.3f}, RMSE = {test_rmse:.3f}',
              fontsize=12, fontweight='bold')
ax1.legend(loc='upper left', frameon=True, shadow=True)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal', adjustable='box')

# Plot 2: Residuals
ax2 = fig.add_subplot(gs[0, 1])
residuals = y_test - y_test_pred
ax2.scatter(y_test_pred, residuals, alpha=0.6, s=40, color='coral',
            edgecolors='darkred', linewidth=0.5)
ax2.axhline(y=0, color='black', linestyle='--', lw=2, label='Zero error')
ax2.set_xlabel('Predicted Omega-arag', fontsize=12, fontweight='bold')
ax2.set_ylabel('Residuals (Actual - Predicted)', fontsize=12, fontweight='bold')
ax2.set_title(f'Residual Analysis\nMean = {residuals.mean():.3f}, Std = {residuals.std():.3f}',
              fontsize=12, fontweight='bold')
ax2.legend(loc='upper left', frameon=True, shadow=True)
ax2.grid(True, alpha=0.3)

# Plot 3: Residual distribution
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(residuals, bins=30, color='skyblue', edgecolor='navy', alpha=0.7)
ax3.axvline(x=0, color='red', linestyle='--', lw=2.5, label='Zero error')
ax3.axvline(x=residuals.mean(), color='green', linestyle='-', lw=2,
            label=f'Mean = {residuals.mean():.3f}')
ax3.set_xlabel('Residuals', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax3.set_title('Residual Distribution', fontsize=12, fontweight='bold')
ax3.legend(frameon=True, shadow=True)
ax3.grid(True, alpha=0.3, axis='y')

plt.savefig('../results/model_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/model_performance.png")

# ============================================================================
# 9. CONFIDENCE SCORE CALCULATION
# ============================================================================

print("\n" + "="*60)
print("CALCULATING CONFIDENCE SCORES")
print("="*60)

print("Analyzing tree agreement...")
tree_predictions = np.array([tree.predict(X_test) for tree in model.estimators_])

pred_std = tree_predictions.std(axis=0)

max_std = pred_std.max()
if max_std > 0:
    confidence_scores = 100 * (1 - pred_std / max_std)
else:
    confidence_scores = np.ones(len(pred_std)) * 100

print(f"\n✓ Confidence scores calculated")
print(f"  Mean confidence: {confidence_scores.mean():.1f}%")
print(f"  Range: {confidence_scores.min():.1f}% - {confidence_scores.max():.1f}%")

# ============================================================================
# 10. PREDICTION EXAMPLES WITH CONFIDENCE
# ============================================================================

print("\n" + "="*60)
print("PREDICTION EXAMPLES")
print("="*60)

sample_indices = np.random.choice(len(y_test), 10, replace=False)

print("\nSample predictions (10 random test samples):")
print(f"{'Actual':>8} {'Predicted':>10} {'Error':>8} {'% Error':>8} {'Confidence':>11}")
print("-" * 60)

for idx in sample_indices:
    actual = y_test[idx]
    predicted = y_test_pred[idx]
    error = actual - predicted
    pct_error = 100 * abs(error) / actual
    conf = confidence_scores[idx]
    print(f"{actual:8.3f} {predicted:10.3f} {error:8.3f} {pct_error:7.1f}% {conf:10.1f}%")

# ============================================================================
# 11. SAVE MODEL WITH CONFIDENCE FUNCTION
# ============================================================================

print("\n" + "="*60)
print("SAVING MODEL AND RESULTS")
print("="*60)

Path("../models").mkdir(exist_ok=True)
Path("../results").mkdir(exist_ok=True)

def predict_with_confidence(model_obj, X_new):
    """
    Make predictions with confidence scores.
    
    Returns: predictions, confidence (both arrays)
    """
    tree_preds = np.array([tree.predict(X_new) for tree in model_obj.estimators_])
    predictions = tree_preds.mean(axis=0)
    pred_std = tree_preds.std(axis=0)
    
    max_std = pred_std.max()
    if max_std > 0:
        confidence = 100 * (1 - pred_std / max_std)
    else:
        confidence = np.ones(len(pred_std)) * 100
    
    return predictions, confidence

model_package = {
    'model': model,
    'feature_cols': FEATURE_COLS,
    'feature_importance': dict(zip(FEATURE_COLS, importances)),
    'predict_with_confidence': predict_with_confidence,
    'performance': {
        'test_r2': float(test_r2),
        'test_rmse': float(test_rmse),
        'test_mae': float(test_mae),
        'cv_r2_mean': float(cv_scores.mean()),
        'cv_r2_std': float(cv_scores.std()),
        'cv_r2_95ci': [float(ci_lower), float(ci_upper)]
    },
    'confidence_info': {
        'mean_confidence': float(confidence_scores.mean()),
        'min_confidence': float(confidence_scores.min()),
        'max_confidence': float(confidence_scores.max())
    },
    'training_info': {
        'n_samples': int(len(df_model)),
        'n_features': int(len(FEATURE_COLS)),
        'train_time_seconds': float(train_time),
        'random_state': 42
    }
}

with open('../models/aragonite_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)
print("✓ Saved: models/aragonite_model.pkl")

# Build report string
feature_list = '\n'.join(f'  - {col}' for col in FEATURE_COLS)
feature_ranking = '\n'.join(
    f'  {i+1}. {FEATURE_COLS[idx]:15s} {importances[idx]:.4f} ({100*importances[idx]:.1f}%)'
    for i, idx in enumerate(indices)
)

sst_idx = list(FEATURE_COLS).index('modis_sst')
chlora_idx = list(FEATURE_COLS).index('modis_chlora')
sat_contribution = 100 * (importances[sst_idx] + importances[chlora_idx])

lat_idx = list(FEATURE_COLS).index('latitude')
lon_idx = list(FEATURE_COLS).index('longitude')
geo_contribution = 100 * (importances[lat_idx] + importances[lon_idx])

bathy_idx = list(FEATURE_COLS).index('bathymetry_m')
bathy_contribution = 100 * importances[bathy_idx]

timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

report = """ARAGONITE SATURATION STATE PREDICTION MODEL
======================================================================

DATASET INFORMATION
-------------------
Total samples:              """ + f"{len(df):,}" + """
Clean samples (no missing): """ + f"{len(df_model):,}" + """
Missing/dropped:            """ + f"{missing:,} ({100*missing/len(df):.1f}%)" + """

Features (""" + str(len(FEATURE_COLS)) + """):
""" + feature_list + """

Target variable: """ + TARGET_COL + """ (Omega-arag)
  Range: """ + f"{y.min():.2f} - {y.max():.2f}" + """
  Mean:  """ + f"{y.mean():.2f} ± {y.std():.2f}" + """

DATA SPLITS
-----------
Training:   """ + f"{len(X_train):,} samples ({100*len(X_train)/len(X):.1f}%)" + """
Validation: """ + f"{len(X_val):,} samples ({100*len(X_val)/len(X):.1f}%)" + """
Test:       """ + f"{len(X_test):,} samples ({100*len(X_test)/len(X):.1f}%)" + """

MODEL ARCHITECTURE
------------------
Algorithm: Random Forest Regressor
Hyperparameters:
  - n_estimators: 100
  - max_depth: 15
  - min_samples_split: 5
  - min_samples_leaf: 2
  - max_features: sqrt
  - random_state: 42

Training time: """ + f"{train_time:.2f} seconds" + """

PERFORMANCE METRICS
-------------------
Test Set (n=""" + str(len(y_test)) + """):
  R² Score:  """ + f"{test_r2:.4f}" + """
  RMSE:      """ + f"{test_rmse:.4f} Omega units ({100*test_rmse/y_test.mean():.1f}% of mean)" + """
  MAE:       """ + f"{test_mae:.4f} Omega units" + """

Cross-Validation (10-fold, 5 repeats = 50 splits):
  Mean R²:   """ + f"{cv_scores.mean():.4f}" + """
  Std Dev:   """ + f"{cv_scores.std():.4f}" + """
  Range:     """ + f"{cv_scores.min():.4f} - {cv_scores.max():.4f}" + """
  95% CI:    [""" + f"{ci_lower:.4f}, {ci_upper:.4f}" + """]

CONFIDENCE SCORES
-----------------
Mean confidence: """ + f"{confidence_scores.mean():.1f}%" + """
Range: """ + f"{confidence_scores.min():.1f}% - {confidence_scores.max():.1f}%" + """

FEATURE IMPORTANCE
------------------
""" + feature_ranking + """

INTERPRETATION
--------------
The model explains """ + f"{100*test_r2:.1f}%" + """ of variance in aragonite saturation state.
Predictions are accurate to within ±""" + f"{test_rmse:.2f}" + """ Omega units on average.

Key findings:
- Salinity is the dominant predictor (""" + f"{100*importances[indices[0]]:.1f}%" + """ importance)
- Satellite-derived features (SST, chlorophyll) contribute """ + f"{sat_contribution:.1f}%" + """
- Geographic location (lat/lon) captures """ + f"{geo_contribution:.1f}%" + """ of variance
- Bathymetry adds """ + f"{bathy_contribution:.1f}%" + """ predictive power

MODEL LIMITATIONS
-----------------
- Trained on """ + f"{len(df_model):,}" + """ samples (performance may improve with more data)
- Cross-validation shows ±""" + f"{cv_scores.std():.3f}" + """ variance in R² (small dataset effect)
- Requires satellite data (clouds may limit coverage)
- Best for surface waters (0-10m depth)

USAGE
-----
To make predictions with confidence scores:

    import pickle
    import numpy as np
    
    with open('aragonite_model.pkl', 'rb') as f:
        package = pickle.load(f)
    
    model = package['model']
    predict_fn = package['predict_with_confidence']
    
    new_data = np.array([[35.0, 26.5, 0.15, 20.5, -155.2, 2500, 5.0]])
    predictions, confidence = predict_fn(model, new_data)
    
    print(f"Prediction: {predictions[0]:.2f} Omega")
    print(f"Confidence: {confidence[0]:.1f}%")

Generated: """ + timestamp + """
"""

with open('../results/model_report.txt', 'w') as f:
    f.write(report)
print("✓ Saved: results/model_report.txt")

metrics_df = pd.DataFrame({
    'Metric': ['R²', 'RMSE', 'MAE', 'CV_R²_mean', 'CV_R²_std', 'Mean_Confidence'],
    'Value': [test_r2, test_rmse, test_mae, cv_scores.mean(), cv_scores.std(), confidence_scores.mean()],
    'Unit': ['', 'Omega', 'Omega', '', '', '%']
})
metrics_df.to_csv('../results/performance_metrics.csv', index=False)
print("✓ Saved: results/performance_metrics.csv")

# ============================================================================
# 12. FINAL SUMMARY
# ============================================================================

print("\n" + "="*60)
print("✓ TRAINING COMPLETE!")
print("="*60)

print(f"\nFinal Model Performance:")
print(f"  Test R²:   {test_r2:.4f}")
print(f"  Test RMSE: {test_rmse:.4f} Omega units")
print(f"  CV R²:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Mean Confidence: {confidence_scores.mean():.1f}%")

print(f"\nTop 3 Features:")
for i in range(3):
    idx = indices[i]
    print(f"  {i+1}. {FEATURE_COLS[idx]:15s} ({100*importances[idx]:.1f}%)")

print(f"\nFiles saved:")
print(f"  models/aragonite_model.pkl")
print(f"  results/feature_importance.png")
print(f"  results/model_performance.png")
print(f"  results/model_report.txt")
print(f"  results/performance_metrics.csv")

print("\n" + "="*60)