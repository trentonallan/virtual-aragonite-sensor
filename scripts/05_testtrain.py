import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import time

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

print("="*60)
print("TRAINING ARAGONITE SATURATION MODEL")
print("="*60)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

df = pd.read_csv("../data/processed/complete_dataset.csv")
print(f"\nLoaded {len(df):,} samples")

# Define features
FEATURE_COLS = [
    'modis_sst',        # Satellite sea surface temperature
    'modis_chlora',     # Satellite chlorophyll-a
    'salinity',         # In-situ salinity
    'bathymetry_m',     # Seafloor depth
    'latitude',         # Geographic location
    'longitude',        # Geographic location
    'depth'             # Sample depth (should be ≤10m)
]

TARGET_COL = 'aragonite'

# Clean data
df_model = df[FEATURE_COLS + [TARGET_COL]].dropna()
print(f"Clean samples (no missing data): {len(df_model):,}")
print(f"Dropped {len(df) - len(df_model):,} samples with missing values")

# ============================================================================
# 2. FEATURE ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("FEATURE STATISTICS")
print("="*60)
print(df_model[FEATURE_COLS].describe())

# Check for outliers
print("\nTarget variable (Ωarag) distribution:")
print(df_model[TARGET_COL].describe())

# ============================================================================
# 3. TRAIN/VAL/TEST SPLIT
# ============================================================================

X = df_model[FEATURE_COLS].values
y = df_model[TARGET_COL].values

# Split: 60% train, 20% validation, 20% test
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

# Normalize features
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

# Normalize target (only for neural network)
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

# ============================================================================
# 4. BASELINE MODELS (TREE-BASED)
# ============================================================================

print("\n" + "="*60)
print("TRAINING BASELINE MODELS")
print("="*60)

results = {}

# ----- Random Forest -----
print("\n[1/4] Random Forest...")
start = time.time()
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

results['Random Forest'] = {
    'model': rf_model,
    'predictions': rf_pred,
    'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
    'mae': mean_absolute_error(y_test, rf_pred),
    'r2': r2_score(y_test, rf_pred),
    'train_time': time.time() - start
}
print(f"  R² = {results['Random Forest']['r2']:.4f}, "
      f"RMSE = {results['Random Forest']['rmse']:.4f}, "
      f"Time = {results['Random Forest']['train_time']:.1f}s")

# ----- XGBoost -----
print("\n[2/4] XGBoost...")
start = time.time()
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

results['XGBoost'] = {
    'model': xgb_model,
    'predictions': xgb_pred,
    'rmse': np.sqrt(mean_squared_error(y_test, xgb_pred)),
    'mae': mean_absolute_error(y_test, xgb_pred),
    'r2': r2_score(y_test, xgb_pred),
    'train_time': time.time() - start
}
print(f"  R² = {results['XGBoost']['r2']:.4f}, "
      f"RMSE = {results['XGBoost']['rmse']:.4f}, "
      f"Time = {results['XGBoost']['train_time']:.1f}s")

# ----- LightGBM -----
print("\n[3/4] LightGBM...")
start = time.time()
lgb_model = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)

results['LightGBM'] = {
    'model': lgb_model,
    'predictions': lgb_pred,
    'rmse': np.sqrt(mean_squared_error(y_test, lgb_pred)),
    'mae': mean_absolute_error(y_test, lgb_pred),
    'r2': r2_score(y_test, lgb_pred),
    'train_time': time.time() - start
}
print(f"  R² = {results['LightGBM']['r2']:.4f}, "
      f"RMSE = {results['LightGBM']['rmse']:.4f}, "
      f"Time = {results['LightGBM']['train_time']:.1f}s")

# ============================================================================
# 5. NEURAL NETWORK
# ============================================================================

print("\n[4/4] Neural Network...")

class AragonitePredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

# PyTorch datasets
train_dataset = TensorDataset(
    torch.FloatTensor(X_train_scaled),
    torch.FloatTensor(y_train_scaled.reshape(-1, 1))
)
val_dataset = TensorDataset(
    torch.FloatTensor(X_val_scaled),
    torch.FloatTensor(y_val_scaled.reshape(-1, 1))
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Initialize model
model = AragonitePredictor(input_dim=len(FEATURE_COLS))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

# Training
start = time.time()
num_epochs = 200
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            val_loss += criterion(model(X_batch), y_batch).item()
    val_loss /= len(val_loader)
    
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'feature_cols': FEATURE_COLS
        }, '../models/aragonite_model.pth')
    else:
        patience_counter += 1
    
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d} | Train: {train_loss:.4f} | "
              f"Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")
    
    if patience_counter >= 30:
        print(f"  Early stopping at epoch {epoch+1}")
        break

nn_train_time = time.time() - start

# Load best model and evaluate
checkpoint = torch.load('../models/aragonite_model.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    y_pred_scaled = model(torch.FloatTensor(X_test_scaled)).numpy()

nn_pred = scaler_y.inverse_transform(y_pred_scaled).flatten()

results['Neural Network'] = {
    'model': model,
    'predictions': nn_pred,
    'rmse': np.sqrt(mean_squared_error(y_test, nn_pred)),
    'mae': mean_absolute_error(y_test, nn_pred),
    'r2': r2_score(y_test, nn_pred),
    'train_time': nn_train_time
}
print(f"  R² = {results['Neural Network']['r2']:.4f}, "
      f"RMSE = {results['Neural Network']['rmse']:.4f}, "
      f"Time = {results['Neural Network']['train_time']:.1f}s")

# ============================================================================
# 6. MODEL COMPARISON
# ============================================================================

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'R²': [results[m]['r2'] for m in results.keys()],
    'RMSE': [results[m]['rmse'] for m in results.keys()],
    'MAE': [results[m]['mae'] for m in results.keys()],
    'Train Time (s)': [results[m]['train_time'] for m in results.keys()]
})

comparison_df = comparison_df.sort_values('R²', ascending=False)
print(comparison_df.to_string(index=False))

# Find best model
best_model_name = comparison_df.iloc[0]['Model']
best_model_r2 = comparison_df.iloc[0]['R²']
best_model_rmse = comparison_df.iloc[0]['RMSE']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   R² = {best_model_r2:.4f}")
print(f"   RMSE = {best_model_rmse:.4f} Ω units")

# Save comparison
comparison_df.to_csv('../results/model_comparison.csv', index=False)
print("\n✓ Saved: results/model_comparison.csv")

# ============================================================================
# 7. FEATURE IMPORTANCE (Tree-based models only)
# ============================================================================

print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, model_name in enumerate(['Random Forest', 'XGBoost', 'LightGBM']):
    importances = results[model_name]['model'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    axes[idx].barh(range(len(FEATURE_COLS)), importances[indices])
    axes[idx].set_yticks(range(len(FEATURE_COLS)))
    axes[idx].set_yticklabels([FEATURE_COLS[i] for i in indices])
    axes[idx].set_xlabel('Importance')
    axes[idx].set_title(f'{model_name}\nR² = {results[model_name]["r2"]:.3f}')
    axes[idx].invert_yaxis()

plt.tight_layout()
plt.savefig('../results/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/feature_importance.png")

# Print feature rankings
print("\nFeature importance (XGBoost):")
xgb_importances = results['XGBoost']['model'].feature_importances_
for i in np.argsort(xgb_importances)[::-1]:
    print(f"  {FEATURE_COLS[i]:15s}: {xgb_importances[i]:.4f}")

# ============================================================================
# 8. PREDICTION VISUALIZATIONS
# ============================================================================

print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# Create 2x2 subplot for all models
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.flatten()

for idx, (model_name, model_results) in enumerate(results.items()):
    ax = axes[idx]
    
    y_pred = model_results['predictions']
    
    # Scatter plot
    ax.scatter(y_test, y_pred, alpha=0.5, s=20, label='Predictions')
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', lw=2, label='Perfect', zorder=10)
    
    # Metrics
    r2 = model_results['r2']
    rmse = model_results['rmse']
    mae = model_results['mae']
    
    ax.set_xlabel('Actual Ωarag', fontsize=12)
    ax.set_ylabel('Predicted Ωarag', fontsize=12)
    ax.set_title(f'{model_name}\nR² = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('../results/model_predictions_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/model_predictions_comparison.png")

# ============================================================================
# 9. RESIDUAL ANALYSIS (Best Model)
# ============================================================================

best_pred = results[best_model_name]['predictions']
residuals = y_test - best_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residual plot
axes[0].scatter(best_pred, residuals, alpha=0.5, s=20)
axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0].set_xlabel('Predicted Ωarag')
axes[0].set_ylabel('Residuals')
axes[0].set_title(f'Residual Plot ({best_model_name})')
axes[0].grid(True, alpha=0.3)

# Residual histogram
axes[1].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1].set_xlabel('Residuals')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Residual Distribution\nMean = {residuals.mean():.3f}, Std = {residuals.std():.3f}')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/residual_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/residual_analysis.png")

# ============================================================================
# 10. SAVE BEST MODEL
# ============================================================================

print("\n" + "="*60)
print("SAVING BEST MODEL")
print("="*60)

if best_model_name == 'Neural Network':
    # Already saved during training
    print(f"✓ Neural Network saved as: models/aragonite_model.pth")
else:
    # Save tree-based model
    model_save_path = f'../models/aragonite_model_{best_model_name.lower().replace(" ", "_")}.pkl'
    with open(model_save_path, 'wb') as f:
        pickle.dump({
            'model': results[best_model_name]['model'],
            'scaler_X': scaler_X,
            'feature_cols': FEATURE_COLS
        }, f)
    print(f"✓ {best_model_name} saved as: {model_save_path}")

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================

print("\n" + "="*60)
print("✓ TRAINING COMPLETE!")
print("="*60)
print(f"\nBest Model: {best_model_name}")
print(f"Test R²: {best_model_r2:.4f}")
print(f"Test RMSE: {best_model_rmse:.4f} Ω units")
print(f"Test MAE: {results[best_model_name]['mae']:.4f} Ω units")
print(f"\nAll models trained and compared.")
print(f"Results saved to: results/")
print(f"Models saved to: models/")