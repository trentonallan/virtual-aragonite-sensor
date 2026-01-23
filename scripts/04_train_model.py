import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

print("="*60)
print("RETRAINING WITH BATHYMETRY")
print("="*60)

# Load data with bathymetry
df = pd.read_csv("../data/processed/complete_dataset.csv")
print(f"\nLoaded {len(df):,} samples")

feature_cols = [
    'modis_sst',
    'modis_chlora',
    'salinity',
    'bathymetry_m',  
    'latitude',
    'longitude',
    'depth'
]

target_col = 'aragonite'

# Drop rows with missing data
df_model = df[feature_cols + [target_col]].dropna()
print(f"Clean samples: {len(df_model):,}")

X = df_model[feature_cols].values
y = df_model[target_col].values.reshape(-1, 1)

# Train/val/test split (same random seed for fair comparison)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"\nData splits:")
print(f"  Training:   {len(X_train):,}")
print(f"  Validation: {len(X_val):,}")
print(f"  Test:       {len(X_test):,}")

# Normalize
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train)
y_val_scaled = scaler_y.transform(y_val)
y_test_scaled = scaler_y.transform(y_test)

# PyTorch datasets
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train_scaled))
val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val_scaled))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Model (now with 7 input features instead of 6)
class AragonitePredictor(nn.Module):
    def __init__(self, input_dim=7):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x)

model = AragonitePredictor(input_dim=len(feature_cols))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

print("\n" + "="*60)
print("TRAINING")
print("="*60)

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
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
            'feature_cols': feature_cols
        }, '../models/model_with_bathymetry.pth')
    else:
        patience_counter += 1
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Best: {best_val_loss:.4f}")
    
    if patience_counter >= 30:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print(f"\n✓ Training complete! Best val loss: {best_val_loss:.4f}")

# Load best model and evaluate
checkpoint = torch.load('../models/model_with_bathymetry.pth', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    y_pred_scaled = model(torch.FloatTensor(X_test_scaled)).numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test_scaled)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
mae = mean_absolute_error(y_test_original, y_pred)
r2 = r2_score(y_test_original, y_pred)

print("\n" + "="*60)
print("TEST RESULTS (WITH BATHYMETRY)")
print("="*60)
print(f"RMSE: {rmse:.3f} Ω units")
print(f"MAE:  {mae:.3f} Ω units")
print(f"R²:   {r2:.3f}")

# Load old model for comparison
print("\n" + "="*60)
print("COMPARISON TO BASELINE")
print("="*60)
print(f"Baseline (6 features):  R² = 0.734, RMSE = 0.447")
print(f"With bathymetry (7):     R² = {r2:.3f}, RMSE = {rmse:.3f}")
print(f"Improvement:            ΔR² = {r2 - 0.734:+.3f}, ΔRMSE = {rmse - 0.447:+.3f}")

# Visualization
plt.figure(figsize=(10, 10))
plt.scatter(y_test_original, y_pred, alpha=0.5, s=20, label='Predictions')
plt.plot([y_test_original.min(), y_test_original.max()],
         [y_test_original.min(), y_test_original.max()],
         'r--', lw=2, label='Perfect')
plt.xlabel('Actual Ωarag')
plt.ylabel('Predicted Ωarag')
plt.title(f'With Bathymetry: R² = {r2:.3f}, RMSE = {rmse:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/predictions_with_bathymetry.png', dpi=300)
print("\n✓ Saved: results/predictions_with_bathymetry.png")

print("\n" + "="*60)
print("✓ BATHYMETRY MODEL COMPLETE!")
print("="*60)