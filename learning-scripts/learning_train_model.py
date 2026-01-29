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

# 1. Load data
# Create dataframe
print('Loading data...')
df = pd.read_csv("../data/processed/complete_dataset.csv")
print('Successfully loaded data')

# Define input features
FEATURE_COLS = [
    'salinity',
    'modis_sst',
    'modis_chlora',
    'latitude',
    'longitude',
    'bathymetry_m',
    'depth'
]

# Define target
TARGET_COL = 'aragonite'

# Select necessary columns and remove rows with NaN values
df_model = df[FEATURE_COLS + [TARGET_COL]].dropna()
missing = len(df) - len(df_model)

# Display summary
print(f"\nDATA SUMMARY")
print(f"Total samples: {len(df):,}")
print(f"Clean samples: {len(df_model):,}")
print(f"Missing data: {missing:,} ({100*missing/len(df):.1f}%)")

# 2. Train/val/test split
# Convert from pandas dataframe to numpy array for sklearn
X = df_model[FEATURE_COLS].values
y = df_model[TARGET_COL].values

# First split: 20% for final testing
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: 80% remaining -> 60% training and 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42
)

# Display data splits
print("\nDATA SPLITS")
print(f"Training:   {len(X_train):,} samples ({100*len(X_train)/len(X):.1f}%)")
print(f"Validation: {len(X_val):,} samples ({100*len(X_val)/len(X):.1f}%)")
print(f"Test:       {len(X_test):,} samples ({100*len(X_test)/len(X):.1f}%)")

# 3. Train model
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

print("\nTraining random forest...")
start_time = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_time

print(f"Successfully completed training in {train_time:.2f}s")

# 4. Evaluate on validation set
# Make predictions on validation data
y_val_pred = model.predict(X_val)

# Calculate R^2
val_r2 = r2_score(y_val, y_val_pred)

# Calculate RMSE
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Calculate MAE
val_mae = mean_absolute_error(y_val, y_val_pred)

# Display results
print("\nVALIDATION SET PERFORMANCE")
print(f"R^2: {val_r2:.4f}")
print(f"RMSE: {val_rmse:.4f} Omega units")
print(f"MAE: {val_mae:.4f} Omega units")

# 5. Evaluate on test set
y_test_pred = model.predict(X_test)

# Calculate R^2
test_r2 = r2_score(y_test, y_test_pred)

# Calculate RMSE
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Calculate MAE
test_mae = mean_absolute_error(y_test, y_test_pred)

# Display results
print("\nTEST SET PERFORMANCE (FINAL)")
print(f"R^2: {test_r2:.4f}")
print(f"RMSE: {test_rmse:.4f} Omega units")
print(f"MAE: {test_mae:.4f} Omega units")
print(f"Relative error: {100*test_rmse/y_test.mean():.1f}% of mean")

# 6. Cross Validation
print("\nRunning 10-fold CV (5 repeats)...")

# Set up K-fold cross validation
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

# Calculate 95% confidence interval
ci_lower = cv_scores.mean() - 1.96*cv_scores.std()
ci_upper = cv_scores.mean() + 1.96*cv_scores.std()

# Display results
print(f"Successfully completed CV in {cv_time:.1f}s\n")
print(f"CROSS-VALIDATION R^2 SCORES (50 folds):")
print(f"Mean: {cv_scores.mean():.4f}")
print(f"Std: {cv_scores.std():.4f}")
print(f"Min: {cv_scores.min():.4f}")
print(f"Max: {cv_scores.max():.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# 7. Feature importance
# Get importance scores for each feature
importances = model.feature_importances_

# Sort features by importance
indices = np.argsort(importances)[::-1]

# Display results
print("\nFEATURE IMPORTANCE (ranked):")
for i, idx in enumerate(indices, 1):
    print(f"{i}. {FEATURE_COLS[idx]}: {importances[idx]:.3f}")

# 8. Confidence score
# Get predictions from all trees seperately
tree_predictions = np.array([tree.predict(X_test) for tree in model.estimators_])

# Calculate standard deviation across trees for each sample
pred_std = tree_predictions.std(axis=0)

# Convert std to confidence percentage
max_std = pred_std.max()
if max_std > 0:
    confidence_scores = 100 * (1 - pred_std / max_std)
else:
    confidence_scores = np.ones(len(pred_std)) * 100

# 8. Save model
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

# Bundle everything into one dictionary
model_package = {
    'model': model,
    'feature_cols': FEATURE_COLS,
    'feature_importance': dict(zip(FEATURE_COLS, importances)),
    'predict_with_confidence': predict_with_confidence,
}

# Save to disk
with open('../models/aragonite_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)