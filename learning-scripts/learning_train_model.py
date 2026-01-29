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
df = pd.read_csv("../data/processed/complete_dataset.csv")

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

start_time = time.time()
model.fit(X_train, y_train)
train_time = time.time() - start_time

# 4. Evaluate on validation set
# Make predictions on validation data
y_val_pred = model.predict(X_val)

# Calculate R^2
val_r2 = r2_score(y_val, y_val_pred)

# Calculate RMSE
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Calculate MAE
val_mae = mean_absolute_error(y_val, y_val_pred)

# 5. Evaluate on test set
y_test_pred = model.predict(X_test)

# Calculate R^2
test_r2 = r2_score(y_test, y_test_pred)

# Calculate RMSE
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Calculate MAE
test_mae = mean_absolute_error(y_test, y_test_pred)

# 6. Cross Validation
# Set up K-fold cross validation
cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)

cv_scores = cross_val_score(
    model,
    np.vstack([X_train, X_val]),
    np.hstack([y_train, y_val]),
    cv=cv,
    scoring='r2',
    n_jobs=-1
)

# Calculate 95% confidence interval
ci_lower = cv_scores.mean() - 1.96*cv_scores.std()
ci_upper = cv_scores.mean() + 1.96*cv_scores.std()

