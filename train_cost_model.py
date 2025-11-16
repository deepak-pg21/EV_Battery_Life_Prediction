#!/usr/bin/env python3
"""
EV Insight: Battery Cost Model Training Script
Trains and saves a RandomForest model to predict EV battery replacement cost.
Author: PG Deepak Chiranjeevi (2025)
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------------------------------------
# Constants and Paths
# -----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'merged_ev_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

os.makedirs(MODEL_DIR, exist_ok=True)

required_features = [
    'battery_temperature', 'voltage', 'current',
    'state_of_charge', 'avg_current',
    'state_of_health', 'mileage_km', 'age_months'
]
target_col = 'battery_replacement_cost'

# -----------------------------------------------------------
# Load and Validate Data
# -----------------------------------------------------------
print(f"Loading dataset from: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"❌ Dataset not found at {DATA_PATH}. Please place merged_ev_data.csv in /data folder."
    )
df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Validate feature and target columns
missing = [col for col in required_features + [target_col] if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Fill nans and clean data
df.dropna(how='all', inplace=True)
for col in required_features + [target_col]:
    if df[col].dtype in [np.float64, np.int64, float, int]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
    else:
        df[col] = df[col].fillna(method='ffill')

# -----------------------------------------------------------
# Prepare Data
# -----------------------------------------------------------
X = df[required_features]
y = df[target_col]

# -----------------------------------------------------------
# Train-Test Split
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# -----------------------------------------------------------
# Model Training
# -----------------------------------------------------------
print("Training Battery Replacement Cost Model...")
cost_model = RandomForestRegressor(
    n_estimators=120,
    max_depth=10,
    random_state=42
)
cost_model.fit(X_train, y_train)
y_pred = cost_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"   MAE: {mae:.2f} | R²: {r2:.3f}")

# -----------------------------------------------------------
# Save Model
# -----------------------------------------------------------
model_path = os.path.join(MODEL_DIR, 'ev_cost_model.pkl')
joblib.dump(cost_model, model_path)
print(f"✅ Battery Cost Model saved to {model_path}")

# -----------------------------------------------------------
# Final Summary
# -----------------------------------------------------------
print("🎯 Battery Cost model created and stored successfully.")

