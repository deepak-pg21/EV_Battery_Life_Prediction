#!/usr/bin/env python3
"""
EV Insight: Battery Health Model Training Script
Trains and saves a RandomForest model to predict a unified battery health index, blending soH + predicted cycles/cost.
Author: PG Deepak Chiranjeevi (2025)
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor

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
state_of_health_col = 'state_of_health'

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

# Check if required columns exist
missing = [col for col in required_features + [state_of_health_col] if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Clean missing values
df.dropna(how='all', inplace=True)
for col in required_features + [state_of_health_col]:
    if df[col].dtype in [np.float64, np.int64, float, int]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
    else:
        df[col] = df[col].fillna(method='ffill')

# -----------------------------------------------------------
# Load Trained Models for Prediction Features
# -----------------------------------------------------------
life_model_path = os.path.join(MODEL_DIR, 'ev_life_model.pkl')
cost_model_path = os.path.join(MODEL_DIR, 'ev_cost_model.pkl')
if not os.path.isfile(life_model_path) or not os.path.isfile(cost_model_path):
    raise FileNotFoundError("❌ Model dependencies not found in /model/ folder. Train life/cost models first.")

life_model = joblib.load(life_model_path)
cost_model = joblib.load(cost_model_path)

X = df[required_features].copy()
X['predicted_cycles'] = life_model.predict(X[required_features])
X['predicted_cost'] = cost_model.predict(X[required_features])

# -----------------------------------------------------------
# Hybrid Target for Battery Health
# -----------------------------------------------------------
# Health index: weighted actual SoH + normalized cycles (max 4000) [you can tune this]
hybrid_target = (
    df['state_of_health'] * 0.6 +
    (X['predicted_cycles'] / 40) * 0.4  # scales cycles to ~100 range
).clip(0, 100)

# -----------------------------------------------------------
# Train Health Model
# -----------------------------------------------------------
print("Training Hybrid Battery Health Model...")
health_model = RandomForestRegressor(
    n_estimators=80,
    max_depth=8,
    random_state=42
)
predictive_features = ['predicted_cycles', 'predicted_cost', 'state_of_health']
health_model.fit(
    X[predictive_features],
    hybrid_target
)

# -----------------------------------------------------------
# Save Model
# -----------------------------------------------------------
model_path = os.path.join(MODEL_DIR, 'ev_health_model.pkl')
joblib.dump(health_model, model_path)
print(f"✅ Battery Health Model saved to {model_path}")

# -----------------------------------------------------------
# Final Summary
# -----------------------------------------------------------
print("🎯 Battery Health hybrid model created and stored successfully.")

