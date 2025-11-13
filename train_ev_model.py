#!/usr/bin/env python3
"""
EV Insight: Unified Model Training Script (Stable Edition)
Trains battery life, cost, and health prediction models safely and cleanly.
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# -------------------------
# Locate dataset
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'merged_ev_data.csv')  # simplified path
MODEL_DIR = os.path.join(BASE_DIR, 'model')

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Looking for dataset at: {DATA_PATH}")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}. Please place merged_ev_data.csv in /data folder.")

# -------------------------
# Load dataset safely
# -------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# -------------------------
# Data Cleaning
# -------------------------
print("\n🧹 Cleaning data...")

# Drop completely empty columns or rows
df.dropna(how='all', inplace=True)

# Fill missing numeric values with median
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        df[col].fillna(df[col].median(), inplace=True)
    else:
        df[col].fillna(method='ffill', inplace=True)

# Convert any non-numeric data in numeric columns safely
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except Exception:
        pass

print("✅ Data cleaned successfully.")

# -------------------------
# Ensure required columns
# -------------------------
required_cols = [
    'battery_temperature','voltage','current','state_of_charge',
    'avg_current','state_of_health','mileage_km','age_months',
    'charge_cycles','battery_replacement_cost'
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -------------------------
# Define features
# -------------------------
features = [
    'battery_temperature','voltage','current','state_of_charge',
    'avg_current','state_of_health','mileage_km','age_months'
]

# -------------------------
# Model 1: Battery Life (charge_cycles)
# -------------------------
print("\n[1/3] 🔋 Training Battery Life Model...")
X = df[features]
y_life = df['charge_cycles']

X_train, X_test, y_train, y_test = train_test_split(X, y_life, test_size=0.2, random_state=42)
life_model = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42)
life_model.fit(X_train, y_train)

life_preds = life_model.predict(X_test)
print(f"   MAE: {mean_absolute_error(y_test, life_preds):.2f} | R²: {r2_score(y_test, life_preds):.3f}")

joblib.dump(life_model, os.path.join(MODEL_DIR, 'ev_life_model.pkl'))
print("✅ Battery Life Model saved.")

# -------------------------
# Model 2: Battery Cost (battery_replacement_cost)
# -------------------------
print("\n[2/3] 💰 Training Battery Cost Model...")
y_cost = df['battery_replacement_cost']

X_train, X_test, y_train, y_test = train_test_split(X, y_cost, test_size=0.2, random_state=42)
cost_model = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42)
cost_model.fit(X_train, y_train)

cost_preds = cost_model.predict(X_test)
print(f"   MAE: {mean_absolute_error(y_test, cost_preds):.2f} | R²: {r2_score(y_test, cost_preds):.3f}")

joblib.dump(cost_model, os.path.join(MODEL_DIR, 'ev_cost_model.pkl'))
print("✅ Battery Cost Model saved.")

# -------------------------
# Model 3: Battery Health (hybrid)
# -------------------------
print("\n[3/3] ❤️ Training Battery Health Model...")
merged = X.copy()
merged['pred_cycles'] = life_model.predict(X)
merged['pred_cost'] = cost_model.predict(X)

hybrid_target = (df['state_of_health'] * 0.6 + merged['pred_cycles'] / 40).clip(0, 100)

health_model = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42)
health_model.fit(merged[['pred_cycles','pred_cost','state_of_health']], hybrid_target)

joblib.dump(health_model, os.path.join(MODEL_DIR, 'ev_health_model.pkl'))
print("✅ Battery Health Model saved.")

# -------------------------
# Final Summary
# -------------------------
print("\n🎯 All models trained and saved successfully in /model/")
print("Files created:")
print("  - ev_life_model.pkl")
print("  - ev_cost_model.pkl")
print("  - ev_health_model.pkl")
