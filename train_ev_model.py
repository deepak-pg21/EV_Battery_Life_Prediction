#!/usr/bin/env python3
"""
EV Insight: Unified Model Training Script
Trains battery life, cost, and health prediction models using merged_ev_data.csv
"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------
# Locate dataset
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'merged_ev_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, '..', 'model')

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Ensure key columns exist
required_cols = [
    'battery_temperature','voltage','current','state_of_charge',
    'avg_current','state_of_health','mileage_km','age_months',
    'charge_cycles','battery_replacement_cost'
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# -------------------------
# Define features & targets
# -------------------------
features = [
    'battery_temperature','voltage','current','state_of_charge',
    'avg_current','state_of_health','mileage_km','age_months'
]

# -------------------------
# Model 1: Predict Life (Charge Cycles)
# -------------------------
print("\nTraining Battery Life Model...")
X = df[features]
y_life = df['charge_cycles']

X_train, X_test, y_train, y_test = train_test_split(X, y_life, test_size=0.2, random_state=42)
life_model = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42)
life_model.fit(X_train, y_train)

life_preds = life_model.predict(X_test)
print(f"Life Model → MAE: {mean_absolute_error(y_test, life_preds):.2f} | R²: {r2_score(y_test, life_preds):.3f}")

joblib.dump(life_model, os.path.join(MODEL_DIR, 'ev_life_model.pkl'))

# -------------------------
# Model 2: Predict Cost (Replacement Cost)
# -------------------------
print("\nTraining Battery Cost Model...")
y_cost = df['battery_replacement_cost']

X_train, X_test, y_train, y_test = train_test_split(X, y_cost, test_size=0.2, random_state=42)
cost_model = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42)
cost_model.fit(X_train, y_train)

cost_preds = cost_model.predict(X_test)
print(f"Cost Model → MAE: {mean_absolute_error(y_test, cost_preds):.2f} | R²: {r2_score(y_test, cost_preds):.3f}")

joblib.dump(cost_model, os.path.join(MODEL_DIR, 'ev_cost_model.pkl'))

# -------------------------
# Model 3: Predict Health Index (Hybrid)
# -------------------------
print("\nTraining Battery Health Model...")
merged = X.copy()
merged['pred_cycles'] = life_model.predict(X)
merged['pred_cost'] = cost_model.predict(X)

# Health target derived from state_of_health and predicted cycles
hybrid_target = (df['state_of_health'] * 0.6 + merged['pred_cycles'] / 40).clip(0, 100)

health_model = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42)
health_model.fit(merged[['pred_cycles','pred_cost','state_of_health']], hybrid_target)

joblib.dump(health_model, os.path.join(MODEL_DIR, 'ev_health_model.pkl'))

print("\n✅ All models trained and saved successfully in /model/")
