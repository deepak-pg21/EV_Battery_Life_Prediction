#!/usr/bin/env python3
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'merged_ev_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

# Load previously trained models
life_model = joblib.load(os.path.join(MODEL_DIR, 'ev_life_model.pkl'))
cost_model = joblib.load(os.path.join(MODEL_DIR, 'ev_cost_model.pkl'))

features = [
    'battery_temperature', 'voltage', 'current',
    'state_of_charge', 'avg_current',
    'state_of_health', 'mileage_km', 'age_months'
]

X = df[features]
merged = X.copy()
merged['pred_cycles'] = life_model.predict(X)
merged['pred_cost'] = cost_model.predict(X)

# Hybrid health index
hybrid_target = (df['state_of_health'] * 0.6 + merged['pred_cycles'] / 40).clip(0, 100)

health_model = RandomForestRegressor(n_estimators=80, max_depth=8, random_state=42)
health_model.fit(merged[['pred_cycles', 'pred_cost', 'state_of_health']], hybrid_target)

joblib.dump(health_model, os.path.join(MODEL_DIR, 'ev_health_model.pkl'))
print("✅ Battery Health Model trained and saved successfully!")
