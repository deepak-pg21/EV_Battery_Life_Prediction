#!/usr/bin/env python3
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'merged_ev_data.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

features = [
    'battery_temperature', 'voltage', 'current',
    'state_of_charge', 'avg_current',
    'state_of_health', 'mileage_km', 'age_months'
]

X = df[features]
y = df['battery_replacement_cost']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"Cost Model → MAE: {mean_absolute_error(y_test, preds):.2f}, R²: {r2_score(y_test, preds):.3f}")

joblib.dump(model, os.path.join(MODEL_DIR, 'ev_cost_model.pkl'))
print("✅ Battery Cost Model saved successfully!")
