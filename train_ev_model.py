#!/usr/bin/env python3
"""
EV Insight: Robust Unified Model Training Script
Tries multiple data locations, creates folders if missing, logs metrics.
"""

import os
import sys
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------
# Helper: find dataset in likely locations
# -------------------------
def find_dataset(filename="merged_ev_data.csv"):
    # folder where this script sits
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, "data", filename),            # <project>/data/merged_ev_data.csv
        os.path.join(script_dir, filename),                    # <project>/merged_ev_data.csv
        os.path.join(script_dir, "..", "data", filename),      # <project>/../data/merged_ev_data.csv
        os.path.join(script_dir, "..", filename),              # <project>/../merged_ev_data.csv
    ]
    # make absolute and dedupe
    seen = set()
    final = []
    for p in candidates:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            final.append(ap)
    for p in final:
        if os.path.exists(p):
            return p
    return final  # return list of tried paths if not found

# -------------------------
# Setup folders
# -------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR  # user is running script from repo root
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
LOG_PATH = os.path.join(MODEL_DIR, "train_log.txt")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print("Looking for dataset `merged_ev_data.csv`...")
found = find_dataset("merged_ev_data.csv")
if isinstance(found, str):
    DATA_PATH = found
    print("Found dataset at:", DATA_PATH)
else:
    print("Could not find merged_ev_data.csv in the usual locations.")
    print("Paths tried:")
    for p in found:
        print("  -", p)
    print("\n➡️ Please place your dataset file named exactly `merged_ev_data.csv` into one of these folders:")
    print("   -", os.path.join(PROJECT_ROOT, "data"))
    print("   -", PROJECT_ROOT)
    print("\nExample (Windows CMD):")
    print(r'  move "C:\path\to\your\merged_ev_data.csv" "%cd%\data\"')
    sys.exit(1)

# -------------------------
# Load dataset
# -------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print("Rows:", len(df), "Columns:", len(df.columns))

# -------------------------
# Required columns check
# -------------------------
required_cols = [
    'battery_temperature','voltage','current','state_of_charge',
    'avg_current','state_of_health','mileage_km','age_months',
    'charge_cycles','battery_replacement_cost'
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print("ERROR: dataset is missing required columns:")
    for m in missing:
        print(" -", m)
    print("\nYou can either add these columns to the CSV or rename your CSV columns accordingly.")
    sys.exit(1)

features = [
    'battery_temperature','voltage','current','state_of_charge',
    'avg_current','state_of_health','mileage_km','age_months'
]

# -------------------------
# Model 1: Battery Life
# -------------------------
print("\n[1/3] Training Battery Life Model (charge_cycles)...")
X = df[features]
y_life = df['charge_cycles']

X_train, X_test, y_train, y_test = train_test_split(X, y_life, test_size=0.2, random_state=42)
life_model = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42)
life_model.fit(X_train, y_train)
life_preds = life_model.predict(X_test)
life_mae = mean_absolute_error(y_test, life_preds)
life_r2 = r2_score(y_test, life_preds)
joblib.dump(life_model, os.path.join(MODEL_DIR, 'ev_life_model.pkl'))
print(f" - Life MAE: {life_mae:.3f}, R2: {life_r2:.3f}")

# -------------------------
# Model 2: Battery Cost
# -------------------------
print("\n[2/3] Training Battery Cost Model (battery_replacement_cost)...")
y_cost = df['battery_replacement_cost']
X_train, X_test, y_train, y_test = train_test_split(X, y_cost, test_size=0.2, random_state=42)
cost_model = RandomForestRegressor(n_estimators=120, max_depth=10, random_state=42)
cost_model.fit(X_train, y_train)
cost_preds = cost_model.predict(X_test)
cost_mae = mean_absolute_error(y_test, cost_preds)
cost_r2 = r2_score(y_test, cost_preds)
joblib
