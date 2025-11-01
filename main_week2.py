"""Week 2 - Data Preparation Script
Run: python Week_2/Code/main_week2.py
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

input_path = os.path.join('dataset', 'ev_battery_data_sample.csv')
df = pd.read_csv(input_path)

print('Initial shape:', df.shape)
print(df.head())

# Basic cleaning
df.drop_duplicates(inplace=True)
# Example: drop rows with any NA
df = df.dropna()

# Feature engineering example
# compute C-rate (approx) if capacity is present
if 'capacity_Ah' in df.columns and 'current_A' in df.columns:
    # avoid division by zero
    df['c_rate'] = df['current_A'] / df['capacity_Ah'].replace(0, 1)

# Scaling numeric features
num_cols = ['temperature_C','voltage_V','current_A','capacity_Ah','internal_resistance_mOhm','c_rate']
available_num = [c for c in num_cols if c in df.columns]
scaler = StandardScaler()
df[available_num] = scaler.fit_transform(df[available_num])

# Save cleaned
out_path = os.path.join('dataset','ev_battery_data_cleaned.csv')
df.to_csv(out_path, index=False)
print('Saved cleaned dataset to', out_path)
print('Cleaned shape:', df.shape)
