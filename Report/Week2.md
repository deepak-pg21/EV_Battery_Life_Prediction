# Week 2: Data Preparation (Cleaning & Preprocessing)

**Project:** EV_Battery_Life_Prediction  
**Author:** P.G. Deepak Chiranjeevi

---

## Objective

Prepare the raw battery dataset to ensure it is clean, consistent, and structured for effective modeling by performing data cleaning, imputation, feature engineering, and data splitting.

---

## Data Import and Initial Inspection

- Loaded dataset:
  import pandas as pd
  df = pd.read_csv('dataset/ev_battery_data_sample.csv')
- Dataset overview using:
- `df.head()`
- `df.info()`
- `df.describe()`

---

## Data Cleaning

- Dropped exact duplicate rows to avoid data redundancy  
- Imputed missing values using appropriate methods:
- Numerical columns: median imputation  
- Forward fill for time series continuity  
- Corrected incorrect or inconsistent data types for numerical and categorical features

---

## Feature Engineering

- Calculated **C-rate** as `current / capacity` to capture the charge/discharge rates  
- Computed **cumulative energy throughput** (integral of power over time) as an indicator of usage intensity  
- Created rolling averages for temperature and current to smooth transient noise if required

---

## Encoding

- Applied One-Hot Encoding or Label Encoding for any categorical variables (e.g., battery model, manufacturer)

---

## Normalization/Scaling

- Used `StandardScaler` or `MinMaxScaler` to scale features for model convergence and improved performance

---

## Splitting Dataset

- Applied `train_test_split` (default 80% train, 20% test) ensuring:
- Stratified split if classification target is used  
- Randomized split for regression focus

---

## Outputs

- Cleaned and preprocessed dataset saved to:  
`dataset/ev_battery_data_cleaned.csv`  
- Code and notebooks committed under:  
`Week_2/Code/main_week2.py`

---
