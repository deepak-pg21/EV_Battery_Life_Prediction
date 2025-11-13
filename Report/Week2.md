# Week 2: Data Preparation (Cleaning & Preprocessing)

**Project:** EV_Battery_Life_Prediction  
**Author:** P.G. Deepak Chiranjeevi

## Objective
Prepare the raw battery dataset for modeling: clean, impute, normalize, and split into train/test sets.

## Steps performed
1. Data import:
```python
import pandas as pd
df = pd.read_csv('dataset/ev_battery_data_sample.csv')
```

2. Initial inspection:
- `df.head()`
- `df.info()`
- `df.describe()`

3. Cleaning:
- Dropped exact duplicates.
- Imputed or handled missing values (example shown in code).
- Corrected data types.

4. Feature engineering:
- Compute C-rate = current / capacity
- Compute cumulative energy throughput
- Rolling averages of temperature or current if needed

5. Encoding:
- If battery type/model is categorical, use One-Hot Encoding or Label Encoding.

6. Normalization/Scaling:
- StandardScaler or MinMaxScaler to scale features before model training.

7. Splitting:
- `train_test_split` with stratify awareness for classification; for regression simple random split.

## Output
- Cleaned dataset written to `dataset/ev_battery_data_cleaned.csv`
- Notebook and code saved in `Week_2/Code/main_week2.py`
