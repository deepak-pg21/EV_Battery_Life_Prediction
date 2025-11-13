
# EV Battery Life & Cost Prediction — Data Preparation Summary

## Step 1: Dataset Creation
Combined features representing real-world EV battery behavior based on prior datasets.

### Columns:
- **battery_temperature (°C):** Temperature inside the battery pack
- **voltage (V):** Measured terminal voltage
- **current (A):** Instantaneous current drawn or supplied
- **state_of_charge (%):** Energy remaining in the battery
- **avg_current (A):** Average current usage profile
- **state_of_health (%):** Health index of the battery over its lifetime
- **mileage_km:** Total kilometers driven
- **age_months:** Battery age in months
- **predicted_life_cycles:** Estimated remaining charge cycles
- **battery_cost_usd:** Projected replacement or maintenance cost

## Step 2: Cleaning & Trimming
- Removed irrelevant columns and standardized names.
- Rounded values to 2 decimals to reduce file size and improve readability.
- Generated 300 balanced samples for demonstration.

## Step 3: Storage Optimization
- File size reduced to under 100 KB.
- CSV format for compatibility with Streamlit & ML models.

## Step 4: Usage
Use this dataset for model training or visualization in your `EV Insight` Streamlit app.
