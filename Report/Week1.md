# Week 1: Problem Definition & Dataset Collection

**Project:** EV_Battery_Life_Prediction  
**Author:** P.G. Deepak Chiranjeevi

## Objective
Develop a machine-learning model to predict the Remaining Useful Life (RUL) and degradation rate of electric vehicle batteries using operational, environmental, and usage data.

## Problem Statement
Given historical battery telemetry (cycle number, temperature, current, voltage, capacity measurements, internal resistance), predict the remaining capacity percentage and estimate degradation trend.

## Scope
- Regression problem (predict continuous Remaining Capacity %)
- Optional classification (Healthy / Degraded / Critical)
- Use features: cycle, temperature, voltage, current, capacity, internal resistance, C-rate, energy throughput

## Dataset Sources (examples)
- NASA battery datasets  
- CALCE Battery Research datasets  
- Kaggle (battery degradation datasets)  
- Manufacturer/telematics data (if available)

## Tools & Environment
- Python (pandas, numpy, scikit-learn)
- Jupyter Notebook / Google Colab
- Streamlit (for demo UI)
- Optional: python-docx for report generation, matplotlib/seaborn for plots

## Deliverables (End of Week 1)
- Problem definition and objectives
- Selected dataset saved in `dataset/`
- Environment setup instructions
- Streamlit skeleton app in `Week_1/Code/main_week1.py`

