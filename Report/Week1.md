# Week 1: Problem Definition & Dataset Collection

**Project:** EV_Battery_Life_Prediction  
**Author:** P.G. Deepak Chiranjeevi

---

## Objective

Develop a machine learning model to accurately predict the Remaining Useful Life (RUL) and degradation rate of electric vehicle (EV) batteries using operational, environmental, and usage data.

---

## Problem Statement

Given historical battery telemetry data — including cycle number, temperature, current, voltage, capacity measurements, and internal resistance — the goal is to:

- Predict the battery’s Remaining Capacity percentage (continuous regression target)
- Optionally classify battery state into Healthy, Degraded, or Critical categories
- Estimate degradation trends to support predictive maintenance and lifecycle planning

---

## Scope

- Formulated as a regression task (predicting Remaining Capacity %)
- Optional classification layering based on thresholded capacity levels
- Key features to be utilized include:  
  - Cycle number  
  - Temperature  
  - Voltage  
  - Current  
  - Battery capacity  
  - Internal resistance  
  - C-rate  
  - Energy throughput
  
---

## Dataset Sources


- CALCE Battery Research datasets  
- Kaggle Battery Degradation datasets  
- Manufacturer or Telematics System data (subject to availability)

---

## Tools & Environment

- Programming Language: Python  
- Libraries: pandas, numpy, scikit-learn  
- Development Environment: Jupyter Notebook / Google Colab  
- Visualization: matplotlib, seaborn  
- User Interface: Streamlit (for future demo application)  
- Documentation: python-docx (for report generation)

---

## Deliverables (End of Week 1)

- Clear problem definition and project objectives  
- Downloaded and saved chosen dataset(s) into `dataset/` directory  
- Environment setup documentation for reproducibility  
- Streamlit skeleton application authored in `Week_1/Code/main_week1.py` for initial exploration

---
