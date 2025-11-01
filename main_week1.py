"""Week 1 - Streamlit Intro App (placeholder)
Run with: streamlit run Week_1/Code/main_week1.py
"""
import streamlit as st

st.set_page_config(page_title="EV Battery RUL Demo", layout="centered")
st.title("EV Battery Life Prediction — Demo Interface")
st.markdown("This is a placeholder Streamlit app for the EV Battery Life Prediction project. Replace with a trained model in Week 4/5.")

st.header("Input battery telemetry (sample)")
temp = st.number_input('Temperature (°C)', min_value=-20.0, max_value=60.0, value=25.0)
cycle = st.number_input('Charge cycle count', min_value=0, max_value=10000, value=50)
current = st.number_input('Average current (A)', min_value=0.0, max_value=200.0, value=20.0)
voltage = st.number_input('Voltage (V)', min_value=2.0, max_value=5.0, value=3.7)

if st.button('Predict (demo)'):
    # Placeholder simple heuristic
    predicted = max(30.0, 100.0 - 0.12 * cycle - 0.3 * (temp-25))
    st.success(f"Predicted Remaining Capacity: {predicted:.2f}% (demo heuristic)")
