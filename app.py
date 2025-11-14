# =====================================================
# 🔋 EV Insight — Battery Life & Cost Predictor
# Streamlit App (FINAL FIXED VERSION)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime


# =====================================================
# 1️⃣ PAGE CONFIGURATION
# =====================================================
st.set_page_config(page_title="EV Insight", page_icon="🔋", layout="wide")

st.markdown("""
<div style='background:linear-gradient(90deg,#d7f9e9,#ecfff3); padding:15px; border-radius:10px'>
    <h1 style='color:#045c3c; text-align:center;'>🔋 EV Insight — Battery Life & Cost Predictor</h1>
    <p style='text-align:center; color:#045c3c;'>Predict EV battery charge cycles, replacement cost, and health with intelligent insights.</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 2️⃣ MODEL LOADING
# =====================================================
MODEL_DIR = os.path.join(os.getcwd(), 'model')

try:
    life_model = joblib.load(os.path.join(MODEL_DIR, 'ev_life_model.pkl'))
    cost_model = joblib.load(os.path.join(MODEL_DIR, 'ev_cost_model.pkl'))
    health_model = joblib.load(os.path.join(MODEL_DIR, 'ev_health_model.pkl'))
    st.success("✅ ML Models loaded from /model/")
except Exception as e:
    st.error(f"❌ Error loading models: {e}")
    st.stop()


# =====================================================
# 3️⃣ DATA HANDLING
# =====================================================
st.subheader("📂 Upload or Use Sample Dataset")

uploaded = st.file_uploader("Upload EV dataset (CSV)", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.success("✔️ Your dataset loaded!")
else:
    data_path = os.path.join(os.getcwd(), "data", "merged_ev_data.csv")
    if not os.path.exists(data_path):
        st.error("❌ Sample dataset not found. Please upload a CSV.")
        st.stop()

    df = pd.read_csv(data_path)
    st.info("ℹ️ Using default sample dataset.")

st.dataframe(df.head(8))


# =====================================================
# 4️⃣ PREDICTION INPUT SELECTION
# =====================================================
st.markdown("---")
st.subheader("🔍 Select a Row to Predict")

idx = st.number_input("Row Index", min_value=0, max_value=len(df) - 1, value=0)
selected = df.iloc[[idx]]

# SAME FEATURE COLUMNS USED DURING TRAINING
feat_cols = [
    'battery_temperature', 'voltage', 'current', 'state_of_charge',
    'avg_current', 'state_of_health', 'mileage_km', 'age_months'
]

X_df = selected[feat_cols]  # <-- FIXED (prevents sklearn warnings)


# =====================================================
# 5️⃣ MODEL PREDICTIONS
# =====================================================
st.markdown("---")
st.subheader("📊 Prediction Results")

try:
    # Life Model
    pred_cycles = life_model.predict(X_df)[0]

    # Cost Model
    pred_cost = cost_model.predict(X_df)[0]

    # Health Model (uses previous outputs)
    health_input = pd.DataFrame([{
        "predicted_cycles": pred_cycles,
        "predicted_cost": pred_cost,
        "state_of_health": selected['state_of_health'].values[0]
    }])

    pred_health = health_model.predict(health_input)[0]

    # Display
    st.metric("🔋 Remaining Charge Cycles", f"{pred_cycles:.0f}")
    st.metric("💰 Estimated Replacement Cost (USD)", f"${pred_cost:,.2f}")
    st.metric("❤️ Battery Health (%)", f"{pred_health:.1f}%")

except Exception as e:
    st.error(f"❌ Prediction Error: {e}")


# =====================================================
# 6️⃣ VISUALIZATION (NO WARNINGS)
# =====================================================
st.markdown("---")
st.subheader("📈 Battery Degradation Trend")

try:
    preds = life_model.predict(df[feat_cols])
    df['predicted_cycles'] = preds

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['predicted_cycles'], linewidth=2)
    ax.set_title("Predicted Battery Cycle Degradation")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Remaining Cycles")
    ax.grid(True)

    st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Plotting Error: {e}")




# -----------------------------------------------------------
#  7️⃣ 🤖 ADVANCED EV CHATBOT
# -----------------------------------------------------------

st.markdown("---")
st.subheader("🤖 EV Smart Chat Assistant")

# Initialize chat history
if "chat" not in st.session_state:
    st.session_state.chat = [
        {"role": "assistant", "content": "Hello! I'm your EV Battery AI Assistant. Ask me anything."}
    ]

# Display chat messages
for msg in st.session_state.chat:
    if msg["role"] == "assistant":
        st.info("**Assistant:** " + msg["content"])
    else:
        st.success("**You:** " + msg["content"])


# User input
user_msg = st.text_input("Ask a question about EV Batteries, Life, Cost, Health, Charging, etc:")

if user_msg:
    st.session_state.chat.append({"role": "user", "content": user_msg})
    text = user_msg.lower()

    # -----------------------------------------------------------
    # 🔥 Smarter Context-Aware Responses
    # -----------------------------------------------------------

    reply = None

    # 🔋 Battery Charge & Life
    if any(word in text for word in ["charge", "charging", "fast charge", "slow charge"]):
        reply = (
            "Fast charging is convenient but increases battery heat, which accelerates degradation. "
            "For long-term health, use slow/normal charging for daily use and fast charging only when needed."
        )

    # 🌡 Temperature
    elif any(word in text for word in ["temperature", "heat", "cooling", "hot", "cold"]):
        reply = (
            "Battery temperature is crucial. EV batteries perform best between **15°C – 35°C**. "
            "High heat speeds up chemical wear; extreme cold reduces efficiency. "
            "Pre-conditioning helps maintain optimal temperature."
        )

    # 🧬 Battery Health
    elif any(word in text for word in ["health", "soh", "state of health"]):
        reply = (
            "Battery health depends on: charging habits, temperature exposure, depth-of-discharge, "
            "and total cycle count. Your predictions above estimate SOH using a hybrid ML model."
        )

    # 💰 Battery Cost / Replacement
    elif any(word in text for word in ["cost", "price", "replacement", "expensive"]):
        reply = (
            "Battery replacement cost varies by brand and capacity. Typically ₹1.5L – ₹3L in India. "
            "Your model predicts cost based on temperature, cycles, and usage behaviour."
        )

    # 🔄 Battery Cycles & Degradation
    elif any(word in text for word in ["cycle", "cycles", "degradation", "life", "lifetime"]):
        reply = (
            "Battery degradation is influenced mainly by charge cycles, temperature, and driving habits. "
            "Your prediction model estimates remaining cycles using real-world patterns."
        )

    # ⚡ Vehicle Performance
    elif any(word in text for word in ["motor", "power", "torque", "range"]):
        reply = (
            "Range depends on battery health, speed, climate control usage, and driving style. "
            "Maintaining steady speeds and avoiding rapid acceleration improves efficiency."
        )

    # 🧠 General fallback answer (Non-Repetitive)
    if reply is None:
        reply = (
            "I'm here to assist with EV batteries, health, cost, cycles, degradation, and charging. "
            "Ask me anything specific, like:\n"
            "- How to increase battery life?\n"
            "- What affects charging speed?\n"
            "- How do high temperatures affect health?"
        )

    # Append reply
    st.session_state.chat.append({"role": "assistant", "content": reply})

    # Refresh UI
    st.rerun()


# =====================================================
# 8️⃣ FOOTER
# =====================================================
st.markdown("---")
st.caption("Generated by EV Insight © " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

