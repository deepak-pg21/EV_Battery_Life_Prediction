#!/usr/bin/env python3
"""
EV Insight: Streamlit Dashboard with AI Chatbot
Predict battery charge cycles, replacement cost, health index, and answer user queries interactively.
Enhanced with secure OpenAI API key handling via .streamlit/secrets.toml.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from openai import OpenAI
from openai.error import OpenAIError

# ------------------------------------------------------------
# --- PAGE CONFIGURATION ---
# ------------------------------------------------------------
st.set_page_config(
    page_title="EV Insight ⚡ Battery Predictor & AI Assistant",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        st.error(f"Model file missing: {filename}. Please train models first.")
        st.stop()
    return joblib.load(path)

def predict_all_models(row):
    X = row[FEATURE_COLUMNS]
    cycles_pred = life_model.predict(X)[0]
    cost_pred = cost_model.predict(X)[0]
    health_input = pd.DataFrame([{
        "predicted_cycles": cycles_pred,
        "predicted_cost": cost_pred,
        "state_of_health": row["state_of_health"].values[0]
    }])
    health_pred = health_model.predict(health_input)[0]
    return cycles_pred, cost_pred, health_pred

def plot_degradation(df, feature_cols):
    try:
        preds = life_model.predict(df[feature_cols])
        df['predicted_cycles'] = preds

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df['predicted_cycles'], color='#047857', linewidth=2)
        ax.set_title("🔋 Predicted Battery Cycle Degradation Over Dataset")
        ax.set_xlabel("Data Index")
        ax.set_ylabel("Remaining Cycles")
        ax.grid(True)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Plotting error: {e}")

def ai_chat_response(prompt: str, client: OpenAI) -> str:
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful EV battery assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return completion.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"⚠️ OpenAI API error: {e}"

# ------------------------------------------------------------
# --- CONSTANTS ---
# ------------------------------------------------------------
MODEL_DIR = os.path.join(os.getcwd(), 'model')
DATA_DIR = os.path.join(os.getcwd(), 'data')
DATA_FILE = os.path.join(DATA_DIR, 'merged_ev_data.csv')

FEATURE_COLUMNS = [
    'battery_temperature', 'voltage', 'current',
    'state_of_charge', 'avg_current',
    'state_of_health', 'mileage_km', 'age_months'
]

# ------------------------------------------------------------
# --- HEADER & INTRO ---
# ------------------------------------------------------------
st.markdown("""
<div style="background:linear-gradient(90deg, #d7f9e9, #ecfff3); padding: 20px; border-radius: 12px; box-shadow: 3px 3px 12px #b7d7c3;">
<h1 style="color: #065f46; text-align: center;">🔋 EV Insight — Battery Life, Cost & Health Predictor</h1>
<p style="text-align: center; font-size: 1.1rem; color: #065f46;">
Predict remaining battery charge cycles, estimate replacement cost, and monitor health with intelligent insights and AI assistance.
</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# --- Load ML Models ---
# ------------------------------------------------------------
try:
    life_model = load_model('ev_life_model.pkl')
    cost_model = load_model('ev_cost_model.pkl')
    health_model = load_model('ev_health_model.pkl')
    st.success("✅ Machine Learning models loaded successfully")
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# ------------------------------------------------------------
# --- Load Dataset ---
# ------------------------------------------------------------
try:
    if not os.path.exists(DATA_FILE):
        st.warning("Sample dataset not found locally. Please upload your dataset below.")
        df = None
    else:
        df = pd.read_csv(DATA_FILE)
        st.info(f"Sample dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    df = None

# ------------------------------------------------------------
# --- Data Upload ---
# ------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your EV dataset (.csv format)", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✔️ Uploaded dataset loaded successfully!")
    except Exception as e:
        st.error(f"Error parsing uploaded CSV: {e}")

if df is None:
    st.stop()

# ------------------------------------------------------------
# --- Display Dataset Preview ---
# ------------------------------------------------------------
st.subheader("🗂️ Dataset Preview (first 8 rows)")
st.dataframe(df.head(8))

# ------------------------------------------------------------
# --- User Selection for Prediction ---
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🔍 Select a Data Record to Predict")

row_index = st.number_input(
    "Choose row index (0-based):",
    min_value=0,
    max_value=len(df) - 1,
    value=0,
    step=1
)
selected_row = df.iloc[[row_index]]

# Validate selected features presence
missing_cols = [c for c in FEATURE_COLUMNS + ['state_of_health'] if c not in selected_row.columns]
if missing_cols:
    st.error(f"Selected data missing columns: {missing_cols}")
    st.stop()

# ------------------------------------------------------------
# --- Perform Predictions ---
# ------------------------------------------------------------
st.markdown("---")
st.subheader("📊 Model Predictions")

try:
    pred_cycles, pred_cost, pred_health = predict_all_models(selected_row)

    st.metric("🔋 Remaining Charge Cycles", f"{pred_cycles:.0f} cycles")
    st.metric("💰 Estimated Replacement Cost (USD)", f"${pred_cost:,.2f}")
    st.metric("❤️ Battery Health Index", f"{pred_health:.1f}%")
except Exception as e:
    st.error(f"Error during prediction: {e}")

# ------------------------------------------------------------
# --- Visualization of Battery Degradation ---
# ------------------------------------------------------------
st.markdown("---")
st.subheader("📈 Battery Cycle Degradation Over Dataset")
plot_degradation(df, FEATURE_COLUMNS)

# ------------------------------------------------------------
# --- AI Chatbot Integration ---
# ------------------------------------------------------------
st.markdown("---")
st.subheader("🤖 EV Battery AI Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Hello! I’m your EV Battery AI Assistant. Ask me anything about EV batteries, life, cost, or health."}
    ]

# Display chat history
for message in st.session_state.chat_history:
    if message['role'] == 'user':
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**Assistant:** {message['content']}")

# Read OpenAI key securely
openai_api_key = None
if "OPENAI_API_KEY" in st.secrets:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
else:
    openai_api_key = os.getenv("OPENAI_API_KEY")

# User Input & Chatbot Response
user_question = st.text_input("Ask your EV battery question below:")

if user_question:
    if openai_api_key:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        client = OpenAI(api_key=openai_api_key)
        with st.spinner("AI is thinking..."):
            answer = ai_chat_response(user_question, client)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.experimental_rerun()
    else:
        st.warning("OpenAI API key missing. Please configure your key in .streamlit/secrets.toml or environment variable.")
        # To avoid repeated warnings on rerun
        user_question = None

# ------------------------------------------------------------
# --- Footer ---
# ------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"<p style='text-align:center; font-size:0.9em; color:gray;'>© {datetime.now().year} EV Insight by PG Deepak Chiranjeevi. Developed for EV battery forecasting and AI assistance.</p>",
    unsafe_allow_html=True
)
