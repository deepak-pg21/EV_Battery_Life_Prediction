#!/usr/bin/env python3
"""
EV Insight: Beautiful, Inspiring Battery Health & AI Chatbot Streamlit App
Author: PG Deepak Chiranjeevi (2025)
"""

import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from transformers import pipeline, Conversation

# Page config
st.set_page_config(page_title="EV Insight ⚡ Battery & AI Assistant", page_icon="🔋", layout="wide", initial_sidebar_state="expanded")

MODEL_DIR = os.path.join(os.getcwd(), 'model')
DATA_DIR = os.path.join(os.getcwd(), 'data')
DATA_FILE = os.path.join(DATA_DIR, 'merged_ev_data.csv')

FEATURE_COLS = [
    'battery_temperature', 'voltage', 'current',
    'state_of_charge', 'avg_current', 'state_of_health',
    'mileage_km', 'age_months'
]

# Utility functions
def load_model(fname):
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        st.error(f"❌ Model missing: {fname}. Please train and save in /model/")
        st.stop()
    return joblib.load(path)

def predict_all(row):
    X = row[FEATURE_COLS]
    cycles = life_model.predict(X)[0]
    cost = cost_model.predict(X)[0]
    health_input = pd.DataFrame([{"predicted_cycles": cycles, "predicted_cost": cost, "state_of_health": row["state_of_health"].values[0]}])
    health = health_model.predict(health_input)[0]
    return cycles, cost, health

def plot_degradation(df):
    try:
        preds = life_model.predict(df[FEATURE_COLS])
        df['predicted_cycles'] = preds
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df.index, df['predicted_cycles'], color='#16a34a', linewidth=2)
        ax.set_title("🔋 Predicted Battery Cycle Degradation")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Remaining Cycles")
        ax.grid(True)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Plot error: {e}")

@st.cache_resource(show_spinner=False)
def load_chatbot():
    return pipeline("conversational", model="microsoft/DialoGPT-medium")

hf_chatbot = load_chatbot()

def hf_chat_response(user_input):
    conversation = Conversation(user_input)
    response = hf_chatbot(conversation)
    return response.generated_responses[-1]

# Hero banner
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap');

.hero {
  font-family: 'Montserrat', sans-serif;
  color: #065f46;
  text-align: center;
  padding: 3rem 1rem 2rem 1rem;
  background: linear-gradient(90deg, #e8f5e9, #d0f0c0);
  border-radius: 15px;
  box-shadow: 0 8px 24px rgba(6, 95, 70, 0.15);
  margin-bottom: 2rem;
}

.tagline {
  font-size: 1.3rem;
  font-weight: 500;
  margin-top: 0.5rem;
  color: #2d6a4f;
  font-style: italic;
}

.metrics-wrapper .stMetric {
  background: #bbf7d0;
  border-radius: 15px;
  padding: 18px 25px;
  box-shadow: 2px 2px 8px rgba(6, 95, 70, 0.15);
  border-left: 8px solid #22c55e;
  font-weight: 800;
  font-size: 1.7rem;
  color: #065f46;
  margin-bottom: 15px;
}

.chat-container {
  background: #ecfdf5;
  border-radius: 12px;
  padding: 1.5rem;
  box-shadow: 1px 1px 15px #94d3ac80;
  max-height: 400px;
  overflow-y: auto;
}

.user-msg {
  background-color: #bbf7d0;
  border-radius: 20px 20px 0 20px;
  padding: 0.8rem 1.2rem;
  margin: 8px 0;
  max-width: 80%;
  align-self: flex-end;
  color: #065f46;
  font-weight: 600;
}

.bot-msg {
  background-color: white;
  border-radius: 20px 20px 20px 0;
  padding: 0.8rem 1.2rem;
  margin: 8px 0;
  max-width: 80%;
  align-self: flex-start;
  color: #334e3e;
}

.chat-input {
  padding: 0.5rem 1rem;
  width: 100%;
  border-radius: 10px;
  border: 2px solid #16a34a;
  font-size: 1.1rem;
}

</style>

<div class="hero">
  <h1>🔋 EV Insight — Your Electric Vehicle Battery Companion</h1>
  <div class="tagline">Predict life, cost, health and chat with your AI Battery Expert</div>
</div>
""", unsafe_allow_html=True)

# Load ML models
life_model = load_model('ev_life_model.pkl')
cost_model = load_model('ev_cost_model.pkl')
health_model = load_model('ev_health_model.pkl')
st.success("✅ ML models loaded.")

# Load or upload dataset
if not os.path.exists(DATA_FILE):
    st.warning("Sample dataset not found! Please upload your CSV file below.")
    df = None
else:
    df = pd.read_csv(DATA_FILE)
    st.info(f"Sample dataset loaded: {df.shape[0]} rows")

uploaded = st.file_uploader("Upload EV dataset (.csv)", type="csv")
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.success("✔️ Uploaded dataset loaded!")
    except Exception as e:
        st.error(f"Failed to parse uploaded CSV: {e}")

if df is None:
    st.stop()

# Dataset preview
st.subheader("🗂️ Dataset Preview")
st.dataframe(df.head(8))

# Selection & Predictions
st.markdown("---")
st.subheader("🔍 Select a data row to predict")

idx = st.number_input("Row index (0-based)", min_value=0, max_value=len(df)-1, value=0)
selected_row = df.iloc[[idx]]

missing = [c for c in FEATURE_COLS+['state_of_health'] if c not in selected_row.columns]
if missing:
    st.error(f"Dataset missing these columns: {missing}")
    st.stop()

st.markdown("---")
st.subheader("📊 Battery Prediction Metrics")

try:
    cycles, cost, health = predict_all(selected_row)

    col1, col2, col3 = st.columns(3)
    col1.metric("🔋 Remaining Charge Cycles", f"{cycles:.0f} cycles")
    col2.metric("💰 Estimated Replacement Cost", f"${cost:,.2f}")
    col3.metric("❤️ Battery Health Index", f"{health:.1f}%")

except Exception as e:
    st.error(f"Prediction failed: {e}")

st.markdown("---")
st.subheader("📈 Battery Cycle Degradation Across Dataset")
plot_degradation(df)

# Chatbot Section
st.markdown("---")
st.subheader("🤖 Interactive AI Battery Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role":"assistant","content":"Hello! I’m here to help with your EV battery queries."}
    ]

def display_chat():
    st.markdown('<div class="chat-container" style="display: flex; flex-direction: column;">', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-msg" style="align-self: flex-end;">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">{message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

display_chat()

user_input = st.text_input("Ask anything about EV battery life, cost, or health:", key="chat_input", placeholder="Type your question and press Enter")

if user_input:
    st.session_state.chat_history.append({"role":"user", "content": user_input})
    with st.spinner("AI is thinking..."):
        answer = hf_chat_response(user_input)
    st.session_state.chat_history.append({"role":"assistant", "content": answer})
    st.experimental_rerun()

st.markdown("---")
st.markdown(f'<p style="text-align:center; color:#94a3b8; font-size:0.9em;">© {datetime.now().year} EV Insight by PG Deepak Chiranjeevi</p>', unsafe_allow_html=True)
