import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime

# -------------------------------
# PAGE SETUP
# -------------------------------
st.set_page_config(page_title="EV Insight", page_icon="🔋", layout="wide")

st.markdown("""
<div style='background:linear-gradient(90deg,#d7f9e9,#ecfff3); padding:15px; border-radius:10px'>
    <h1 style='color:#045c3c; text-align:center;'>🔋 EV Insight — Battery Life & Cost Predictor</h1>
    <p style='text-align:center; color:#045c3c;'>Predict EV battery charge cycles, replacement cost, and health with intelligent insights.</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# MODEL LOADING
# -------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')
try:
    life_model = joblib.load(os.path.join(MODEL_DIR, 'ev_life_model.pkl'))
    cost_model = joblib.load(os.path.join(MODEL_DIR, 'ev_cost_model.pkl'))
    health_model = joblib.load(os.path.join(MODEL_DIR, 'ev_health_model.pkl'))
except Exception as e:
    st.error(f"⚠️ Model files not found or corrupted: {e}")
    st.stop()

# -------------------------------
# DATA HANDLING
# -------------------------------
st.subheader("📂 Upload or Use Sample Dataset")

uploaded = st.file_uploader("Upload your EV dataset (CSV)", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success("✅ Uploaded dataset successfully loaded.")
    except Exception as e:
        st.error(f"Error loading your file: {e}")
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'merged_ev_data.csv'))
else:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'merged_ev_data.csv'))
    st.info("ℹ️ Using sample dataset (merged_ev_data.csv)")

st.dataframe(df.head(8))

# -------------------------------
# USER INPUT
# -------------------------------
st.markdown("---")
st.subheader("🔍 Select a Sample for Prediction")

idx = st.number_input("Row Index (0-based)", min_value=0, max_value=max(0, len(df) - 1), value=0)
selected = df.iloc[[idx]]

feat_cols = [
    'battery_temperature', 'voltage', 'current', 'state_of_charge',
    'avg_current', 'state_of_health', 'mileage_km', 'age_months'
]

# -------------------------------
# MODEL PREDICTIONS
# -------------------------------
st.markdown("---")
st.subheader("📊 Prediction Results")

try:
    Xsel = selected[feat_cols].values
    pred_cycles = life_model.predict(Xsel)[0]
    pred_cost = cost_model.predict(Xsel)[0]

    st.metric("Predicted Remaining Charge Cycles", f"{pred_cycles:.0f}")
    st.metric("Estimated Battery Replacement Cost (USD)", f"${pred_cost:,.2f}")

    hybrid_in = np.array([[pred_cycles, pred_cost, selected['state_of_health'].values[0]]])
    health = health_model.predict(hybrid_in)[0]
    st.metric("Battery Health Index (0-100)", f"{health:.1f}%")

except Exception as e:
    st.error(f"Prediction failed: {e}")

# -------------------------------
# PLOT SECTION
# -------------------------------
st.markdown("---")
st.subheader("📈 Battery Degradation Trend (Demo)")

try:
    sample = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'merged_ev_data.csv'))
    preds = life_model.predict(sample[feat_cols].values)
    sample['predicted_cycles'] = preds

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sample.index, sample['predicted_cycles'], color='#0b6b3a', linewidth=2)
    ax.set_title("Predicted Battery Cycle Degradation")
    ax.set_xlabel("Data Index")
    ax.set_ylabel("Predicted Remaining Cycles")
    ax.grid(True)
    st.pyplot(fig)
except Exception as e:
    st.error(f"Plotting failed: {e}")

# -------------------------------
# CHATBOT SECTION
# -------------------------------
st.markdown("---")
st.subheader("🤖 AI Chat Assistant")

if 'chat' not in st.session_state:
    st.session_state['chat'] = [
        {'role': 'assistant', 'content': 'Hi there! Ask me about EV battery life, cost, or maintenance tips.'}
    ]

for msg in st.session_state['chat']:
    if msg['role'] == 'assistant':
        st.info(f"**Assistant:** {msg['content']}")
    else:
        st.success(f"**You:** {msg['content']}")

user_input = st.text_input("Ask your question...")

if user_input:
    st.session_state['chat'].append({'role': 'user', 'content': user_input})
    text = user_input.lower()

    if 'charge' in text:
        ans = "Avoid frequent fast charging — it shortens battery lifespan. Prefer slow charging."
    elif 'temperature' in text:
        ans = "High temperatures accelerate degradation — store your EV in shaded or cool areas."
    elif 'cost' in text or 'price' in text:
        ans = "Battery replacement costs vary by capacity — upload your dataset for refined results."
    elif 'health' in text:
        ans = "Battery health depends on cycles, age, and temperature — your dashboard shows predictions."
    else:
        ans = "I can help with EV battery performance, cost predictions, and maintenance recommendations."

    st.session_state['chat'].append({'role': 'assistant', 'content': ans})
    st.experimental_rerun()

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("Generated by EV Insight © " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
