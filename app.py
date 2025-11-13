# =====================================================
# 🔋 EV Insight — Battery Life & Cost Predictor
# Streamlit App with 3 Models: Life, Cost, Health
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

# Title Banner
st.markdown("""
<div style='background:linear-gradient(90deg,#d7f9e9,#ecfff3); padding:15px; border-radius:10px'>
    <h1 style='color:#045c3c; text-align:center;'>🔋 EV Insight — Battery Life & Cost Predictor</h1>
    <p style='text-align:center; color:#045c3c;'>Predict EV battery charge cycles, replacement cost, and health with intelligent insights.</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 2️⃣ MODEL LOADING SECTION
# =====================================================
# All models must be saved in the "model" folder
MODEL_DIR = os.path.join(os.getcwd(), 'model')

try:
    life_model = joblib.load(os.path.join(MODEL_DIR, 'ev_life_model.pkl'))
    cost_model = joblib.load(os.path.join(MODEL_DIR, 'ev_cost_model.pkl'))
    health_model = joblib.load(os.path.join(MODEL_DIR, 'ev_health_model.pkl'))
    st.success("✅ Models loaded successfully from 'model/' folder.")
except Exception as e:
    st.error(f"⚠️ Could not load models: {e}")
    st.stop()


# =====================================================
# 3️⃣ DATA HANDLING SECTION
# =====================================================
st.subheader("📂 Upload or Use Sample Dataset")

# Allow user to upload their own CSV file
uploaded = st.file_uploader("Upload your EV dataset (CSV)", type=["csv"])

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success("✅ Uploaded dataset successfully loaded.")
    except Exception as e:
        st.error(f"Error loading your file: {e}")
        st.stop()
else:
    # Use default dataset if no upload provided
    DATA_PATH = os.path.join(os.getcwd(), 'data', 'merged_ev_data.csv')
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        st.info("ℹ️ Using sample dataset (merged_ev_data.csv)")
    else:
        st.error("❌ No dataset found. Please upload a CSV file.")
        st.stop()

# Display first 8 rows of dataset
st.dataframe(df.head(8))


# =====================================================
# 4️⃣ USER SELECTION FOR PREDICTION
# =====================================================
st.markdown("---")
st.subheader("🔍 Select a Sample for Prediction")

# User can pick a specific row index to test
idx = st.number_input("Row Index (0-based)", min_value=0, max_value=max(0, len(df) - 1), value=0)
selected = df.iloc[[idx]]

# Important input features used by models
feat_cols = [
    'battery_temperature', 'voltage', 'current', 'state_of_charge',
    'avg_current', 'state_of_health', 'mileage_km', 'age_months'
]


# =====================================================
# 5️⃣ MODEL PREDICTIONS
# =====================================================
st.markdown("---")
st.subheader("📊 Prediction Results")

try:
    # Extract selected row data
    Xsel = selected[feat_cols].values

    # Predict charge cycles (life model)
    pred_cycles = life_model.predict(Xsel)[0]

    # Predict battery replacement cost (cost model)
    pred_cost = cost_model.predict(Xsel)[0]

    # Display results in metric boxes
    st.metric("Predicted Remaining Charge Cycles", f"{pred_cycles:.0f}")
    st.metric("Estimated Battery Replacement Cost (USD)", f"${pred_cost:,.2f}")

    # Combine predictions + SOH for health model
    hybrid_in = np.array([[pred_cycles, pred_cost, selected['state_of_health'].values[0]]])
    health = health_model.predict(hybrid_in)[0]

    # Display health index
    st.metric("Battery Health Index (0-100)", f"{health:.1f}%")

except Exception as e:
    st.error(f"Prediction failed: {e}")


# =====================================================
# 6️⃣ VISUALIZATION SECTION (Degradation Plot)
# =====================================================
st.markdown("---")
st.subheader("📈 Battery Degradation Trend")

try:
    # Predict all samples for visualization
    preds = life_model.predict(df[feat_cols].values)
    df['predicted_cycles'] = preds

    # Create line plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df.index, df['predicted_cycles'], color='#0b6b3a', linewidth=2)
    ax.set_title("Predicted Battery Cycle Degradation")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Predicted Remaining Cycles")
    ax.grid(True)

    # Display the plot
    st.pyplot(fig)

except Exception as e:
    st.error(f"Plotting failed: {e}")


# =====================================================
# 7️⃣ CHATBOT SECTION
# =====================================================
st.markdown("---")
st.subheader("🤖 AI Chat Assistant")

# Store conversation in session
if 'chat' not in st.session_state:
    st.session_state['chat'] = [
        {'role': 'assistant', 'content': 'Hi there! Ask me about EV battery life, cost, or maintenance tips.'}
    ]

# Display chat messages
for msg in st.session_state['chat']:
    if msg['role'] == 'assistant':
        st.info(f"**Assistant:** {msg['content']}")
    else:
        st.success(f"**You:** {msg['content']}")

# Text input for new message
user_input = st.text_input("Ask your question...")

# Chat response logic
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

    # Append answer and refresh UI
    st.session_state['chat'].append({'role': 'assistant', 'content': ans})
    st.rerun()


# =====================================================
# 8️⃣ FOOTER
# =====================================================
st.markdown("---")
st.caption("Generated by EV Insight © " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
