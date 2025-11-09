import streamlit as st, pandas as pd, numpy as np, joblib, os, matplotlib.pyplot as plt
from datetime import datetime
st.set_page_config(page_title="EV Insight", page_icon="🔋", layout="wide")
st.markdown("<div style='background:linear-gradient(90deg,#f0fff4,#e6fbef); padding:12px; border-radius:8px'><h1 style='color:#0b6b3a'>🔋 EV Insight — Battery Life & Cost Predictor</h1></div>", unsafe_allow_html=True)
MODEL_DIR = os.path.join(os.path.dirname(__file__),'..','model')
life_model = joblib.load(os.path.join(MODEL_DIR,'ev_life_model.pkl'))
cost_model = joblib.load(os.path.join(MODEL_DIR,'ev_cost_model.pkl'))
health_model = joblib.load(os.path.join(MODEL_DIR,'ev_health_model.pkl'))
col1, col2 = st.columns([2,1])
with col1:
    st.subheader("Dataset")
    uploaded = st.file_uploader("Upload EV CSV (optional)", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success("Loaded uploaded dataset")
        except Exception as e:
            st.error("Could not read uploaded file: "+str(e))
            df = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','data','merged_ev_data.csv'))
    else:
        df = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','data','merged_ev_data.csv'))
        st.info("Using demo merged dataset")
    st.dataframe(df.head(8))
    st.markdown("---")
    st.subheader("Select row for prediction")
    idx = st.number_input("Row index (0-based)", min_value=0, max_value=max(0,len(df)-1), value=0)
    selected = df.iloc[[idx]]
with col2:
    st.subheader("Predictions")
    feat_cols = ['battery_temperature','voltage','current','state_of_charge','avg_current','state_of_health','mileage_km','age_months']
    Xsel = selected[feat_cols].values
    pred_cycles = life_model.predict(Xsel)[0]
    pred_cost = cost_model.predict(Xsel)[0]
    st.metric("Predicted Remaining Charge Cycles", f"{pred_cycles:.0f}")
    st.metric("Estimated Battery Replacement Cost (USD)", f"${pred_cost:,.2f}")
    hybrid_in = np.array([[pred_cycles, pred_cost, selected['state_of_health'].values[0]]])
    health = health_model.predict(hybrid_in)[0]
    st.metric("Battery Health Index (0-100)", f"{health:.1f}%")
st.markdown("---")
st.header("Degradation Trend (demo)")
sample = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','data','merged_ev_data.csv'))
feat_cols = ['battery_temperature','voltage','current','state_of_charge','avg_current','state_of_health','mileage_km','age_months']
preds = life_model.predict(sample[feat_cols].values)
sample['predicted_cycles'] = preds
fig, ax = plt.subplots(figsize=(10,3))
ax.plot(sample.index, sample['predicted_cycles'], color='#0b6b3a')
ax.set_xlabel('Index'); ax.set_ylabel('Predicted Remaining Cycles')
st.pyplot(fig)
st.markdown('---')
st.header('AI Chat (rule-based)')
if 'chat' not in st.session_state:
    st.session_state['chat'] = [{'role':'assistant','content':'Hello — ask me about battery life, charging, or cost.'}]
for m in st.session_state['chat']:
    if m['role']=='assistant':
        st.info('**Assistant:** '+m['content'])
    else:
        st.success('**You:** '+m['content'])
q = st.text_input('Ask a question...', '')
if q:
    st.session_state['chat'].append({'role':'user','content':q})
    text = q.lower()
    if 'charge' in text:
        ans = 'Avoid frequent fast charging; prefer moderate charging to extend battery life.'
    elif 'temperature' in text:
        ans = 'High temperature accelerates degradation; keep battery cool.'
    elif 'cost' in text or 'price' in text:
        ans = 'Upload your price dataset to refine cost predictions.'
    else:
        ans = 'I can help with battery life drivers, charging tips, and cost estimation.'
    st.session_state['chat'].append({'role':'assistant','content':ans})
    st.experimental_rerun()
st.write('Generated: '+datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
