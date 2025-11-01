import streamlit as st
import pandas as pd, numpy as np, os, joblib, matplotlib.pyplot as plt

st.set_page_config(page_title='EV Battery Life Prediction', layout='wide')
st.title('🔋 EV Battery Life Prediction — Advanced')

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'ev_battery_model.pkl')
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

col1, col2 = st.columns([2,1])

with col1:
    st.header('Dataset')
    uploaded = st.file_uploader('Upload EV dataset CSV (or use sample)', type=['csv'])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.success('Loaded uploaded dataset')
    else:
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'ev_battery_data_sample_small.csv'))
        st.info('Using included sample dataset for demo.')

    st.dataframe(df.head(10))
    st.markdown('---')
    st.subheader('Select row for prediction')
    idx = st.number_input('Row index (0-based)', min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
    selected = df.iloc[[idx]]

with col2:
    st.header('Prediction & Health')
    if model is None:
        st.error('Model not found. Run training script to create model.')
    else:
        feat_cols = ['battery_temperature','voltage','current','state_of_charge','avg_current','state_of_health']
        Xsel = selected[feat_cols].values
        pred = model.predict(Xsel)[0]
        st.metric('Predicted Remaining Charge Cycles', f'{pred:.0f}')
        soh = float(selected['state_of_health'].values[0]) if 'state_of_health' in selected.columns else 80.0
        est_health = max(0.0, min(100.0, soh * (pred / 3000.0)))
        st.metric('Estimated Battery Health (%)', f'{est_health:.2f}%')
        st.markdown('Note: Health% is a derived estimate for demo purposes.')

st.markdown('---')
st.header('Degradation Trend (sample)')
sample_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'ev_battery_data_sample_small.csv'))
feat_cols = ['battery_temperature','voltage','current','state_of_charge','avg_current','state_of_health']
if model is not None:
    preds = model.predict(sample_df[feat_cols].values)
    sample_df['predicted_cycles'] = preds
    fig, ax = plt.subplots(figsize=(9,3))
    ax.plot(sample_df.index, sample_df['predicted_cycles'])
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Predicted Remaining Cycles')
    st.pyplot(fig)
else:
    st.info('Model missing - cannot show trend.')

st.markdown('---')
if st.button('Retrain model on included sample'):
    import subprocess, sys
    train_script = os.path.join(os.path.dirname(__file__), '..', 'model', 'train_model.py')
    res = subprocess.run([sys.executable, train_script], capture_output=True, text=True)
    st.text(res.stdout)
    if res.returncode == 0:
        st.success('Retrained model. Refresh to load new model.')
    else:
        st.error('Retraining failed. See output.')
