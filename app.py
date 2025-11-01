import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="EV Battery Life Prediction", layout="wide")

st.title("🔋 EV Battery Life Prediction Dashboard")
st.write("This app predicts the remaining battery life and degradation rate of Electric Vehicles using Machine Learning.")

uploaded_file = st.file_uploader("Upload EV IoT Dataset (CSV format)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    model = joblib.load("model/ev_battery_model.pkl")

    st.subheader("🔍 Model Prediction")
    if st.button("Predict Battery Life"):
        X = df.select_dtypes(include=[np.number])
        predictions = model.predict(X)
        df['Predicted_Battery_Life'] = predictions
        st.success("Prediction complete!")
        st.dataframe(df[['Predicted_Battery_Life']].head())
else:
    st.info("Upload your dataset to begin.")
