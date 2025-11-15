# 🔋 EV Insight — Battery Life, Cost & Health Predictor

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

![EV Insight Banner](https://user-images.githubusercontent.com/000000/ev-banner-green.png)

EV Insight is a predictive maintenance and analytics web app for electric vehicles (EVs).  
It forecasts remaining battery charge cycles, estimates battery replacement cost, and computes a Battery Health Index (BHI) using hybrid ML models and real-world telemetry.  
With a responsive Streamlit dashboard and optional OpenAI chatbot, it’s built for vivid insights, rapid deployment, and easy extension.

---

## 🚀 Key Features

- Battery Life Prediction: Model remaining charge/discharge cycles with real-world usage, temperature, and voltage telemetry.
- Cost Estimation: Forecast battery replacement price using model, age, brand, and regional price data.
- Battery Health Index: Aggregates predictive outputs into a concise battery health score (0–100%).
- Interactive Dashboard: Streamlit web UI with visual analytics and AI explanations.
- Hybrid ML Pipeline: Uses regression (for cycle/cost) and ensemble classification (for health score).
- AI Chatbot: OpenAI GPT-powered chat mode with local fallback for Q&A and predictive explanation.

---

## 🧠 ML Pipeline Overview

| Module               | Description                        | Algorithms / Tools             |
|----------------------|------------------------------------|-------------------------------|
| Data Preprocessing   | Clean and merge telemetry data     | pandas, numpy, sklearn         |
| Feature Engineering  | Degradation, thermal metrics       | FeatureUnion, custom scripts   |
| Life Prediction      | Remaining charge cycles            | Random Forest, XGBoost, LSTM   |
| Cost Estimation      | Replacement price projection       | Linear, CatBoost, XGBoost      |
| BHI Computation      | Composite health score             | Weighted ensemble, calibration |
| Visualization        | Charts and insights                | matplotlib, plotly, streamlit  |
| Chat Assistant       | Explain model outputs              | OpenAI API, LangChain fallback |

---

## 🖥️ Dashboard Highlights

- Sliders for EV model, mileage, temperature, and more.
- Life vs cost projection visualizations with overlays.
- Charge/thermal heatmaps and health warning alerts.
- Built-in AI chatbot for explanations and custom queries.
- Green-themed accessibility modes (light/dark).

---

## 🧭 Project Goals

- Help EV users anticipate battery degradation and maintenance needs.
- Reduce premature battery replacements with predictive analytics.
- Enable OEMs and researchers to interpret ML results for EV fleets.
- Share a complete ML → Streamlit → GitHub workflow for rapid prototyping.

---

## 🏗️ File Structure
```
EV-Insight/
│
├── data/
│   ├── raw/                 # Raw telemetry and pricing datasets (for reference or future use)
│   ├── processed/           # Cleaned and processed datasets used by app.py
│   └── sample_inputs.csv    # Sample dataset CSV for demo/testing
│
├── model/                   # Directory containing trained ML models as serialized files
│   ├── ev_life_model.pkl
│   ├── ev_cost_model.pkl
│   └── ev_health_model.pkl
│
├── app.py                   # Main Streamlit application with prediction & chatbot features
│
├── requirements.txt         # Python dependencies needed to run app.py
├── README.md                # Project overview, setup, usage, and notes
├── LICENSE                  # Project license file (MIT or preferred)
```
## ⚙️ Setup & Deployment

**Clone & Install**

git clone https://github.com/username/EV-Insight.git
cd EV-Insight
pip install -r requirements.txt


**Run Locally**

streamlit run app/dashboard.py


---

## 🔬 Tech Stack

- **Frontend:** Streamlit, Plotly, Matplotlib
- **Backend & ML:** Python, scikit-learn, XGBoost, CatBoost, pandas, numpy
- **AI Assistant:** OpenAI API, LangChain
- **Example Data:** NREL battery degradation, web-scraped EV specs and prices, simulated telemetry

---

## 🌿 Future Enhancements

- Real-time IoT telemetry integration.
- Advanced LSTM/RNN forecasting for long-term health.
- Auto-updating price data from APIs.
- Mobile web/app companion.
- Expanded chemistry and model datasets.


---

EV Insight — driving smarter, cleaner, and longer EV journeys through battery intelligence.

**Deploy Instantly**
- Streamlit Cloud: Push repo → auto-deploy.
- GitHub Codespaces: Ready-to-go in browser.
- Docker (optional):

docker build -t ev-insight .
docker run -p 8501:8501 ev-insight








