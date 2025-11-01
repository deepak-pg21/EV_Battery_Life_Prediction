# EV_Battery_Life_Prediction (Advanced)

Predict remaining battery charge cycles, estimate battery health percentage, and visualize degradation trends.

## How to run locally

1. Install dependencies:
   pip install -r requirements.txt

2. Run Streamlit app:
   streamlit run app/app.py

## Notes
- The included model is trained on a small synthetic sample for demo purposes.
- To use the full dataset, place it in `data/` and run `python model/train_model.py`.
