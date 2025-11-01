import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib, os

data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ev_battery_data_sample_small.csv')
df = pd.read_csv(data_path)

features = ['battery_temperature','voltage','current','state_of_charge','avg_current','state_of_health']
X = df[features]
y = df['charge_cycles']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"Model trained. MAE={mae:.2f}, R2={r2:.3f}")

out = os.path.join(os.path.dirname(__file__), '..', 'model', 'ev_battery_model.pkl')
joblib.dump(model, out)
print('Saved model to', out)
