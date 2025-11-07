#!/usr/bin/env python3
import os,pandas as pd,joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score
data_path = os.path.join(os.path.dirname(__file__),'..','data','merged_ev_data.csv')
df = pd.read_csv(data_path)
features = ['battery_temperature','voltage','current','state_of_charge','avg_current','state_of_health','mileage_km','age_months']
X = df[features]
y1 = df['charge_cycles']
y2 = df['battery_replacement_cost']
X_train, X_test, y_train, y_test = train_test_split(X,y1,test_size=0.2,random_state=42)
m1 = RandomForestRegressor(n_estimators=50,max_depth=10,random_state=42)
m1.fit(X_train,y_train)
print('Cycles MAE:', mean_absolute_error(y_test,m1.predict(X_test)),'R2:', r2_score(y_test,m1.predict(X_test)))
joblib.dump(m1, os.path.join(os.path.dirname(__file__),'..','model','ev_life_model.pkl'))
X_train, X_test, y_train, y_test = train_test_split(X,y2,test_size=0.2,random_state=42)
m2 = RandomForestRegressor(n_estimators=50,max_depth=10,random_state=42)
m2.fit(X_train,y_train)
print('Cost MAE:', mean_absolute_error(y_test,m2.predict(X_test)),'R2:', r2_score(y_test,m2.predict(X_test)))
joblib.dump(m2, os.path.join(os.path.dirname(__file__),'..','model','ev_cost_model.pkl'))
merged = X.copy()
merged['pred_cycles'] = m1.predict(X)
merged['pred_cost'] = m2.predict(X)
hybrid_target = (df['state_of_health'] * 0.6 + merged['pred_cycles'] / 40).clip(0,100)
m3 = RandomForestRegressor(n_estimators=30,max_depth=8,random_state=42)
m3.fit(merged[['pred_cycles','pred_cost','state_of_health']], hybrid_target)
joblib.dump(m3, os.path.join(os.path.dirname(__file__),'..','model','ev_health_model.pkl'))
print('Hybrid model trained and saved.')