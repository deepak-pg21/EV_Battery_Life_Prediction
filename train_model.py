# EV Battery Life Prediction - Model Training
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
# Example: from KaggleHub or manually added
data = pd.read_csv("sample_data.csv")

# Feature-target split
X = data.drop(columns=['battery_life'])
y = data['battery_life']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model/ev_battery_model.pkl")

print("✅ Model trained and saved successfully!")
