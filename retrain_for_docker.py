# Run this AFTER installing the compatible packages
# pip install scikit-learn==1.6.1 numpy==2.2.4 xgboost==2.1.4

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# Load your processed data
df = pd.read_csv("data/processed/final_dataset.csv")
df.columns = df.columns.str.replace(" ", "")

flow_col = [c for c in df.columns if "flow" in c.lower()][0]
X = df.drop([flow_col], axis=1)
y = df[flow_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train best model (GradientBoosting based on your earlier results)
model = GradientBoostingRegressor()
model.fit(X_train, y_train)

# Save with protocol 4
joblib.dump(model, "models/best_model.pkl", protocol=4)
print("Model saved successfully!")

from sklearn.metrics import r2_score
print(f"R2: {r2_score(y_test, model.predict(X_test)):.4f}")