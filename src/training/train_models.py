import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
load_dotenv()  # loads .env file automatically


# ==============================
# MLflow configuration
# FIX: Only ONE tracking URI — DagsHub (removed duplicate sqlite URI)
# FIX: os.environ key must be the env var NAME, not your username
# ==============================

DAGSHUB_USERNAME = os.environ["DAGSHUB_USERNAME"]
DAGSHUB_REPO     = "Bayesian-Optimization-for-Efficient-Fan-Blower-Performance-Tuning-"

mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
)
mlflow.set_experiment("FanBlower_Models")


# ==============================
# Paths
# ==============================

DATA_PATH = "data/processed/final_dataset.csv"

PLOT_DIR  = "artifacts/plots"
MODEL_DIR = "models"

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ==============================
# Load dataset
# ==============================

def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.replace(" ", "")

    flow_col = None
    for col in df.columns:
        if "flow" in col.lower():
            flow_col = col

    if flow_col is None:
        raise ValueError("Flow rate column not found")

    X = df.drop([flow_col], axis=1)
    y = df[flow_col]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# ==============================
# Evaluation
# ==============================

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    r2    = r2_score(y_test, preds)
    mae   = mean_absolute_error(y_test, preds)
    mse   = mean_squared_error(y_test, preds)
    return preds, r2, mae, mse


# ==============================
# Plot functions
# ==============================

def plot_predictions(y_test, preds, model_name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds)
    plt.xlabel("Actual Flow Rate")
    plt.ylabel("Predicted Flow Rate")
    plt.title(f"{model_name} Prediction vs Actual")
    path = f"{PLOT_DIR}/{model_name}_prediction.png"
    plt.savefig(path)
    plt.close()
    return path


def plot_residuals(y_test, preds, model_name):
    residuals = y_test - preds
    plt.figure(figsize=(6, 4))
    sns.histplot(residuals, kde=True)
    plt.title(f"{model_name} Residual Distribution")
    path = f"{PLOT_DIR}/{model_name}_residuals.png"
    plt.savefig(path)
    plt.close()
    return path


# ==============================
# Training pipeline
# ==============================

def main():
    X_train, X_test, y_train, y_test = load_data()

    models = {
        "LinearRegression":   LinearRegression(),
        "RandomForest":       RandomForestRegressor(n_estimators=200),
        "GradientBoosting":   GradientBoostingRegressor(),
        "SVR":                SVR(),
        "KNN":                KNeighborsRegressor(n_neighbors=5),
        "XGBoost":            xgb.XGBRegressor()
    }

    results    = []
    best_r2    = -float("inf")
    best_model = None          # FIX: track best model across all runs
    best_name  = None

    for name, model in models.items():

        with mlflow.start_run(run_name=name):

            model.fit(X_train, y_train)
            preds, r2, mae, mse = evaluate_model(model, X_test, y_test)

            mlflow.log_param("model", name)
            mlflow.log_metric("R2",  r2)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MSE", mse)

            mlflow.sklearn.log_model(model, name)

            pred_plot     = plot_predictions(y_test, preds, name)
            residual_plot = plot_residuals(y_test, preds, name)

            mlflow.log_artifact(pred_plot)
            mlflow.log_artifact(residual_plot)

            results.append({"Model": name, "R2": r2, "MAE": mae, "MSE": mse})

            print(f"\n{name}")
            print(f"R2:  {r2}")
            print(f"MAE: {mae}")
            print(f"MSE: {mse}")
            print("---------------------")

            # FIX: track which model is best by R2
            if r2 > best_r2:
                best_r2    = r2
                best_model = model
                best_name  = name

    # ==============================
    # FIX: Save best model as joblib file so rpm_optimizer can load it
    # ==============================
    best_model_path = f"{MODEL_DIR}/best_model.pkl"
    joblib.dump(best_model, best_model_path)
    print(f"\nBest model: {best_name} (R2={best_r2:.4f}) saved to {best_model_path}")

    # ==============================
    # Save model comparison table
    # ==============================
    results_df   = pd.DataFrame(results)
    results_path = f"{MODEL_DIR}/model_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print("\nModel comparison saved.")


if __name__ == "__main__":
    main()