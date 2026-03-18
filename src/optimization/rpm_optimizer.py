import numpy as np
import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
from skopt import gp_minimize
from skopt.space import Real
import random

from dotenv import load_dotenv
load_dotenv()  # loads .env file automatically

DATA_PATH  = "data/processed/final_dataset.csv"
MODEL_PATH = "models/best_model.pkl"

# ==============================
# MLflow configuration
# FIX: Same DagsHub URI as train_models.py
# FIX: os.environ key must be env var NAME not your username
# ==============================

DAGSHUB_USERNAME = os.environ["DAGSHUB_USERNAME"]
DAGSHUB_REPO     = "Bayesian-Optimization-for-Efficient-Fan-Blower-Performance-Tuning-"

mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow"
)
mlflow.set_experiment("FanBlower_Models")


# -------------------------------
# FIX: Load model from joblib file instead of MLflow experiment
# This works in CI because models/best_model.pkl is tracked by DVC
# -------------------------------
def load_best_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. "
            "Run train_models.py first, then `dvc add models/ && dvc push`."
        )
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from: {MODEL_PATH}")
    return model


# -------------------------------
# Load dataset
# -------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.replace(" ", "")
    X = df.drop("FlowRate", axis=1)
    return df, X


# -------------------------------
# Performance score
# -------------------------------
def performance_score(rpm, vibration):
    return rpm / (1 + vibration)


def compute_vibration(row):
    return (row["A_rms"] + row["H_rms"] + row["V_rms"]) / 3


# -------------------------------
# GRID SEARCH
# -------------------------------
def grid_search(model, df, X):
    preds        = model.predict(X)
    df["PredFlow"]  = preds
    df["Vibration"] = df.apply(compute_vibration, axis=1)
    df["Score"]     = df.apply(
        lambda row: performance_score(row["RPM"], row["Vibration"]), axis=1
    )
    best = df.loc[df["Score"].idxmax()]
    print("\nGRID SEARCH RESULT")
    print("------------------")
    print(f"RPM:       {best['RPM']}")
    print(f"Vibration: {best['Vibration']:.4f}")
    print(f"Score:     {best['Score']:.4f}")
    return best["RPM"]


# -------------------------------
# RANDOM SEARCH
# -------------------------------
def random_search(model, df, X, n_iter=50):
    indices        = random.sample(range(len(df)), min(n_iter, len(df)))
    best_score     = -np.inf
    best_rpm       = None
    best_vibration = None

    for idx in indices:
        row       = df.iloc[idx]
        vibration = compute_vibration(row)
        rpm       = row["RPM"]
        score     = performance_score(rpm, vibration)
        if score > best_score:
            best_score     = score
            best_rpm       = rpm
            best_vibration = vibration

    print("\nRANDOM SEARCH RESULT")
    print("--------------------")
    print(f"RPM:       {best_rpm}")
    print(f"Vibration: {best_vibration:.4f}")
    print(f"Score:     {best_score:.4f}")
    return best_rpm


# -------------------------------
# BAYESIAN OPTIMIZATION
# -------------------------------
def bayesian_optimization(model, df, X):
    rpm_min = float(df["RPM"].min())
    rpm_max = float(df["RPM"].max())

    def get_nearest_row(rpm_val):
        idx = (df["RPM"] - rpm_val).abs().idxmin()
        return df.iloc[idx], X.iloc[idx]

    def objective(params):
        rpm_val    = params[0]
        row, x_row = get_nearest_row(rpm_val)
        vibration  = compute_vibration(row)
        score      = performance_score(rpm_val, vibration)
        return -score

    print("\nRunning Bayesian Optimization over RPM space...")

    result = gp_minimize(
        objective,
        [Real(rpm_min, rpm_max, name="RPM")],
        n_calls=50,
        n_initial_points=10,
        noise=0.01,
        random_state=42,
        verbose=False
    )

    best_rpm       = result.x[0]
    nearest_idx    = (df["RPM"] - best_rpm).abs().idxmin()
    best_row       = df.iloc[nearest_idx]
    best_vibration = compute_vibration(best_row)
    best_score     = performance_score(best_row["RPM"], best_vibration)

    print("\nBAYESIAN OPTIMIZATION RESULT")
    print("----------------------------")
    print(f"RPM (raw):  {best_rpm:.2f}")
    print(f"RPM (snap): {best_row['RPM']}")
    print(f"Vibration:  {best_vibration:.4f}")
    print(f"Score:      {best_score:.4f}")
    return best_row["RPM"]


# -------------------------------
# MAIN
# -------------------------------
def main():
    df, X      = load_data()
    model      = load_best_model()
    grid_rpm   = grid_search(model, df.copy(), X.copy())
    random_rpm = random_search(model, df.copy(), X.copy())
    bayes_rpm  = bayesian_optimization(model, df.copy(), X.copy())

    print("\n" + "=" * 40)
    print("FINAL COMPARISON")
    print("=" * 40)
    print(f"Grid Search RPM:           {grid_rpm}")
    print(f"Random Search RPM:         {random_rpm}")
    print(f"Bayesian Optimization RPM: {bayes_rpm}")
    print("=" * 40)


if __name__ == "__main__":
    main()