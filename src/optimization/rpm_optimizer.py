import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from skopt import gp_minimize
from skopt.space import Real
import random

DATA_PATH = "data/processed/final_dataset.csv"


# -------------------------------
# Load trained model from MLflow
# -------------------------------
def load_best_model():
    try:
        mlflow.set_tracking_uri("./mlruns")

        # Delete corrupted experiment folder if meta.yaml is missing
        import os, shutil
        corrupted = "./mlruns/1"
        if os.path.isdir(corrupted) and not os.path.exists(f"{corrupted}/meta.yaml"):
            print("Removing corrupted MLflow experiment folder: mlruns/1")
            shutil.rmtree(corrupted)

        runs = mlflow.search_runs(experiment_names=["FanBlower_Models"])

        if runs.empty:
            raise ValueError("No runs found in experiment 'FanBlower_Models'.")

        best_run = runs.sort_values("metrics.R2", ascending=False).iloc[0]
        run_id = best_run.run_id
        model_uri = f"runs:/{run_id}/GradientBoosting"

        model = mlflow.sklearn.load_model(model_uri)
        print(f"Loaded model from run: {run_id}")
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        raise


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
# FIX: Maximize RPM while minimizing vibration
# Score = RPM / (1 + vibration) -- rewards high RPM and penalizes high vibration
# -------------------------------
def performance_score(rpm, vibration):
    return rpm / (1 + vibration)


# -------------------------------
# Helper: compute vibration for a row
# -------------------------------
def compute_vibration(row):
    return (row["A_rms"] + row["H_rms"] + row["V_rms"]) / 3


# -------------------------------
# GRID SEARCH
# Exhaustive scan over all rows
# -------------------------------
def grid_search(model, df, X):
    preds = model.predict(X)

    df["PredFlow"] = preds
    df["Vibration"] = df.apply(compute_vibration, axis=1)

    # FIX: score based on RPM and vibration (not flow)
    df["Score"] = df.apply(
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
# Sample n_iter random rows
# -------------------------------
def random_search(model, df, X, n_iter=50):
    indices = random.sample(range(len(df)), min(n_iter, len(df)))

    best_score = -np.inf
    best_rpm = None
    best_vibration = None

    for idx in indices:
        row = df.iloc[idx]

        vibration = compute_vibration(row)
        rpm = row["RPM"]

        # FIX: score based on RPM and vibration
        score = performance_score(rpm, vibration)

        if score > best_score:
            best_score = score
            best_rpm = rpm
            best_vibration = vibration

    print("\nRANDOM SEARCH RESULT")
    print("--------------------")
    print(f"RPM:       {best_rpm}")
    print(f"Vibration: {best_vibration:.4f}")
    print(f"Score:     {best_score:.4f}")

    return best_rpm


# -------------------------------
# BAYESIAN OPTIMIZATION
# FIX: Search over continuous RPM space, not row indices
# -------------------------------
def bayesian_optimization(model, df, X):
    rpm_min = float(df["RPM"].min())
    rpm_max = float(df["RPM"].max())

    # Precompute vibration for each unique RPM bin using nearest-row lookup
    def get_nearest_row(rpm_val):
        idx = (df["RPM"] - rpm_val).abs().idxmin()
        return df.iloc[idx], X.iloc[idx]

    call_count = [0]

    def objective(params):
        rpm_val = params[0]
        call_count[0] += 1

        row, x_row = get_nearest_row(rpm_val)

        vibration = compute_vibration(row)
        score = performance_score(rpm_val, vibration)

        # Minimize negative score = maximize score
        return -score

    print("\nRunning Bayesian Optimization over RPM space...")

    result = gp_minimize(
        objective,
        [Real(rpm_min, rpm_max, name="RPM")],   # FIX: continuous RPM space
        n_calls=50,                               # more calls for better coverage
        n_initial_points=10,                      # random exploration before GP fits
        noise=0.01,                               # small noise for numerical stability
        random_state=42,
        verbose=False
    )

    best_rpm = result.x[0]

    # Snap to nearest real RPM in dataset
    nearest_idx = (df["RPM"] - best_rpm).abs().idxmin()
    best_row = df.iloc[nearest_idx]
    best_vibration = compute_vibration(best_row)
    best_score = performance_score(best_row["RPM"], best_vibration)

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
    df, X = load_data()
    model = load_best_model()

    grid_rpm   = grid_search(model, df.copy(), X.copy())
    random_rpm = random_search(model, df.copy(), X.copy())
    bayes_rpm  = bayesian_optimization(model, df.copy(), X.copy())

    print("\n" + "="*40)
    print("FINAL COMPARISON")
    print("="*40)
    print(f"Grid Search RPM:          {grid_rpm}")
    print(f"Random Search RPM:        {random_rpm}")
    print(f"Bayesian Optimization RPM:{bayes_rpm}")
    print("="*40)


if __name__ == "__main__":
    main()