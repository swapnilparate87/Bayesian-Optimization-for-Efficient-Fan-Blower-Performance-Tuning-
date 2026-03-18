from fastapi import FastAPI, UploadFile, File
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from io import BytesIO

app = FastAPI(title="Fan Blower AI System")


# -------------------------------
# Load best model from MLflow
# -------------------------------


def load_model():

    runs = mlflow.search_runs(experiment_names=["FanBlower_Models"])
    best_run = runs.sort_values("metrics.R2", ascending=False).iloc[0]

    run_id = best_run.run_id

    # IMPORTANT: use correct artifact name
    model_uri = f"runs:/{run_id}/{best_run['params.model']}"

    model = mlflow.sklearn.load_model(model_uri)

    return model


# -------------------------------
# FFT Feature Extraction
# -------------------------------
def extract_fft_features(signal):

    fft_vals = np.fft.fft(signal)
    fft_vals = np.abs(fft_vals)

    features = {
        "mean": np.mean(signal),
        "std": np.std(signal),
        "rms": np.sqrt(np.mean(signal**2)),
        "kurtosis": pd.Series(signal).kurt(),
        "skew": pd.Series(signal).skew(),
        "dominant_freq": np.argmax(fft_vals)
    }

    return features


# -------------------------------
# Parse CSV
# -------------------------------

def process_file(file):

    # Fix encoding problem
    df = pd.read_csv(BytesIO(file), encoding="latin1", header=None)

    # Debug print
    print("Raw dataframe shape:", df.shape)

    # Find the row where the actual signal starts
    value_row = None
    for i, row in df.iterrows():
        if "Value" in str(row.values):
            value_row = i
            break

    if value_row is None:
        raise ValueError("Could not find 'Value' column in file")

    # Extract signal data
    signal_df = df.iloc[value_row + 1:]

    signal = signal_df.iloc[:, 0]

    signal = pd.to_numeric(signal, errors="coerce").dropna()

    print("Signal length:", len(signal))

    return signal.values

# -------------------------------
# Combine axis features
# -------------------------------
def build_feature_vector(signal):

    features = extract_fft_features(signal)

    feature_dict = {
        "A_mean": features["mean"],
        "A_std": features["std"],
        "A_rms": features["rms"],
        "A_kurtosis": features["kurtosis"],
        "A_skew": features["skew"],
        "A_dominant_freq": features["dominant_freq"],

        # TEMP FIX
        "H_mean": 0,
        "V_mean": 0,
        "RPM": 1000
    }

    return pd.DataFrame([feature_dict])
# -------------------------------
# API: Predict Flow
# -------------------------------
@app.post("/predict_flow")
async def predict_flow(file: UploadFile = File(...)):

    content = await file.read()
    signal = process_file(content)

    X = build_feature_vector(signal)

    print("Features:", X.columns)
    print("Shape:", X.shape)

    prediction = model.predict(X)[0]

    return {
        "predicted_flow_rate": float(prediction)
    }


# -------------------------------
# API: Optimize RPM
# -------------------------------
@app.post("/optimize_rpm")
async def optimize_rpm(file: UploadFile = File(...)):

    content = await file.read()
    signal = process_file(content)

    X = build_feature_vector(signal)

    flow = model.predict(X)[0]

    vibration = X["A_rms"].values[0]

    score = flow / (1 + vibration)

    return {
        "predicted_flow": float(flow),
        "vibration": float(vibration),
        "performance_score": float(score),
        "recommended_rpm": "Use optimizer module for full system"
    }