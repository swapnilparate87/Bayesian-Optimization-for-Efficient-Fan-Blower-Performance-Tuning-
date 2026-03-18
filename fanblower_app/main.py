"""
main.py  —  FastAPI backend
---------------------------
Endpoints:
  POST /predict      — upload 3 CSV files → returns predicted flow rate
  GET  /health       — health check
  GET  /model-info   — model metadata
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from datetime import datetime

from fanblower_app.feature_extractor import build_feature_row, get_signal_and_fft

# ──────────────────────────────────────────────
# App setup
# ──────────────────────────────────────────────
app = FastAPI(
    title="Fan Blower Flow Rate Prediction API",
    description="Upload 3 vibration CSV files (Axial, Horizontal, Vertical) to predict flow rate.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Load model once at startup
# ──────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "models/best_model.pkl")

@app.on_event("startup")
def load_model():
    global MODEL
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at '{MODEL_PATH}'. "
            "Run train_models.py and dvc push first."
        )
    MODEL = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")


# ──────────────────────────────────────────────
# Response schemas
# ──────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_flow_rate: float
    rpm: int
    unit: str
    timestamp: str
    features: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    model_type: str
    model_path: str
    feature_count: int


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    return HealthResponse(
        status="ok",
        model_loaded=MODEL is not None,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Monitoring"])
def model_info():
    return ModelInfoResponse(
        model_type=type(MODEL).__name__,
        model_path=MODEL_PATH,
        feature_count=len(MODEL.feature_names_in_) if hasattr(MODEL, "feature_names_in_") else -1,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    files: List[UploadFile] = File(
        ..., description="Upload exactly 3 CSV files: one per axis (Axial, Horizontal, Vertical)"
    )
):
    # ── Validate file count ──
    if len(files) != 3:
        raise HTTPException(
            status_code=422,
            detail=f"Exactly 3 files required (Axial, Horizontal, Vertical). Got {len(files)}."
        )

    # ── Validate file types ──
    for f in files:
        if not f.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=422,
                detail=f"File '{f.filename}' is not a CSV."
            )

    # ── Read file bytes (UploadFile is async) ──
    import io
    file_objects = []
    for f in files:
        content = await f.read()
        file_objects.append(io.BytesIO(content))

    # ── Feature extraction ──
    try:
        feature_df = build_feature_row(file_objects)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # ── Prediction ──
    try:
        prediction = float(MODEL.predict(feature_df)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    rpm = int(feature_df["RPM"].iloc[0])

    return PredictionResponse(
        predicted_flow_rate=round(prediction, 4),
        rpm=rpm,
        unit="m³/s",
        timestamp=datetime.utcnow().isoformat(),
        features=feature_df.iloc[0].to_dict(),
    )