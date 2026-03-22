<div align="center">

# 🌀 Bayesian Optimization for Efficient Fan-Blower Performance Tuning

<img src="https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
<img src="https://img.shields.io/badge/Streamlit-1.45-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
<img src="https://img.shields.io/badge/AWS-EC2%20Deployed-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white"/>
<img src="https://img.shields.io/badge/DVC-Data%20Versioned-945DD6?style=for-the-badge&logo=dvc&logoColor=white"/>
<img src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?style=for-the-badge&logo=mlflow&logoColor=white"/>
<img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white"/>

<br/>

> **An end-to-end production ML system** that predicts fan-blower flow rate from raw vibration signals using FFT feature extraction, Bayesian optimization, and a full MLOps stack — deployed live on AWS.

<br/>

![Demo](https://img.shields.io/badge/🚀%20Live%20Demo-32.192.12.46:8501-success?style=for-the-badge)
![API](https://img.shields.io/badge/📡%20FastAPI%20Docs-32.192.12.46:8000/docs-informational?style=for-the-badge)

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Tech Stack](#-tech-stack)
- [ML Pipeline](#-ml-pipeline)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [MLOps Stack](#-mlops-stack)
- [Deployment](#-deployment)

---

## 🎯 Project Overview

Fan-blower systems are critical industrial components where **flow rate optimization** directly impacts energy efficiency and operational safety. Traditional measurement approaches require expensive hardware sensors — this project eliminates that by **predicting flow rate from vibration signals alone**.

### The Problem
- Flow rate measurement requires costly dedicated sensors
- Vibration data is already collected for maintenance monitoring
- No existing system correlates vibration patterns to flow rate

### The Solution
A complete ML system that:
1. **Ingests** raw vibration time-series signals from 3 axes (Axial, Horizontal, Vertical)
2. **Extracts** 37 statistical + FFT features automatically
3. **Predicts** flow rate in real-time via a REST API
4. **Optimizes** RPM settings using Bayesian Optimization to maximize performance

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│                   Streamlit App (Port 8501)                     │
│         Upload 3 CSV files → Get Flow Rate Prediction           │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP REST API
┌─────────────────────▼───────────────────────────────────────────┐
│                    FASTAPI BACKEND (Port 8000)                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │             Feature Extraction Pipeline                  │   │
│  │  Raw Signal → Parse CSV → Extract RPM/Axis               │   │
│  │  → Time Domain Features → FFT Features → 37 Features    │   │
│  └─────────────────────┬────────────────────────────────────┘   │
│                        │                                        │
│  ┌─────────────────────▼────────────────────────────────────┐   │
│  │           ML Model (GradientBoosting)                    │   │
│  │           R² = 0.9978  |  MAE = 0.567                   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                    MLOPS INFRASTRUCTURE                         │
│  DVC (Data) → DagsHub (Remote) → MLflow (Experiments)          │
│  GitHub Actions (CI/CD) → Docker → AWS EC2                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔬 **Auto Feature Extraction** | Parses OMNITREND CSV format, auto-detects RPM & axis from filename |
| 📊 **37 Engineered Features** | RMS, mean, std, min, max, peak, skewness, kurtosis, crest factor, dominant frequency, spectral centroid, spectral entropy per axis |
| 🤖 **6 ML Models Compared** | Linear Regression, Random Forest, Gradient Boosting, SVR, KNN, XGBoost |
| 🎯 **Bayesian Optimization** | Grid Search, Random Search, and Gaussian Process Bayesian Optimization for RPM tuning |
| 📡 **REST API** | FastAPI with `/predict`, `/health`, `/model-info` endpoints |
| 🎨 **Interactive Dashboard** | Streamlit UI with signal visualization, FFT plots, prediction history |
| 🐳 **Dockerized** | Multi-container setup with docker-compose |
| ☁️ **Cloud Deployed** | Live on AWS EC2 with Elastic IP |
| 🔄 **Full CI/CD** | Automated pipeline with GitHub Actions + DVC |

---

## 🛠️ Tech Stack

### Machine Learning
| Tool | Purpose |
|---|---|
| `scikit-learn` | ML models (Random Forest, Gradient Boosting, SVR, KNN) |
| `xgboost` | XGBoost regressor |
| `scipy` | FFT computation, statistical features |
| `scikit-optimize` | Bayesian Optimization (Gaussian Process) |
| `numpy / pandas` | Data processing |

### MLOps
| Tool | Purpose |
|---|---|
| `MLflow` | Experiment tracking, model registry |
| `DVC` | Data versioning and pipeline management |
| `DagsHub` | Remote storage for data + MLflow server |
| `GitHub Actions` | CI/CD pipeline automation |

### Application
| Tool | Purpose |
|---|---|
| `FastAPI` | REST API backend |
| `Streamlit` | Interactive frontend |
| `Plotly` | Signal and FFT visualizations |
| `Docker + Compose` | Containerization |
| `AWS EC2` | Cloud deployment |

---

## 🧠 ML Pipeline

### 1. Raw Signal Processing
```
OMNITREND CSV File
       ↓
Parse Header → Extract RPM + Axis (Horizontal/Vertical/Axial)
       ↓
Extract Signal Values (14,745 samples per file)
```

### 2. Feature Engineering (Per Axis)
```
Time Domain (7 features):         Frequency Domain (5 features):
├── Mean                          ├── Dominant Frequency
├── Std Deviation                 ├── Spectral Centroid
├── RMS                           ├── Spectral Entropy
├── Min / Max                     ├── FFT Mean Magnitude
├── Peak (|max|)                  └── FFT Max Magnitude
├── Skewness
├── Kurtosis
└── Crest Factor

Total: 12 features × 3 axes + 1 RPM = 37 features
```

### 3. Model Training & Selection
```
6 Models Trained → MLflow Tracking → Best Model Selected
       ↓
GradientBoostingRegressor (Best)
R² = 0.9978 | MAE = 0.567 | MSE = 0.721
```

### 4. RPM Optimization
```
Grid Search    → Exhaustive scan over all RPM/vibration combinations
Random Search  → Stochastic sampling (n=50 iterations)
Bayesian Opt   → Gaussian Process over continuous RPM space (n=50 calls)
       ↓
Score = RPM / (1 + Vibration)  ← Maximize RPM, Minimize Vibration
```

---

## 📈 Results

### Model Comparison

| Model | R² Score | MAE | MSE |
|---|---|---|---|
| **Gradient Boosting** ⭐ | **0.9978** | **0.567** | **0.721** |
| Random Forest | 0.9972 | 0.548 | 0.719 |
| XGBoost | 0.9965 | 0.671 | 0.872 |
| KNN | 0.9821 | 1.203 | 2.341 |
| Linear Regression | 0.9654 | 2.108 | 5.672 |
| SVR | 0.9512 | 2.876 | 7.123 |

### Prediction Accuracy (Live Tests)
- At **100 RPM**: Predicted `~1.2 m³/s` ✅
- At **225 RPM**: Predicted `3.8633 m³/s` ✅
- At **250 RPM**: Predicted `3.8032 m³/s` ✅

---

## 📁 Project Structure

```
📦 Bayesian-Optimization-for-Efficient-Fan-Blower-Performance-Tuning/
│
├── 📂 src/
│   ├── 📂 feature_engineering/
│   │   └── build_dataset.py        # FFT feature extraction pipeline
│   ├── 📂 training/
│   │   └── train_models.py         # Multi-model training + MLflow
│   ├── 📂 optimization/
│   │   └── rpm_optimizer.py        # Grid/Random/Bayesian optimization
│   └── 📂 preprocessing/
│
├── 📂 fanblower_app/
│   ├── feature_extractor.py        # Shared feature extraction module
│   ├── main.py                     # FastAPI backend
│   ├── app.py                      # Streamlit frontend
│   └── __init__.py
│
├── 📂 data/
│   ├── raw/                        # Raw OMNITREND CSV (DVC tracked)
│   ├── raw_signals/                # Signal data (DVC tracked)
│   ├── flowrate/                   # Flow rate reference data
│   └── processed/
│       └── final_dataset.csv       # Engineered feature dataset
│
├── 📂 models/
│   └── best_model.pkl              # Trained GradientBoosting model
│
├── 📂 artifacts/plots/             # Training visualization plots
├── 📂 .github/workflows/
│   └── ml-pipeline.yml             # GitHub Actions CI/CD
├── 📂 .streamlit/
│   └── config.toml
│
├── 🐳 Dockerfile.api               # FastAPI container
├── 🐳 Dockerfile.streamlit         # Streamlit container
├── 🐳 docker-compose.yml           # Multi-container orchestration
├── ⚙️  dvc.yaml                    # DVC pipeline definition
├── ⚙️  params.yaml                 # Pipeline parameters
└── 📋 requirements.txt
```

---

## 🚀 Quick Start

### Option 1 — Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/aashuparate12/Bayesian-Optimization-for-Efficient-Fan-Blower-Performance-Tuning-.git
cd Bayesian-Optimization-for-Efficient-Fan-Blower-Performance-Tuning-

# Set environment variables
export DAGSHUB_USERNAME=your_username
export DAGSHUB_TOKEN=your_token

# Pull and run
docker-compose pull
docker-compose up -d

# Open app
# Streamlit → http://localhost:8501
# FastAPI   → http://localhost:8000/docs
```

### Option 2 — Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Pull data
dvc pull

# Run pipeline
dvc repro

# Terminal 1 — FastAPI
uvicorn fanblower_app.main:app --reload --port 8000

# Terminal 2 — Streamlit
streamlit run fanblower_app/app.py
```

### Using the App
1. Go to `http://localhost:8501`
2. Upload **3 CSV files** — one per axis (Axial, Horizontal, Vertical)
3. Click **🚀 Predict Flow Rate**
4. View prediction, features, and vibration health indicators

---

## 📡 API Reference

### `POST /predict`
Upload 3 vibration CSV files → returns predicted flow rate.

```bash
curl -X POST "http://32.192.12.46:8000/predict" \
  -F "files=@100A1.csv" \
  -F "files=@100H1.csv" \
  -F "files=@100V1.csv"
```

**Response:**
```json
{
  "predicted_flow_rate": 3.8633,
  "rpm": 225,
  "unit": "m³/s",
  "timestamp": "2026-03-20T07:58:36",
  "features": {
    "RPM": 225.0,
    "A_rms": 0.093,
    "H_rms": 0.154,
    "V_rms": 0.091
  }
}
```

### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

### `GET /model-info`
```json
{
  "model_type": "GradientBoostingRegressor",
  "feature_count": 37
}
```

---

## ⚙️ MLOps Stack

### Data Versioning with DVC
```bash
dvc add data/raw data/raw_signals data/flowrate models/
dvc push   # Push to DagsHub remote
dvc pull   # Pull in CI/CD
```

### Experiment Tracking with MLflow
All 6 models tracked with parameters, metrics, and artifact plots at:
`https://dagshub.com/aashuparate12/Bayesian-Optimization-for-Efficient-Fan-Blower-Performance-Tuning-.mlflow`

### CI/CD Pipeline
```
Push to main
    ↓
GitHub Actions triggers
    ↓
Install deps → Configure DVC remote → dvc pull → dvc repro -f
    ↓
Pipeline verified ✅
```

---

## ☁️ Deployment

### AWS EC2
- **Instance**: t2.micro (Free Tier)
- **OS**: Ubuntu 24.04 LTS
- **Elastic IP**: `32.192.12.46` (fixed, never changes)
- **Ports Open**: 8000 (API), 8501 (Streamlit)

### Docker Hub
```bash
docker pull swapnil8848/fanblower-api:latest
docker pull swapnil8848/fanblower-streamlit:latest
```

### One-Command Deploy on Any Server
```bash
mkdir fanblower && cd fanblower
# Create docker-compose.yml with your image names
docker-compose pull && docker-compose up -d
```

---

<div align="center">

**Built with ❤️ by Swapnil Parate**

[![GitHub](https://img.shields.io/badge/GitHub-swapnilparate87-181717?style=for-the-badge&logo=github)](https://github.com/swapnilparate87)
[![DagsHub](https://img.shields.io/badge/DagsHub-Experiments-945DD6?style=for-the-badge&logo=dvc)](https://dagshub.com/aashuparate12)

*⭐ Star this repo if you found it helpful!*

</div>