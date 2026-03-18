# Fan Blower Flow Rate Prediction App

## Project Structure
```
fanblower_app/
├── feature_extractor.py   # Shared: parse CSV + extract FFT features
├── main.py                # FastAPI backend
├── app.py                 # Streamlit frontend
├── requirements.txt
└── README.md
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set environment variables (create .env in project root)
```
DAGSHUB_USERNAME=aashuparate12
MLFLOW_TRACKING_USERNAME=aashuparate12
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
MODEL_PATH=models/best_model.pkl
```

## Run Locally

### Terminal 1 — Start FastAPI backend
```bash
cd path/to/your/project
uvicorn fanblower_app.main:app --reload --port 8000
```

### Terminal 2 — Start Streamlit frontend
```bash
cd path/to/your/project
streamlit run fanblower_app/app.py
```

Then open: http://localhost:8501

## API Docs
FastAPI auto-generates docs at: http://localhost:8000/docs

## How to Use
1. Go to **🔮 Predict Flow Rate**
2. Upload 3 CSV files — one each for Axial, Horizontal, Vertical axes
3. Click **Predict Flow Rate**
4. View prediction, extracted features, and vibration health
5. Check **📊 Signal Analysis** for time-domain + FFT plots
6. Check **📈 Prediction History** to track all predictions

## File Naming Convention
Files must follow the OMNITREND format with RPM and axis in the header:
- Example: `100H1.csv` → RPM=100, Axis=Horizontal
- The app extracts RPM and axis **automatically** from the file header — no manual input needed.