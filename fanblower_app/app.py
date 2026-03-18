"""
app.py  —  Streamlit frontend
------------------------------
Pages:
  1. Predict         — upload 3 files → prediction + feature summary
  2. Signal Analysis — time-domain + FFT plots per axis
  3. Prediction History — table + trend chart of past predictions
"""

import io
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

from feature_extractor import build_feature_row, get_signal_and_fft

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Fan Blower Optimizer",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/fan.png", width=80)
    st.title("Fan Blower\nOptimizer")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🔮 Predict Flow Rate", "📊 Signal Analysis", "📈 Prediction History"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # API health check
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        if resp.status_code == 200:
            st.success("✅ API Online")
        else:
            st.error("❌ API Error")
    except Exception:
        st.warning("⚠️ API Offline\n(Using local mode)")

    st.markdown("---")
    st.caption("Upload 3 vibration CSV files\n(Axial · Horizontal · Vertical)")


# ══════════════════════════════════════════════
# PAGE 1 — PREDICT FLOW RATE
# ══════════════════════════════════════════════
if page == "🔮 Predict Flow Rate":

    st.title("🌀 Fan Blower Flow Rate Prediction")
    st.markdown("Upload **3 vibration signal files** (one per axis). The app extracts features automatically and predicts the flow rate.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📁 Axial (A)")
        axial_file = st.file_uploader("Upload Axial CSV", type="csv", key="axial")

    with col2:
        st.markdown("### 📁 Horizontal (H)")
        horiz_file = st.file_uploader("Upload Horizontal CSV", type="csv", key="horiz")

    with col3:
        st.markdown("### 📁 Vertical (V)")
        vert_file = st.file_uploader("Upload Vertical CSV", type="csv", key="vert")

    st.markdown("---")

    files_ready = all([axial_file, horiz_file, vert_file])

    if not files_ready:
        missing = []
        if not axial_file: missing.append("Axial")
        if not horiz_file: missing.append("Horizontal")
        if not vert_file:  missing.append("Vertical")
        st.info(f"⏳ Waiting for: **{', '.join(missing)}** file(s)")

    if files_ready:
        st.success("✅ All 3 files uploaded! Ready to predict.")

        if st.button("🚀 Predict Flow Rate", type="primary", use_container_width=True):
            with st.spinner("Extracting features and predicting..."):

                uploaded_files = [axial_file, horiz_file, vert_file]

                # ── Try API first, fall back to local ──
                try:
                    api_files = []
                    for f in uploaded_files:
                        f.seek(0)
                        api_files.append(("files", (f.name, f.read(), "text/csv")))

                    response = requests.post(f"{API_URL}/predict", files=api_files, timeout=30)

                    if response.status_code == 200:
                        result = response.json()
                        predicted_flow  = result["predicted_flow_rate"]
                        rpm             = result["rpm"]
                        features_dict   = result["features"]
                        source          = "API"
                    else:
                        st.warning(f"API error {response.status_code}: {response.text}. Using local mode.")
                        raise Exception("API failed")

                except Exception:
                    # Local fallback
                    for f in uploaded_files:
                        f.seek(0)
                    import joblib, os
                    model_path = os.environ.get("MODEL_PATH", "models/best_model.pkl")
                    model      = joblib.load(model_path)
                    for f in uploaded_files:
                        f.seek(0)
                    feature_df      = build_feature_row(uploaded_files)
                    predicted_flow  = float(model.predict(feature_df)[0])
                    rpm             = int(feature_df["RPM"].iloc[0])
                    features_dict   = feature_df.iloc[0].to_dict()
                    source          = "Local"

                # ── Save to history ──
                st.session_state.history.append({
                    "Timestamp":          datetime.now().strftime("%H:%M:%S"),
                    "RPM":                rpm,
                    "Predicted Flow Rate": round(predicted_flow, 4),
                    "Source":             source,
                })

            # ── Results ──
            st.markdown("## 🎯 Prediction Result")

            m1, m2, m3 = st.columns(3)
            m1.metric("🌊 Predicted Flow Rate", f"{predicted_flow:.4f} m³/s")
            m2.metric("⚙️ RPM",                 f"{rpm}")
            m3.metric("🔧 Mode",                source)

            st.markdown("---")

            # ── Feature summary ──
            st.markdown("### 📊 Extracted Features")

            axes = ["A", "H", "V"]
            tabs = st.tabs(["Axial (A)", "Horizontal (H)", "Vertical (V)", "All Features"])

            for i, axis in enumerate(axes):
                with tabs[i]:
                    axis_feats = {
                        k.replace(f"{axis}_", ""): round(v, 5)
                        for k, v in features_dict.items()
                        if k.startswith(f"{axis}_")
                    }
                    feat_df = pd.DataFrame(
                        list(axis_feats.items()), columns=["Feature", "Value"]
                    )
                    fig = px.bar(
                        feat_df, x="Feature", y="Value",
                        title=f"{axis} Axis Features",
                        color="Value", color_continuous_scale="Blues",
                    )
                    fig.update_layout(height=350, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            with tabs[3]:
                all_feats = pd.DataFrame(
                    list(features_dict.items()), columns=["Feature", "Value"]
                )
                all_feats["Value"] = all_feats["Value"].round(5)
                st.dataframe(all_feats, use_container_width=True, height=400)

            # ── Vibration health indicator ──
            st.markdown("---")
            st.markdown("### 🔔 Vibration Health Summary")

            h1, h2, h3 = st.columns(3)
            for col, axis in zip([h1, h2, h3], ["A", "H", "V"]):
                rms_val = features_dict.get(f"{axis}_rms", 0)
                kurt    = features_dict.get(f"{axis}_kurtosis", 0)
                if rms_val < 0.2:
                    status, color = "✅ Normal", "green"
                elif rms_val < 0.4:
                    status, color = "⚠️ Moderate", "orange"
                else:
                    status, color = "🔴 High", "red"
                axis_name = {"A": "Axial", "H": "Horizontal", "V": "Vertical"}[axis]
                col.markdown(
                    f"**{axis_name}**\n\n"
                    f"RMS: `{rms_val:.4f}` — :{color}[{status}]\n\n"
                    f"Kurtosis: `{kurt:.3f}`"
                )


# ══════════════════════════════════════════════
# PAGE 2 — SIGNAL ANALYSIS
# ══════════════════════════════════════════════
elif page == "📊 Signal Analysis":

    st.title("📊 Vibration Signal Analysis")
    st.markdown("Upload any vibration CSV file to visualize its **time-domain signal** and **FFT spectrum**.")
    st.markdown("---")

    uploaded = st.file_uploader("Upload a vibration CSV file", type="csv", key="analysis")

    if uploaded:
        with st.spinner("Processing signal..."):
            rpm, axis_code, signal, freqs, fft_vals = get_signal_and_fft(uploaded)

        axis_names = {"A": "Axial", "H": "Horizontal", "V": "Vertical"}
        axis_name  = axis_names.get(axis_code, axis_code)

        st.markdown(f"### File: `{uploaded.name}`")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("RPM",    rpm)
        c2.metric("Axis",   axis_name)
        c3.metric("Samples", len(signal))
        c4.metric("RMS",    f"{np.sqrt(np.mean(signal**2)):.4f}")

        st.markdown("---")

        # ── Time domain plot ──
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Time-Domain Signal (m/s²)", "FFT Spectrum"),
            vertical_spacing=0.15,
        )

        # Downsample for performance
        ds = max(1, len(signal) // 2000)

        fig.add_trace(
            go.Scatter(
                y=signal[::ds],
                mode="lines",
                name="Signal",
                line=dict(color="#00b4d8", width=0.8),
            ),
            row=1, col=1,
        )

        # FFT — show top 500 frequency bins only
        fig.add_trace(
            go.Scatter(
                x=freqs[:500],
                y=fft_vals[:500],
                mode="lines",
                name="FFT Magnitude",
                line=dict(color="#f77f00", width=1),
                fill="tozeroy",
                fillcolor="rgba(247,127,0,0.15)",
            ),
            row=2, col=1,
        )

        fig.update_xaxes(title_text="Sample Index",  row=1, col=1)
        fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude (m/s²)", row=1, col=1)
        fig.update_yaxes(title_text="Magnitude",         row=2, col=1)
        fig.update_layout(height=600, template="plotly_dark", showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

        # ── Dominant frequency ──
        dom_idx  = int(np.argmax(fft_vals[:500]))
        dom_freq = freqs[dom_idx]
        st.info(f"🎯 **Dominant Frequency:** `{dom_freq:.2f} Hz`  |  "
                f"**Peak FFT Magnitude:** `{fft_vals[dom_idx]:.2f}`")


# ══════════════════════════════════════════════
# PAGE 3 — PREDICTION HISTORY
# ══════════════════════════════════════════════
elif page == "📈 Prediction History":

    st.title("📈 Prediction History")
    st.markdown("Track all predictions made in this session.")
    st.markdown("---")

    if not st.session_state.history:
        st.info("No predictions yet. Go to **🔮 Predict Flow Rate** to get started.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)

        # ── Summary metrics ──
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Predictions", len(hist_df))
        m2.metric("Avg Flow Rate",     f"{hist_df['Predicted Flow Rate'].mean():.4f}")
        m3.metric("Max Flow Rate",     f"{hist_df['Predicted Flow Rate'].max():.4f}")
        m4.metric("Min Flow Rate",     f"{hist_df['Predicted Flow Rate'].min():.4f}")

        st.markdown("---")

        # ── Trend chart ──
        fig = px.line(
            hist_df,
            x=hist_df.index,
            y="Predicted Flow Rate",
            markers=True,
            title="Flow Rate Predictions Over Time",
            labels={"index": "Prediction #"},
            color_discrete_sequence=["#00b4d8"],
        )
        fig.update_layout(height=350, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # ── RPM vs Flow Rate scatter ──
        if len(hist_df) > 1:
            fig2 = px.scatter(
                hist_df,
                x="RPM",
                y="Predicted Flow Rate",
                title="RPM vs Predicted Flow Rate",
                color="Predicted Flow Rate",
                color_continuous_scale="Blues",
                size_max=15,
            )
            fig2.update_layout(height=350, template="plotly_dark")
            st.plotly_chart(fig2, use_container_width=True)

        # ── Table ──
        st.markdown("### 📋 All Predictions")
        st.dataframe(hist_df, use_container_width=True)

        # ── Download ──
        csv = hist_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download History as CSV",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv",
        )

        if st.button("🗑️ Clear History", type="secondary"):
            st.session_state.history = []
            st.rerun()