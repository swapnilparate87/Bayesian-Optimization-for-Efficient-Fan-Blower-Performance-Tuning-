"""
feature_extractor.py
--------------------
Parses raw vibration CSV files (OMNITREND format),
extracts features that EXACTLY match what best_model.pkl was trained on.

Model expects these 37 features:
['RPM', 'A_mean', 'A_std', 'A_rms', 'A_min', 'A_max', 'A_peak',
 'A_skewness', 'A_kurtosis', 'A_crest_factor', 'A_dominant_freq',
 'A_spectral_centroid', 'A_spectral_entropy',
 'H_mean', 'H_std', 'H_rms', 'H_min', 'H_max', 'H_peak',
 'H_skewness', 'H_kurtosis', 'H_crest_factor', 'H_dominant_freq',
 'H_spectral_centroid', 'H_spectral_entropy',
 'V_mean', 'V_std', 'V_rms', 'V_min', 'V_max', 'V_peak',
 'V_skewness', 'V_kurtosis', 'V_crest_factor', 'V_dominant_freq',
 'V_spectral_centroid', 'V_spectral_entropy']
"""

import re
import io
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
from typing import Dict


AXIS_MAP = {
    "horizontal": "H",
    "vertical":   "V",
    "axial":      "A",
}

# EXACT feature order matching best_model.pkl
FEATURE_ORDER = [
    "RPM",
    "A_mean", "A_std", "A_rms", "A_min", "A_max", "A_peak",
    "A_skewness", "A_kurtosis", "A_crest_factor",
    "A_dominant_freq", "A_spectral_centroid", "A_spectral_entropy",
    "H_mean", "H_std", "H_rms", "H_min", "H_max", "H_peak",
    "H_skewness", "H_kurtosis", "H_crest_factor",
    "H_dominant_freq", "H_spectral_centroid", "H_spectral_entropy",
    "V_mean", "V_std", "V_rms", "V_min", "V_max", "V_peak",
    "V_skewness", "V_kurtosis", "V_crest_factor",
    "V_dominant_freq", "V_spectral_centroid", "V_spectral_entropy",
]


# ──────────────────────────────────────────────
# Parse a single OMNITREND CSV file
# ──────────────────────────────────────────────
def parse_signal_file(filepath_or_bytes) -> tuple:
    """
    Returns (rpm, axis_code, signal_array)
    axis_code is one of: 'A', 'H', 'V'
    """
    if hasattr(filepath_or_bytes, "read"):
        content = filepath_or_bytes.read()
        df = pd.read_csv(io.BytesIO(content), encoding="latin1")
    else:
        df = pd.read_csv(filepath_or_bytes, encoding="latin1")

    header = df.columns[0]

    # Extract RPM from header e.g. "...100 rpm..."
    rpm_match = re.search(r"(\d+)\s*rpm", header, re.IGNORECASE)
    if not rpm_match:
        raise ValueError(f"Could not extract RPM from file header:\n{header}")
    rpm = int(rpm_match.group(1))

    # Extract axis from header e.g. "...Horizontal..."
    axis_match = re.search(r"(Horizontal|Vertical|Axial)", header, re.IGNORECASE)
    if not axis_match:
        raise ValueError(f"Could not extract axis from file header:\n{header}")
    axis_code = AXIS_MAP[axis_match.group(1).lower()]

    # Signal values start at row 7, column index 3 ("Value" column)
    signal = pd.to_numeric(df.iloc[7:, 3], errors="coerce").dropna().values

    if len(signal) == 0:
        raise ValueError(f"No signal data found in file. Check file format.")

    return rpm, axis_code, signal


# ──────────────────────────────────────────────
# Extract features — matches build_dataset.py
# ──────────────────────────────────────────────
def extract_features(signal: np.ndarray, axis_prefix: str) -> Dict[str, float]:
    """
    Extract 12 features per axis matching EXACTLY what the model was trained on:
    mean, std, rms, min, max, peak, skewness, kurtosis,
    crest_factor, dominant_freq, spectral_centroid, spectral_entropy
    """
    n   = len(signal)
    rms = float(np.sqrt(np.mean(signal ** 2)))

    # ── Time domain ──
    feats = {
        f"{axis_prefix}_mean":     float(np.mean(signal)),
        f"{axis_prefix}_std":      float(np.std(signal)),
        f"{axis_prefix}_rms":      rms,
        f"{axis_prefix}_min":      float(np.min(signal)),
        f"{axis_prefix}_max":      float(np.max(signal)),
        f"{axis_prefix}_peak":     float(np.max(np.abs(signal))),
        f"{axis_prefix}_skewness": float(stats.skew(signal)),
        f"{axis_prefix}_kurtosis": float(stats.kurtosis(signal)),
        f"{axis_prefix}_crest_factor": float(
            np.max(np.abs(signal)) / (rms + 1e-10)
        ),
    }

    # ── Frequency domain ──
    fft_vals  = np.abs(fft(signal))[: n // 2]
    freqs     = fftfreq(n, d=1.0 / n)[: n // 2]

    # Dominant frequency — freq with highest FFT magnitude
    dom_idx = int(np.argmax(fft_vals))
    feats[f"{axis_prefix}_dominant_freq"] = float(freqs[dom_idx])

    # Spectral centroid — weighted mean frequency
    total_mag = np.sum(fft_vals) + 1e-10
    feats[f"{axis_prefix}_spectral_centroid"] = float(
        np.sum(freqs * fft_vals) / total_mag
    )

    # Spectral entropy — measure of spectral complexity
    psd_norm = fft_vals / total_mag
    psd_norm = psd_norm[psd_norm > 0]  # avoid log(0)
    feats[f"{axis_prefix}_spectral_entropy"] = float(
        -np.sum(psd_norm * np.log2(psd_norm))
    )

    return feats


# ──────────────────────────────────────────────
# Main pipeline: 3 files → feature DataFrame
# ──────────────────────────────────────────────
def build_feature_row(files: list) -> pd.DataFrame:
    """
    Accept a list of 3 file paths or UploadedFile objects
    (one per axis: A, H, V — any order).
    Returns a single-row DataFrame with all 37 features in FEATURE_ORDER.
    """
    all_features: Dict[str, float] = {}
    rpm_values = []
    found_axes = []

    for f in files:
        rpm, axis_code, signal = parse_signal_file(f)
        rpm_values.append(rpm)
        found_axes.append(axis_code)
        axis_feats = extract_features(signal, axis_code)
        all_features.update(axis_feats)

    # Validate RPM consistency
    if len(set(rpm_values)) != 1:
        raise ValueError(
            f"RPM mismatch across files: {dict(zip(found_axes, rpm_values))}. "
            "All 3 files must be from the same RPM reading."
        )

    # Validate all 3 axes present
    missing_axes = set(["A", "H", "V"]) - set(found_axes)
    if missing_axes:
        axis_names = {"A": "Axial", "H": "Horizontal", "V": "Vertical"}
        raise ValueError(
            f"Missing axis files: {[axis_names[a] for a in missing_axes]}. "
            "Please upload one file per axis."
        )

    all_features["RPM"] = float(rpm_values[0])

    # Build DataFrame with exact column order matching model
    row = {col: all_features[col] for col in FEATURE_ORDER}
    return pd.DataFrame([row])


# ──────────────────────────────────────────────
# Utility: get signal + FFT for plotting
# ──────────────────────────────────────────────
def get_signal_and_fft(filepath_or_bytes):
    """Returns (rpm, axis_code, signal, fft_freqs, fft_magnitude) for plotting."""
    rpm, axis_code, signal = parse_signal_file(filepath_or_bytes)
    n        = len(signal)
    fft_vals = np.abs(fft(signal))[: n // 2]
    freqs    = fftfreq(n, d=1.0 / n)[: n // 2]
    return rpm, axis_code, signal, freqs, fft_vals