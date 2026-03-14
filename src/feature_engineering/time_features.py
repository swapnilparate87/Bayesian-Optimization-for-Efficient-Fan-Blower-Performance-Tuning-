import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


def compute_time_features(signal):

    signal = np.array(signal)

    mean = np.mean(signal)
    std = np.std(signal)
    rms = np.sqrt(np.mean(signal**2))
    minimum = np.min(signal)
    maximum = np.max(signal)
    peak = np.max(np.abs(signal))

    skewness = skew(signal)
    kurt = kurtosis(signal)

    crest_factor = peak / rms if rms != 0 else 0

    features = {
        "mean": mean,
        "std": std,
        "rms": rms,
        "min": minimum,
        "max": maximum,
        "peak": peak,
        "skewness": skewness,
        "kurtosis": kurt,
        "crest_factor": crest_factor
    }

    return features