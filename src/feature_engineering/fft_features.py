import numpy as np


def compute_fft_features(signal, sampling_rate=25600):

    signal = np.array(signal)

    N = len(signal)

    fft_vals = np.fft.rfft(signal)
    fft_vals = np.abs(fft_vals)

    freqs = np.fft.rfftfreq(N, d=1/sampling_rate)

    dominant_freq = freqs[np.argmax(fft_vals)]

    spectral_centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals)

    spectral_entropy = -np.sum(
        (fft_vals / np.sum(fft_vals)) *
        np.log(fft_vals / np.sum(fft_vals) + 1e-12)
    )

    features = {
        "dominant_freq": dominant_freq,
        "spectral_centroid": spectral_centroid,
        "spectral_entropy": spectral_entropy
    }

    return features