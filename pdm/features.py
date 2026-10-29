"""Vibration feature extraction shared by training and the serving API."""
import numpy as np
from scipy.stats import kurtosis, skew

FS = 20_000  # IMS sampling rate (Hz)
BAND_EDGES = np.array([0, 500, 1000, 2000, 3000, 4000, 5000, 6500, 8000, 10000])
FEATURE_NAMES = (
    ["rms", "peak", "crest", "kurtosis", "skew", "p2p_over_rms"]
    + [f"band_{int(a)}_{int(b)}" for a, b in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]
    + ["spec_centroid", "env_kurtosis"]
)


def extract(sig: np.ndarray, fs: int = FS) -> np.ndarray:
    """Time and frequency domain features for one vibration snapshot."""
    x = np.asarray(sig, dtype=float)
    x = x - x.mean()
    rms = np.sqrt(np.mean(x**2)) + 1e-12
    peak = np.max(np.abs(x))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / fs)
    bands = [spec[(freqs >= a) & (freqs < b)].sum() for a, b in zip(BAND_EDGES[:-1], BAND_EDGES[1:])]
    total = spec.sum() + 1e-12
    centroid = (freqs * spec).sum() / total
    # envelope of the high-frequency resonance band, where bearing impacts show up
    mask = (freqs >= 2000) & (freqs < 9000)
    env = np.abs(np.fft.irfft(np.where(mask, np.fft.rfft(x), 0), n=len(x)))
    return np.array(
        [rms, peak, peak / rms, kurtosis(x), skew(x), (x.max() - x.min()) / rms]
        + [np.log10(b + 1e-12) for b in bands]
        + [centroid, kurtosis(env)]
    )
