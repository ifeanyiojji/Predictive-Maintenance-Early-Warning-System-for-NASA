"""Feature normalisation, labelling and alarm logic shared by training and serving."""
import numpy as np
import pandas as pd

from pdm.features import FEATURE_NAMES

N_BASE = 50  # first snapshots of an asset define its healthy baseline (~8h at 10 min spacing)
SMOOTH = 6  # rolling-median window (~1h)
LOG_FEATS = ["rms", "peak", "crest", "p2p_over_rms", "spec_centroid"]
# (test_set, bearing) -> failed at end of run. Everything else was still healthy when the test stopped.
FAILED = {(1, 3), (1, 4), (2, 1), (3, 3)}


def normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """raw: time-ordered feature rows of ONE bearing. Returns baseline-relative, smoothed features."""
    x = raw[FEATURE_NAMES].astype(float).copy()
    x[LOG_FEATS] = np.log(x[LOG_FEATS].clip(lower=1e-9))
    base = x.iloc[:N_BASE].median()
    return (x - base).rolling(SMOOTH, min_periods=1).median()


def label_fault(ts: pd.Series, failed: bool, horizon_days: float) -> np.ndarray:
    """1 when the bearing is within `horizon_days` of the end of a run-to-failure test."""
    if not failed:
        return np.zeros(len(ts), dtype=int)
    ttf = (ts.iloc[-1] - ts).dt.total_seconds() / 86400
    return (ttf <= horizon_days).astype(int).to_numpy()


def first_alarm(prob: np.ndarray, threshold: float, persist: int) -> int | None:
    """Index where `persist` consecutive scores exceed threshold (after the baseline period)."""
    run = 0
    for i, p in enumerate(prob):
        if i < N_BASE:
            continue
        run = run + 1 if p >= threshold else 0
        if run >= persist:
            return i - persist + 1
    return None


def health_score(p_fault: float) -> float:
    return float(round(100 * (1 - p_fault), 1))
