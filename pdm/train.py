"""Leave-one-test-set-out evaluation, then train the final model and save artifacts."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pdm.core import FAILED, N_BASE, first_alarm, label_fault, normalise
from pdm.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent
HORIZON_DAYS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
THRESHOLD, PERSIST = 0.9, 12  # alarm = P(fault) >= 0.9 for 12 consecutive snapshots (~2h)

MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "random_forest": lambda: RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=0),
    "hist_gb": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=50, class_weight="balanced", random_state=0),
}


def build(df):
    frames = []
    for (s, b), g in df.groupby(["test_set", "bearing"]):
        g = g.sort_values("timestamp").reset_index(drop=True)
        z = normalise(g)
        z["test_set"], z["bearing"], z["timestamp"] = s, b, g["timestamp"]
        z["y"] = label_fault(g["timestamp"], (s, b) in FAILED, HORIZON_DAYS)
        z["failed"] = (s, b) in FAILED
        z["idx"] = np.arange(len(g))
        frames.append(z)
    return pd.concat(frames, ignore_index=True)


def main():
    df = pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["timestamp"])
    d = build(df)
    d = d[d.idx >= N_BASE].reset_index(drop=True) if False else d
    train_mask_all = d.idx >= N_BASE  # baseline window is not scored
    results, oof = {}, []
    for name, mk in MODELS.items():
        scores = pd.Series(np.nan, index=d.index)
        for held in (1, 2, 3):
            tr = (d.test_set != held) & train_mask_all
            te = d.test_set == held
            m = mk().fit(d.loc[tr, FEATURE_NAMES], d.loc[tr, "y"])
            scores[te] = m.predict_proba(d.loc[te, FEATURE_NAMES])[:, 1]
        d[f"p_{name}"] = scores
        ev = d[train_mask_all]
        pred = (ev[f"p_{name}"] >= THRESHOLD).astype(int)
        tn, fp, fn, tp = confusion_matrix(ev.y, pred).ravel()
        r = dict(
            accuracy=accuracy_score(ev.y, pred), balanced_accuracy=balanced_accuracy_score(ev.y, pred),
            precision=precision_score(ev.y, pred), recall=recall_score(ev.y, pred), f1=f1_score(ev.y, pred),
            pr_auc=average_precision_score(ev.y, ev[f"p_{name}"]), confusion=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)),
        )
        leads, false_alarms = {}, {}
        for (s, b), g in d.groupby(["test_set", "bearing"]):
            g = g.sort_values("timestamp")
            i = first_alarm(g[f"p_{name}"].to_numpy(), THRESHOLD, PERSIST)
            if (s, b) in FAILED:
                leads[f"set{s}_bearing{b}"] = None if i is None else round((g.timestamp.iloc[-1] - g.timestamp.iloc[i]).total_seconds() / 86400, 2)
            else:
                false_alarms[f"set{s}_bearing{b}"] = i is not None
        r["lead_time_days"] = leads
        r["false_alarm_on_healthy_bearings"] = false_alarms
        results[name] = r
    majority = max(d[train_mask_all].y.mean(), 1 - d[train_mask_all].y.mean())
    out = dict(horizon_days=HORIZON_DAYS, majority_baseline_accuracy=majority, threshold=THRESHOLD, persist=PERSIST, models=results)
    print(json.dumps(out, indent=1, default=float))
    if len(sys.argv) > 2 and sys.argv[2] == "save":
        best = sys.argv[3]
        (ROOT / "artifacts").mkdir(exist_ok=True)
        (ROOT / "artifacts" / "metrics.json").write_text(json.dumps(out, indent=1, default=float))
        m = MODELS[best]().fit(d.loc[train_mask_all, FEATURE_NAMES], d.loc[train_mask_all, "y"])
        joblib.dump(dict(model=m, features=FEATURE_NAMES, horizon_days=HORIZON_DAYS, threshold=THRESHOLD, persist=PERSIST, name=best), ROOT / "artifacts" / "model.joblib")
        d["p_oof"] = d[f"p_{best}"]
        keep = d[["test_set", "bearing", "timestamp", "idx", "y", "failed", "p_oof"]]
        keep.to_csv(ROOT / "artifacts" / "oof_scores.csv", index=False)


if __name__ == "__main__":
    main()

