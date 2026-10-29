"""Serving API: ingest vibration readings per asset, return health score, status and alarms."""
from datetime import datetime
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pdm.core import N_BASE, first_alarm, health_score, normalise
from pdm.features import FEATURE_NAMES, extract

ART = Path(__file__).resolve().parent.parent / "artifacts"
bundle = joblib.load(ART / "model.joblib")
model, THRESHOLD, PERSIST = bundle["model"], bundle["threshold"], bundle["persist"]

app = FastAPI(title="Predictive Maintenance Early-Warning API", version="1.0")
_assets: dict[str, pd.DataFrame] = {}
_lock = Lock()


class Reading(BaseModel):
    timestamp: datetime
    features: dict[str, float] | None = Field(None, description=f"Pre-computed features: {FEATURE_NAMES}")
    signal: list[float] | None = Field(None, description="Raw vibration snapshot (20 kHz); features are computed server-side")


class ReadingBatch(BaseModel):
    readings: list[Reading]


def _status(p: float, alarm: bool, calibrating: bool) -> str:
    if calibrating:
        return "CALIBRATING"
    return "ALARM" if alarm else ("WARNING" if p >= 0.5 else ("WATCH" if p >= 0.2 else "OK"))


def _score(history: pd.DataFrame) -> pd.DataFrame:
    z = normalise(history)
    p = model.predict_proba(z[FEATURE_NAMES])[:, 1]
    p[:N_BASE] = 0.0
    out = pd.DataFrame({"timestamp": history["timestamp"].values, "p_fault": p})
    out["health_score"] = [health_score(x) for x in p]
    run, alarms = 0, []
    for i, x in enumerate(p):
        run = run + 1 if (i >= N_BASE and x >= THRESHOLD) else 0
        alarms.append(run >= PERSIST)
    out["alarm"] = alarms
    out["status"] = [_status(x, a, i < N_BASE) for i, (x, a) in enumerate(zip(p, alarms))]
    return out


@app.get("/health")
def health():
    return {"ok": True, "model": bundle["name"], "baseline_readings": N_BASE, "alarm_rule": f"P(fault)>={THRESHOLD} for {PERSIST} consecutive readings"}


@app.get("/metrics")
def metrics():
    import json

    return json.loads((ART / "metrics.json").read_text())


@app.post("/assets/{asset_id}/readings")
def ingest(asset_id: str, batch: ReadingBatch):
    rows = []
    for r in batch.readings:
        if r.signal is not None:
            f = dict(zip(FEATURE_NAMES, extract(np.asarray(r.signal))))
        elif r.features is not None:
            missing = set(FEATURE_NAMES) - set(r.features)
            if missing:
                raise HTTPException(422, f"missing features: {sorted(missing)}")
            f = r.features
        else:
            raise HTTPException(422, "provide either 'signal' or 'features'")
        rows.append({"timestamp": pd.Timestamp(r.timestamp), **{k: f[k] for k in FEATURE_NAMES}})
    if not rows:
        raise HTTPException(422, "no readings")
    with _lock:
        hist = pd.concat([_assets.get(asset_id, pd.DataFrame()), pd.DataFrame(rows)], ignore_index=True)
        hist = hist.sort_values("timestamp").reset_index(drop=True)
        _assets[asset_id] = hist
        scored = _score(hist)
    new = scored.tail(len(rows))
    return {"asset_id": asset_id, "n_readings": len(hist), "latest": _record(scored.iloc[-1]), "results": [_record(r) for _, r in new.iterrows()]}


def _record(r) -> dict:
    return {"timestamp": pd.Timestamp(r["timestamp"]).isoformat(), "health_score": float(r["health_score"]), "p_fault": round(float(r["p_fault"]), 4), "status": r["status"], "alarm": bool(r["alarm"])}


@app.get("/assets")
def assets():
    with _lock:
        return {a: _record(_score(h).iloc[-1]) for a, h in _assets.items()}


@app.get("/assets/{asset_id}")
def asset(asset_id: str):
    with _lock:
        if asset_id not in _assets:
            raise HTTPException(404, "unknown asset")
        scored = _score(_assets[asset_id])
    alarm_idx = first_alarm(scored["p_fault"].to_numpy(), THRESHOLD, PERSIST)
    return {"asset_id": asset_id, "latest": _record(scored.iloc[-1]), "first_alarm": None if alarm_idx is None else scored["timestamp"].iloc[alarm_idx].isoformat(), "history": [_record(r) for _, r in scored.iterrows()]}


@app.delete("/assets/{asset_id}")
def reset(asset_id: str):
    with _lock:
        _assets.pop(asset_id, None)
    return {"deleted": asset_id}
