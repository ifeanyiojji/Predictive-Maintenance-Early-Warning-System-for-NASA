"""Streamlit dashboard: fleet alarm table, health score trend and headline results."""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from pdm.core import FAILED, N_BASE, first_alarm
from pdm.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parent.parent
API = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("Predictive Maintenance Early-Warning System")
st.caption("NASA IMS bearing run-to-failure data, 3 test sets, 12 bearings. Scores shown are held-out (leave-one-test-set-out).")


@st.cache_data
def load():
    feats = pd.read_csv(ROOT / "data" / "features.csv", parse_dates=["timestamp"])
    oof = pd.read_csv(ROOT / "artifacts" / "oof_scores.csv", parse_dates=["timestamp"])
    return feats, oof, json.loads((ROOT / "artifacts" / "metrics.json").read_text())


feats, oof, metrics = load()
m = metrics["models"]["hist_gb"]
thr, per = metrics["threshold"], metrics["persist"]

leads = [v for v in m["lead_time_days"].values() if v is not None]
c = st.columns(5)
c[0].metric("Failures caught", f"{len(leads)}/{len(m['lead_time_days'])}")
c[1].metric("Mean warning lead time", f"{sum(leads) / len(leads):.1f} days")
c[2].metric("Range", f"{min(leads):.1f} - {max(leads):.1f} days")
c[3].metric("False alarms (healthy bearings)", f"{sum(m['false_alarm_on_healthy_bearings'].values())}/{len(m['false_alarm_on_healthy_bearings'])}")
c[4].metric("Balanced accuracy / recall", f"{m['balanced_accuracy']:.1%} / {m['recall']:.1%}")

rows = []
for (s, b), g in oof.groupby(["test_set", "bearing"]):
    g = g.sort_values("timestamp")
    i = first_alarm(g.p_oof.to_numpy(), thr, per)
    end = g.timestamp.iloc[-1]
    rows.append({"asset": f"Set {s} / Bearing {b}", "truth": "failed" if (s, b) in FAILED else "healthy",
                 "health_now": round(100 * (1 - g.p_oof.iloc[-1]), 1),
                 "first_alarm": None if i is None else g.timestamp.iloc[i],
                 "days_before_failure": None if (i is None or (s, b) not in FAILED) else round((end - g.timestamp.iloc[i]).total_seconds() / 86400, 2)})
st.subheader("Fleet alerts")
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.subheader("Asset health")
asset = st.selectbox("Asset", [r["asset"] for r in rows], index=[r["asset"] for r in rows].index("Set 2 / Bearing 1"))
s, b = int(asset.split()[1]), int(asset.split()[-1])
g = oof[(oof.test_set == s) & (oof.bearing == b)].sort_values("timestamp").reset_index(drop=True)
cursor = st.slider("Replay position (snapshot)", N_BASE + 1, len(g), len(g))
sub = g.iloc[:cursor]
health = 100 * (1 - sub.p_oof)
fig = go.Figure(go.Scatter(x=sub.timestamp, y=health, name="Health score"))
fig.add_hline(y=100 * (1 - thr), line_dash="dash", line_color="red", annotation_text="alarm level")
i = first_alarm(sub.p_oof.to_numpy(), thr, per)
if i is not None:
    fig.add_vline(x=sub.timestamp.iloc[i], line_color="orange", annotation_text="ALARM")
if (s, b) in FAILED:
    fig.add_vline(x=g.timestamp.iloc[-1], line_color="black", annotation_text="failure")
fig.update_layout(yaxis=dict(range=[0, 100], title="Health score (100 = healthy)"), height=400)
st.plotly_chart(fig, use_container_width=True)

if st.button("Score through live API"):
    f = feats[(feats.test_set == s) & (feats.bearing == b)].sort_values("timestamp").iloc[:cursor]
    key = f"s{s}b{b}"
    try:
        requests.delete(f"{API}/assets/{key}", timeout=10)
        payload = {"readings": [{"timestamp": r.timestamp.isoformat(), "features": {k: getattr(r, k) for k in FEATURE_NAMES}} for r in f.itertuples()]}
        out = requests.post(f"{API}/assets/{key}/readings", json=payload, timeout=120).json()["latest"]
        st.success(f"API status: **{out['status']}**, health score {out['health_score']}")
        st.caption("The deployed model is trained on all four failures, so this replay is in-sample; the chart and table above are held-out.")
    except Exception as e:
        st.error(f"API unreachable: {e}")


st.divider()
st.subheader("Live feed (from pdm/stream.py)")


@st.fragment(run_every=2)
def live_panel():
    try:
        assets = {k: v for k, v in requests.get(f"{API}/assets", timeout=5).json().items() if k.startswith("live-")}
    except Exception as e:
        st.error(f"API unreachable: {e}")
        return
    if not assets:
        st.info("No live asset yet. Run: python pdm\\stream.py --set 2 --bearing 1 --interval 0.5")
        return
    for key, latest in assets.items():
        h = requests.get(f"{API}/assets/{key}", timeout=10).json()
        hist = pd.DataFrame(h["history"])
        icon = {"ALARM": "🔴", "WARNING": "🟠", "WATCH": "🟡", "OK": "🟢"}.get(latest["status"], "⚪")
        st.markdown(f"**{key}** {icon} {latest['status']} | health {latest['health_score']} | {len(hist)} readings | last {latest['timestamp']}")
        if h["first_alarm"]:
            st.error(f"ALARM first raised at {h['first_alarm']}")
        fig = go.Figure(go.Scatter(x=hist.timestamp, y=hist.health_score))
        fig.add_hline(y=100 * (1 - thr), line_dash="dash", line_color="red")
        fig.update_layout(yaxis=dict(range=[0, 100], title="Health score"), height=300)
        st.plotly_chart(fig, use_container_width=True, key=f"live-{key}")


live_panel()
