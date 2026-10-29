# Predictive Maintenance Early-Warning System

Bearing fault early warning on the public **NASA IMS run-to-failure** vibration data (3 test sets, 12 bearings, 4 failures).

```
pdm/features.py        vibration features (RMS, kurtosis, crest, 9 FFT bands, envelope kurtosis...)
pdm/core.py            baseline normalisation, labelling, alarm rule
pdm/build_features.py  raw snapshots -> data/features.csv
pdm/train.py           leave-one-test-set-out evaluation + final model -> artifacts/
pdm/api.py             FastAPI service
pdm/dashboard.py       Streamlit dashboard
```

## Run

```
.venv\Scripts\activate
set PYTHONPATH=.
python pdm\build_features.py                    # needs data\ims\r1,r2,r3 (extracted from the NASA IMS archive)
python pdm\train.py 1.0 save hist_gb
uvicorn pdm.api:app --port 8000                 # docs at /docs
streamlit run pdm\dashboard.py                  # http://localhost:8501
```

API: `POST /assets/{id}/readings` (raw `signal` or `features`) -> health score (0-100), status
(CALIBRATING/OK/WATCH/WARNING/ALARM), alarm flag. Also `GET /assets`, `GET /assets/{id}`, `GET /metrics`, `DELETE /assets/{id}`.
The first 50 readings of an asset (~8 h) are its healthy baseline.

## Method
- Label: snapshot is "fault" if within 1 day of the end of a run-to-failure test; healthy bearings are all 0.
- Features are relative to each bearing's own baseline, smoothed with a 1 h rolling median.
- Validation: **leave-one-test-set-out** (train on 2 test sets, test on the third), so no bearing or run is seen in training.
- Alarm: P(fault) >= 0.9 for 12 consecutive readings (~2 h). Lead time = failure time - first alarm.

## Results (held-out)
| Metric | Value |
|---|---|
| Failures detected | 4 / 4 |
| False alarms on 8 healthy bearings | 0 / 8 |
| Warning lead time | 1.88, 4.02, 0.70, 1.08 days (mean 1.9) |
| Snapshot balanced accuracy / recall / precision | 86.7% / 75.0% / 33.1% |
| Snapshot accuracy | 98.3% (majority-class baseline: 99.0%) |

## Live API check (dashboard)
The dashboard's **Score through live API** button replays the selected bearing's snapshots to the running API
(`POST /assets/{id}/readings`) and shows the returned status. Verified end to end on Set 2 / Bearing 1
(all 984 snapshots): API returned **ALARM, health score 0.1**. This is in-sample (the deployed model was trained on that failure).
Start the API before the dashboard (the sidebar "API URL" defaults to `http://127.0.0.1:8000`).

## Live feed simulation
`python pdm\stream.py --set 2 --bearing 1 --interval 0.5` replays raw 20 kHz snapshots one at a time into the API,
with timestamps shifted so the run ends "now". The dashboard's **Live feed** panel polls the API every 2 s and shows
status, health trend and the first alarm time. Tested on Set 2 / Bearing 1: CALIBRATING for the first 50 readings,
OK, then ALARM raised 0.92 days before failure (in-sample deployed model; held-out lead time was 0.70 days).
This is a simulation of recent data, not a real sensor connection.

## Caveats
- Raw accuracy is below the majority baseline because only ~1% of snapshots are in the final day; use balanced accuracy/recall and alarm-level results.
- Threshold and persistence were chosen after looking at the held-out scores, so the 0/8 false-alarm result is optimistic. With only 4 failures, lead times are indicative, not statistically tight.
- Lead time is bounded by the 1-day label horizon plus how early the signal drifts; no bearing was warned more than ~4 days ahead.
- The IMS data has no true health labels for healthy bearings beyond "did not fail"; IMS test 1 bearings 3/4 had early drift that a 1-day label does not capture.
- The deployed model is trained on all 4 failures; the dashboard's fleet table and chart use the held-out scores.
