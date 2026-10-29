"""Replay a run-to-failure test as a live feed into the API.

Raw 20 kHz snapshots are posted one at a time; timestamps are shifted so the run ends "now",
and one snapshot is sent every --interval seconds (a 10 min IMS snapshot interval compressed).

  python pdm\\stream.py --set 2 --bearing 1 --interval 0.5
"""
import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent / "data" / "ims"
FOLDERS = {1: ROOT / "r1" / "1st_test", 2: ROOT / "r2" / "2nd_test", 3: ROOT / "r3" / "4th_test" / "txt"}
CH_PER_BEARING = {1: 2, 2: 1, 3: 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=int, default=2)
    ap.add_argument("--bearing", type=int, default=1)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between snapshots")
    ap.add_argument("--start", type=int, default=0, help="skip this many snapshots")
    ap.add_argument("--asset", default=None)
    a = ap.parse_args()

    files = sorted(f for f in FOLDERS[a.set].iterdir() if f.is_file())
    stamps = [datetime.strptime(f.name, "%Y.%m.%d.%H.%M.%S") for f in files]
    shift = datetime.now() - stamps[-1]
    asset = a.asset or f"live-s{a.set}b{a.bearing}"
    requests.delete(f"{a.api}/assets/{asset}", timeout=10)
    col = (a.bearing - 1) * CH_PER_BEARING[a.set]
    last = None
    for i in range(a.start, len(files)):
        sig = np.loadtxt(files[i])[:, col].tolist()
        r = requests.post(f"{a.api}/assets/{asset}/readings", json={"readings": [{"timestamp": (stamps[i] + shift).isoformat(), "signal": sig}]}, timeout=30).json()["latest"]
        print(f"{i + 1}/{len(files)} {r['timestamp']} health={r['health_score']:5.1f} {r['status']}", flush=True)
        if r["alarm"] and last != "ALARM":
            print(f"*** ALARM raised {len(files) - 1 - i} snapshots before the end of the run ({(stamps[-1] - stamps[i]).total_seconds() / 86400:.2f} days before failure)", flush=True)
        last = r["status"]
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
