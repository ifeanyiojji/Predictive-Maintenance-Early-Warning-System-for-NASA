"""Extract per-bearing features from the raw NASA IMS snapshots into data/features.parquet-like CSV."""
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pdm.features import FEATURE_NAMES, extract

ROOT = Path(__file__).resolve().parent.parent / "data" / "ims"
# test set -> (folder, channels per bearing)
SETS = {1: (ROOT / "r1" / "1st_test", 2), 2: (ROOT / "r2" / "2nd_test", 1), 3: (ROOT / "r3" / "4th_test" / "txt", 1)}


def process(path_ch):
    path, ch_per_bearing = path_ch
    data = np.loadtxt(path)
    ts = datetime.strptime(path.name, "%Y.%m.%d.%H.%M.%S")
    # first channel of each bearing keeps all test sets comparable
    return ts, [extract(data[:, b * ch_per_bearing]) for b in range(4)]


def main():
    rows = []
    with Pool() as pool:
        for s, (folder, cpb) in SETS.items():
            if not folder.exists():
                alt = [p for p in (ROOT / f"r{s}").rglob("*") if p.is_dir() and any(c.is_file() for c in p.iterdir())]
                folder = alt[0]
            files = sorted(f for f in folder.iterdir() if f.is_file())
            print(f"set {s}: {len(files)} files in {folder}", flush=True)
            for ts, feats in pool.imap(process, [(f, cpb) for f in files], chunksize=8):
                for b, f in enumerate(feats):
                    rows.append([s, b + 1, ts, *f])
    df = pd.DataFrame(rows, columns=["test_set", "bearing", "timestamp", *FEATURE_NAMES])
    df.to_csv(ROOT.parent / "features.csv", index=False)
    print(df.shape)


if __name__ == "__main__":
    main()
