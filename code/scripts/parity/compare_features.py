"""Feature parity check: our re-extracted SlowFast features vs. the published ActionFormer ones.

Usage:
  python code/scripts/parity/compare_features.py --ours features/slowfast_epic/P28_18.npy \
      --published data/af_features/P28_18.npz --variant "short256_center224" \
      --out experiments/parity_P28_18_short256_center224.json

Pass criterion (docs/FEATURES_PARITY.md): with the best temporal offset in [-2, 2] clips,
median cosine >= 0.95, 5th percentile >= 0.90, and clip counts within +/-1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MEDIAN_MIN, P5_MIN, COUNT_TOL, MAX_OFFSET = 0.95, 0.90, 1, 2


def load(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path) as d:
            key = "feats" if "feats" in d.files else d.files[0]
            x = d[key]
    else:
        x = np.load(path)
    x = np.asarray(x, np.float32)
    if x.ndim != 2:
        raise ValueError(f"{path}: expected 2-D [T, D] features, got shape {x.shape}")
    return x


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return (a * b).sum(1) / np.maximum(na * nb, 1e-12)


def compare(ours: np.ndarray, pub: np.ndarray, max_offset: int = MAX_OFFSET) -> dict:
    if ours.shape[1] != pub.shape[1]:
        return {"pass": False, "error": f"dim mismatch ours={ours.shape[1]} published={pub.shape[1]}"}
    best = None
    for off in range(-max_offset, max_offset + 1):
        # off > 0: our clip i matches published clip i + off
        a = ours[max(0, -off):]
        b = pub[max(0, off):]
        n = min(len(a), len(b))
        if n < 5:
            continue
        cs = cosine_rows(a[:n], b[:n])
        r = {"offset": off, "n_compared": int(n), "median": float(np.median(cs)),
             "p5": float(np.percentile(cs, 5)), "mean": float(cs.mean()),
             "min": float(cs.min())}
        if best is None or r["median"] > best["median"]:
            best = r
    if best is None:
        return {"pass": False, "error": "too few overlapping clips"}
    count_ok = abs(len(ours) - len(pub)) <= COUNT_TOL
    best.update(n_ours=int(len(ours)), n_published=int(len(pub)), count_ok=bool(count_ok),
                norm_ratio=float(np.linalg.norm(ours, axis=1).mean() /
                                 max(np.linalg.norm(pub, axis=1).mean(), 1e-12)))
    best["pass"] = bool(best["median"] >= MEDIAN_MIN and best["p5"] >= P5_MIN and count_ok)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--published", required=True)
    ap.add_argument("--variant", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    r = compare(load(a.ours), load(a.published))
    r.update(variant=a.variant, ours=str(a.ours), published=str(a.published))
    print(json.dumps(r, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(r, indent=2))
    return 0 if r.get("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
