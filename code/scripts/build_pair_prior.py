"""Build weights/epic_pair_prior.json: P(verb | noun) from the EPIC-KITCHENS-100 TRAIN split.

Only EPIC_100_train.csv is read. The validation split is the Track 4 test set and is never
touched. Annotations: CC BY-NC 4.0 (github.com/epic-kitchens/epic-kitchens-100-annotations).

Usage: python code/scripts/build_pair_prior.py [--ann-dir data/epic_annotations] [--top 5]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def class_keys(path: Path) -> dict[int, str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["id"]): r["key"] for r in csv.DictReader(f)}


def build(ann_dir: Path, top: int = 5) -> dict:
    verbs = class_keys(ann_dir / "EPIC_100_verb_classes.csv")
    nouns = class_keys(ann_dir / "EPIC_100_noun_classes.csv")
    counts: dict[str, Counter] = defaultdict(Counter)
    n_rows = 0
    with open(ann_dir / "EPIC_100_train.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            counts[nouns[int(r["noun_class"])]][verbs[int(r["verb_class"])]] += 1
            n_rows += 1
    prior = {}
    for noun, c in counts.items():
        total = sum(c.values())
        prior[noun] = {v: round(k / total, 4) for v, k in c.most_common(top)}
    return {"source": "EPIC-KITCHENS-100 train split (EPIC_100_train.csv) only",
            "train_segments": n_rows, "top_verbs_per_noun": top, "verb_given_noun": prior}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", type=Path, default=ROOT / "data" / "epic_annotations")
    ap.add_argument("--out", type=Path, default=ROOT / "weights" / "epic_pair_prior.json")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args(argv)
    out = build(a.ann_dir, a.top)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {a.out}: {len(out['verb_given_noun'])} nouns from {out['train_segments']} train segments")


if __name__ == "__main__":
    main()
