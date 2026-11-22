"""One-time setup (needs internet): fill weights/ so run.sh works offline.

  1. CLIP ViT-B/16 (MIT licence)  -> weights/clip-vit-base-patch16/   (~600 MB)
  2. EPIC-100 annotations (train) -> weights/epic_pair_prior.json       (verb prior, a few KB)

Usage: python code/scripts/download_weights.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

ANN_REPO = "https://github.com/epic-kitchens/epic-kitchens-100-annotations"


def main():
    from transformers import CLIPModel, CLIPProcessor

    from camspector.clip_encoder import HUB_ID, LOCAL_DIR

    if not (LOCAL_DIR / "config.json").exists():
        print(f"downloading {HUB_ID} -> {LOCAL_DIR}")
        CLIPModel.from_pretrained(HUB_ID).save_pretrained(LOCAL_DIR)
        CLIPProcessor.from_pretrained(HUB_ID).save_pretrained(LOCAL_DIR)
    else:
        print(f"CLIP already in {LOCAL_DIR}")

    ann = ROOT / "data" / "epic_annotations"
    if not (ann / "EPIC_100_train.csv").exists():
        print(f"cloning {ANN_REPO} -> {ann}")
        subprocess.run(["git", "clone", "--depth", "1", ANN_REPO, str(ann)], check=True)
    from scripts.build_pair_prior import main as build_prior
    build_prior([])
    print("done: weights/ is ready for offline runs")


if __name__ == "__main__":
    main()
