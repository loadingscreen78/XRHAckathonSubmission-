# Storage and compute plan for EPIC-100 features (Phase 0.5h)

**Status: estimates.** Every number marked *(est.)* is an assumption to be replaced by a
measurement: download speed and GB/hour after the first video, GPU throughput from the
teammate's `results_*.zip`. Hours of video come from `EPIC_100_video_info.csv` (measured).

## Inputs

| Quantity | Value | Source |
|---|---|---|
| EPIC-100 train video | 74.7 h (495 videos) | annotation CSVs (measured) |
| EPIC-100 validation video | 13.2 h (138 videos) | annotation CSVs (measured) |
| Longest single video | 3,708 s (~62 min) | annotation CSVs (measured) |
| Video size | ~7 GB per hour *(est.)* | ~700 GB for ~100 h total; re-measure on first download |
| Free disk on D: | 141 GB | `df` (measured) |
| SlowFast features (ActionFormer setting) | 112.5 vectors/min × 2304 × fp16 = **0.52 MB per video-minute** | arithmetic |
| Decode 1080p60 H.264, laptop CPU | 11.8 s per video-minute (synthetic, low-complexity content) | `experiments/bench_encoders_laptop_cpu.json` (measured); real footage is likely 1.5–2× slower *(est.)* |
| SlowFast-R50 GPU compute | pending | teammate GPU run / Kaggle T4 |

## Options

| Option | What we download | Peak disk | Download volume | Compute (Kaggle T4) |
|---|---|---|---|---|
| **A. Published features + parity test** (preferred) | ActionFormer `epic_kitchens.tar.gz` (~5–8 GB *(est.)*) + 2 videos for parity (~1–10 GB) | ~20 GB | ~15 GB | parity test only: < 1 h |
| A + local end-to-end eval on the 138 val videos | + val videos, streamed one at a time | ~10 GB (1 video + features) | ~92 GB *(est.)* | run.sh on 13.2 h of video: decode-bound, ~3–6 h *(est.)* |
| **B. Extract all train features ourselves** | all 495 train videos, streamed | ~10 GB | ~523 GB *(est.)* | decode ~20 s/video-min → **~25 h** + GPU ~3–5 h *(est.)* |
| **C. Extract a 30% train subset** | ~150 videos stratified by participant, kitchen and pair coverage | ~10 GB | ~157 GB *(est.)* | ~8 h decode + ~1–2 h GPU *(est.)* |

Features for all of train + val (option B) are only ~2.7 GB in fp16, so features are never the
problem. The bottlenecks are **download volume** and **1080p decoding**.

## Download → extract → delete loop (options B, C and A-eval)

Run on Kaggle or Colab, so that nothing touches the laptop disk. Planned script
`code/scripts/extract_loop.py`, written once the parity test decides the extractor:

1. Read a work list (video_id, URL) and skip IDs already in `features/<enc>/done.txt`
   (resumable across 12 h Kaggle sessions).
2. Download one video to local scratch and check its md5 against the official download
   scripts' list.
3. Extract with the **same** extractor `run.sh` uses, then write `<video_id>.npy`
   (fp16) and `meta.json` (fps, stride, timestamps, config hash).
4. Delete the video, then append to `done.txt`. Sync features to a Kaggle dataset or
   Google Drive every N videos.
5. Prefetch the next video while extracting the current one, so download and compute overlap.

Peak disk use is one video (≤ ~7.5 GB for the 62-minute one) plus the features.

## Recommendation

1. Do **A** first: 2 videos for the parity test, and they need your approval before download.
2. If parity passes, train on the published features and stream only the val videos for
   end-to-end evaluation.
3. If parity fails, do **C** first, so there's a working model early, then extend towards
   **B** as time allows.
4. GPU throughput from the teammate run will replace the *(est.)* compute numbers.
