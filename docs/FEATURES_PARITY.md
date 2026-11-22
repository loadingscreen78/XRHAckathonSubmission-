# Precomputed EPIC-100 features: can we train on them? (Phase 0.5g, research only)

Nothing was downloaded for this note. Sources: ActionFormer README
(github.com/happyharrycn/actionformer_release, section "To Reproduce Our Results on EPIC
Kitchens 100"), epic-kitchens-slowfast README (github.com/epic-kitchens/epic-kitchens-slowfast),
ActionFormer paper (arXiv 2202.07925) and the ActionFormer Ego4D report (arXiv 2211.09074).

## What is published

| Item | Status | Detail |
|---|---|---|
| Features | Public | `epic_kitchens.tar.gz` (md5 `add9803756afd9a023bc9a9c547e0229`), via Box / Google Drive / Baidu. Contains SlowFast features + annotations in ActivityNet-style JSON. |
| Extractor checkpoint | Public | SlowFast-R50 8x8 trained on EPIC-100 train (action classification), from the epic-kitchens-slowfast README: `SlowFast.pyth` on Dropbox. Config `configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml` in that repo. |
| Extractor code | **Not published** as a script for these exact features. ActionFormer points to the model only. |
| Documented settings | Partial | 32-frame clips, **30 fps**, stride **16 frames** → one feature per 16/30 ≈ 0.533 s. Feature dim 2304 (slow 2048 + fast 256, global-avg-pooled before the head). |
| Not documented | — | Input resolution / crop (likely short side 256 → centre crop 224 for test, per PySlowFast defaults), whether frames came from the official 456×256 JPEG `rgb_frames` or from re-decoded MP4s, how 59.94 fps source is resampled to 30 fps, padding at video ends, pooling details, fp32 vs fp16. |
| ActionFormer weights | Public | Verb and noun ActionFormer checkpoints on Google Drive (expected val avg mAP ≈ 23.4 verb / 21.9 noun). Inference needs >10 GB VRAM per the README; training ~4.5 GB. |

Licence: EPIC-KITCHENS data and derived models are CC BY-NC 4.0. Fine for a non-commercial
hackathon, but confirm with the organisers that CC BY-NC weights are allowed in the submission.

## Why the parity test is needed

Rule: we must not train on features that `run.sh` cannot reproduce on unseen videos. If we
train ActionFormer on the published features, our inference extractor must output the same
features. The checkpoint is public, but the preprocessing is under-documented, so we must
prove equivalence empirically.

## Proposed feature parity test

1. Download 2 EPIC-100 validation videos **on the teammate machine** (approved; the main laptop
   has no space): `P28_18` (120 s, 59.94 fps) and `P09_08` (126 s, 29.97 fps), both 1080p.
   Two frame rates exercise the resampling to 30 fps. About 0.5 GB in total *(est.)*. Script:
   `code/scripts/parity/download_parity_data.py`; instructions: `docs/dev/AGENT_TASK.md`.
2. Implement `features/slowfast_epic.py`: load `SlowFast.pyth` (PySlowFast key names mapped to
   a pytorchvideo `create_slowfast` R50 8x8, or vendored PySlowFast model code), decode with
   our `video_io`, resample to 30 fps by timestamp, clip 32 frames with stride 16, slow
   pathway = every 4th frame (alpha=4), output the 2304-d pooled vector.
3. Sweep the undocumented choices (resize 256 short side + centre crop 224 vs. full-frame
   resize, MP4 decode vs. 456×256 JPEG-equivalent downscale, start offset 0 vs. 8 frames,
   end padding), and pick the one that matches best.
4. **Pass criterion:** per-clip cosine similarity to the published features, median ≥ 0.95
   and 5th percentile ≥ 0.90, with the same number of clips (±1) per video. Also: an
   ActionFormer trained on published features, evaluated on our re-extracted features for
   those 2 videos, should give per-video verb/noun mAP within ~1 point of its score on the
   published features.
5. If it fails, fall back to extracting all features ourselves (see STORAGE_PLAN.md).

## Runtime risk to check before committing

SlowFast-R50 at the ActionFormer setting is ~112.5 clips per video-minute plus decoding
1080p@60 video. Measured on the laptop (see experiments/bench_encoders.json); must also be
measured on a T4 against MAX_RUNTIME_PER_VIDEO once that limit is known.
