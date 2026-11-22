# CAMSPECTOR — Team SpectraCore

**Samsung PRISM XR Hackathon 2026 · Track 4 — Action-Aware Video Summarization**
("Meaningful Highlights from Egocentric Video")

| | |
|---|---|
| Team name / `team_id` | **SpectraCore** |
| Members | Jagannath Samrtha Mishra · S Hanish Kumar · Dasarigalla Raviteja · Bhumi Awasthi |
| Institution | SRM Institute of Science and Technology |
| Entry point | `./run.sh <video.mp4> <out.json>` |

CAMSPECTOR takes one first-person MP4 and writes one JSON with the video-level label,
timestamped verb–noun actions and highlight segments (≤ 15% of the original footage, each
≥ 4.0 s), in the exact format of the student kit (`TEAM_README.md`, section 3).

## Current status (22 Nov 2026)

| Part | State |
|---|---|
| Output contract (`code/camspector/contract.py`) | **Done.** Matches the kit's 6-field schema and the organisers' 20 Nov clarifications. |
| `validate.py` (repo root) | **Done.** Written by the team, as the organisers asked. |
| `run.sh` and robust decoding | **Done.** PTS-based decoding (VFR, rotation, corrupt or truncated files). Always writes a JSON. |
| Model path (`code/camspector/zeroshot.py`) | **Done, training-free.** CLIP ViT-B/16 frame embeddings → zero-shot video label, interaction-aware + diversity-aware highlight selection, and noun zero-shot × EPIC-train verb prior for actions. Not yet scored on real videos. |
| Fallback baseline | Used only if the model path fails: evenly spaced 4.5 s windows (14.5% of the video), no actions, label `"other"`. |
| SlowFast EPIC-100 feature extractor | Implemented; parity with the published features **not reached yet** (median cosine 0.26 vs ≥ 0.95). Not used by `run.sh`. See `docs/PARITY_RESULTS.md`. |
| Hardware check | RTX 4060 Laptop GPU, 8 GB, CUDA OK, fp16 OK (`experiments/gpu_check.json`). |

## Setup

```bash
# Python 3.11 (3.12 also works)
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126   # or .../whl/cpu
pip install -r requirements.txt
python code/scripts/download_weights.py   # once, needs internet: CLIP -> weights/, EPIC-train verb prior
```

After this, `weights/` holds everything `run.sh` needs, so it runs offline
(`weights/clip-vit-base-patch16/`, `weights/epic_pair_prior.json`).

## Weights and demo

- **Weights:** run `python code/scripts/download_weights.py`, which downloads CLIP ViT-B/16 from
  Hugging Face and builds the verb prior. For offline use, download
  [`SpectraCore_upload.zip`](https://github.com/loadingscreen78/XRHAckathonSubmission-/releases/download/v1.0/SpectraCore_upload.zip)
  from the [v1.0 Release](https://github.com/loadingscreen78/XRHAckathonSubmission-/releases/tag/v1.0) instead.
- **Demo video:** [`CAMSPECTOR_demo.mp4`](https://github.com/loadingscreen78/XRHAckathonSubmission-/releases/download/v1.0/CAMSPECTOR_demo.mp4)
  (v1.0 Release).

## Run

```bash
./run.sh <video.mp4> <out.json>                       # one video; logs go to stderr
code/scripts/run_all.sh <video_root> outputs/         # every .mp4 under <video_root>, recursively
python validate.py --dir outputs/                     # checks every file + manifest coverage
python validate.py --file outputs/P02_109.json -v     # one file, with warnings
```

`CAMSPECTOR_PYTHON` chooses the interpreter (default `python`).

**No big disk or local GPU?** `code/scripts/stream_run.py` downloads one EPIC video, processes it and
deletes it before the next (peak disk: one video + ~600 MB of weights), and resumes after a
disconnect. `notebooks/SpectraCore_Colab_run.ipynb` runs the whole manifest on a free Google
Colab GPU with results saved to Google Drive.

## Data (self-download, not redistributed)

- **EPIC-KITCHENS-100 validation** — 138 videos (~184 GB) with the official downloader:
  `python epic_downloader.py --videos --specific-videos "<ids from manifest.csv>" --output-path ../EPIC-KITCHENS`
- **FPVSum** — 49 manifest IDs via `yt-dlp`. `Scuba_KD74Ywuy9ng` and `Scuba_Um4kqB8bQBg` are no
  longer on YouTube; the organisers confirmed they are **excluded from scoring**, so
  **185 videos** are scored.

`code/configs/manifest.csv` and `code/configs/supported_classes.json` are the kit files.

## Model architecture (CAMSPECTOR)

**Implemented now (training-free, `code/camspector/zeroshot.py`):**

1. **Perceive**: PTS-based decoding at 2 fps (1 fps above 10 min), short side 224 px;
   CLIP ViT-B/16 image embeddings (`code/camspector/clip_encoder.py`).
2. **Label**: zero-shot against descriptive prompts for the 15 kit labels
   (e.g. "skiing down a snowy mountain slope"), averaged over three templates. No FPVSum
   annotation and no file name is used.
3. **Highlights**: 4.5 s candidate windows every 1 s, chosen by Maximal Marginal Relevance:
   relevance (hand–object interaction prompts minus idle prompts on cooking videos; the predicted
   activity on FPVSum) balanced against similarity to the windows already chosen. Why:
   coverage counts *distinct* verb–noun pairs with ≥ 50% of an action inside the highlights,
   and EPIC actions are short (median 2 s) and repeat (~2× per video; `docs/EPIC_STATS.md`), so many
   short, visually different windows reach more distinct pairs than a few long ones.
4. **Actions**: 2 s windows every 1 s; top-3 nouns by zero-shot (all 300 kit nouns) × top-2 verbs per
   noun from P(verb | noun) counted on the EPIC-100 **train** split (`code/scripts/build_pair_prior.py`;
   the validation split, which is the test set, is never read). Consecutive windows with the same
   pair merge into one segment; top 2000 by score.
5. **Constraints**: `contract.sanitize_highlights` enforces sorted, non-overlapping segments,
   each ≥ 4.0 s, total ≤ 15%.

**Next (planned):** fine-tuned temporal action head (ActionFormer-style) on EPIC-100 train features,
a hand–object interaction branch and a temporal sequence model, each kept only if it improves the
official metrics (ablation ladder in the presentation).

Any failure in the model path falls back to the baseline, so every video always gets a valid JSON.
Set `CAMSPECTOR_BASELINE=1` to force the baseline.

## Output format

```json
{
 "team_id": "SpectraCore",
 "video_id": "P02_109",
 "duration_sec": 512.3,
 "label": {"name": "cooking", "score": 0.96},
 "actions": [{"segment": [95.2, 141.6], "verb": "cut", "noun": "onion", "score": 0.88}],
 "highlights": [{"segment": [96.0, 128.0]}, {"segment": [303.0, 317.0]}]
}
```

## Organiser clarifications applied ("Updated Track 04" email, 20 Nov 2026)

- `validate.py` was not shipped → the team wrote its own (this repo).
- **Action coverage:** an action is captured if ≥ 50% of its duration lies inside the union of
  the highlights; labels do not need to match. Highlight selection should therefore cover as many
  distinct actions as possible.
- Two FPVSum videos excluded (above).
- Labels must come from `supported_classes.json` (e.g. `"ski"`, not `"skiing"`).
- No hardware or runtime limits for re-runs.
- FPVSum annotations are treated as **hidden test labels**: not used for training or tuning.

## Integrity

Outputs are produced only by `run.sh` on the video file. No video IDs are hardcoded, no ground
truth is read, and the label is never parsed from the file name (FPVSum names start with the
category).

## Edge cases handled

- Videos shorter than 26.7 s (3 in the manifest): one 4.0 s segment is already over the 15%
  budget, so exactly one 4.0 s segment is written (soft penalty accepted; ≥ 1 segment ≥ 4 s is mandatory).
- Videos up to 33 min (17 over 10 min): highlights scale with duration (max 10 windows).
- Unreadable or truncated files: decoding stops cleanly; a JSON is still written.

## Tests

```bash
PYTHONPATH=code python -m pytest code/tests -q
```
`test_contract.py`, `test_validate.py` and `test_zeroshot.py` check the schema, the validator and
the zero-shot heads on synthetic embeddings (no GPU or video needed); `test_pipeline.py` and `test_video_io.py` need PyAV and generate
synthetic videos (`code/scripts/make_synthetic_videos.py`).

## Layout

| Path | Purpose |
|---|---|
| `run.sh` | Entry point: `run.sh <video.mp4> <out.json>` |
| `validate.py` | Team-written validator (stdlib only) |
| `code/camspector/contract.py` | The only place schema details live |
| `code/camspector/pipeline.py` | Orchestration with baseline fallback |
| `code/camspector/zeroshot.py` | Label, highlight and action heads (numpy) |
| `code/camspector/clip_encoder.py` | CLIP ViT-B/16 loader (local `weights/` first) |
| `code/camspector/video_io.py` | PTS-based decoding |
| `code/camspector/features/` | SlowFast EPIC-100 feature extractor |
| `code/configs/` | Kit files: `manifest.csv`, `supported_classes.json` |
| `code/scripts/` | Batch runner, EPIC stats, encoder benchmark, parity test |
| `code/tests/` | pytest suite |
| `docs/` | EPIC stats, feature-parity results, storage plan, dev notes |
| `experiments/` | Measured results only (GPU check, encoder benchmark) |
| `weights/` | CLIP + verb prior, created by `code/scripts/download_weights.py` (not in git) |
| `outputs/` | One `<video_id>.json` per manifest video |

## Licences

EPIC-KITCHENS-100 annotations and models are CC BY-NC 4.0 and are not redistributed in git;
`weights/epic_pair_prior.json` is derived from the train split at setup time. CLIP weights: MIT (OpenAI).
