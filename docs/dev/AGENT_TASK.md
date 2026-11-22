# Task for the coding agent on the teammate's laptop

You are helping the CAMSPECTOR team (Samsung PRISM XR Hackathon 2026, Track 4). This repo is
the team's code. The team member's own laptop has no room for data and only a 4 GB GPU, so
**you run the GPU and data work on this machine**. Work in order, stop at the end, and
report. Read `README.md`, `docs/FEATURES_PARITY.md` and `docs/STORAGE_PLAN.md` first.

## Rules (non-negotiable)

1. **Never commit or upload videos, annotation CSVs, feature files or weights.** They are
   CC BY-NC licensed datasets/models and are git-ignored (`data/`, `weights/`, `/features/`).
   Only code, docs and small result JSON/MD files go back.
2. **Never report a number you did not compute.** Every number must come from a file in
   `experiments/` produced by code in this repo.
3. Do not edit `code/camspector/contract.py` (owned by the main machine, waiting on the kit).
4. Stay inside this folder. Keep all caches in `./.cache` (the setup script sets this).
5. If something needs a login, a licence click-through or more than ~15 GB of disk, **stop and
   ask the human** at this machine.

## Step 1: Environment + GPU benchmark

- Windows: `powershell -ExecutionPolicy Bypass -File .\setup_and_bench.ps1`
- Linux: `bash setup_and_bench.sh`

**Done when:**
- `experiments/gpu_check.json` has `"all_ok": true`.
- `experiments/pytest.txt` shows that all tests passed.
- `experiments/bench_encoders_<host>.json` has GPU fp16 numbers for all 4 encoders.

If the GPU check fails, fix the driver/torch install. Don't continue on CPU.

## Step 2: Download the parity data (~0.5 GB of video + the features tarball)

```bash
.venv/bin/python code/scripts/parity/download_parity_data.py      # Windows: .venv\Scripts\python.exe
```
This downloads only from official sources:
- videos P28_18 and P09_08, 2 minutes each, into `data/epic_videos/`;
- the SlowFast EPIC-100 checkpoint into `weights/slowfast_epic100/`;
- from the ActionFormer features tarball, only those two videos' features, into `data/af_features/`.

If the official EPIC downloader or the direct URL fails, the script says so. Look at
`.cache/epic-kitchens-download-scripts` to find the right command or URL, and fix the script
(commit the fix). Before going further, record in your report:
- the exact feature file names;
- the `.npz` keys and shape;
- the `fps`, `feature_stride` and `num_frames` stored in the annotation JSON.

## Step 3: Implement the SlowFast EPIC extractor

Create `code/camspector/features/slowfast_epic.py` with:
```python
def extract(video_path, *, variant: dict, device="cuda") -> tuple[np.ndarray, np.ndarray]:
    """Return (features [N, 2304] float32, clip_centre_times_s [N])."""
```
Requirements:
- Decode with **our** `camspector.video_io.iter_frames` (PTS-based). Resample to **30 fps by
  timestamp**: the sources are 59.94 and 29.97 fps.
- Clips are 32 frames with a stride of 16 frames. The fast pathway uses all 32 frames and the
  slow pathway uses every 4th frame (alpha = 4). The output is the 2304-d global-average-pooled
  feature (2048 slow + 256 fast) that goes into the classification head, taken **before** the
  head projection.
- Build the model with the **same architecture as the checkpoint**:
  `configs/EPIC-KITCHENS/SLOWFAST_8x8_R50.yaml` from
  https://github.com/epic-kitchens/epic-kitchens-slowfast, plus the checkpoint's
  `model_state`. Vendoring the needed model files under
  `code/camspector/features/third_party/` is fine **if that repo's LICENSE allows it**. Check
  the licence, keep its header, and note it in the report. Mapping the
  weights onto `pytorchvideo.models.create_slowfast` is also fine, **only if** every
  checkpoint key is used and the shapes match. Print and assert this.
- Normalise with the mean/std from that config, in fp16 on GPU.
- Things the `variant` dict must let you sweep:
  - resize: short side 256 then centre crop 224, or full frame resized to 224×224, or short side 224 with no crop;
  - frame source: resize straight from the 1080p frames, or first downscale to 456×256 to mimic the official `rgb_frames` JPEGs;
  - first clip start offset: 0 or 8 frames;
  - end handling: drop the incomplete last clip, or pad it by repeating the last frame.

## Step 4: Run the parity test

For each variant and each of the 2 videos:
```bash
python code/scripts/parity/compare_features.py --ours features/slowfast_epic/<variant>/<vid>.npy \
   --published data/af_features/<vid>.npz --variant <variant> \
   --out experiments/parity/<vid>__<variant>.json
```
Write `code/scripts/parity/run_parity.py`, which sweeps the variants, extracts, compares, and
writes `experiments/parity/summary.md`. The summary is a table with one row per (video,
variant): offset, median cosine, p5, clip counts and pass/fail.

**Pass:** median cosine ≥ 0.95, p5 ≥ 0.90, and clip counts within ±1 **on both videos**,
with the same variant. Also record:
- extraction seconds per video-minute;
- peak VRAM;
- whether decoding or the model is the bottleneck.

## Step 5: Report and return results

1. Write `docs/PARITY_RESULTS.md`. Include:
   - what you ran;
   - the summary table, copied from `experiments/parity/summary.md`;
   - the best variant and whether it passed;
   - extraction speed and VRAM;
   - every problem you hit;
   - your recommendation: train on the published features (pass), or extract ourselves (fail).
2. Return the results:
   - **If you can push to the GitHub repo:** create branch `parity-results` and commit the code
     plus `experiments/` and `docs/PARITY_RESULTS.md`. Run `git status` first to check that
     no data or weights are staged. Push the branch. **Do not push to `main`.**
   - **Otherwise:** zip `experiments/`, `docs/PARITY_RESULTS.md` and any changed code
     into `results_parity_<host>.zip` and give it to the human to send back.
3. You can then delete `data/epic_videos/` and `features/` to free space. Keep
   `data/af_features/` and `weights/` in case the team needs a re-run.
