#!/usr/bin/env python3
"""Run the whole Track 4 manifest fast, with almost no disk.

For every manifest video (the 2 Scuba videos excluded by the organisers are skipped):
  EPIC-KITCHENS-100 -> download that one video (official downloader) -> run -> delete it
  FPVSum            -> read it from --fpvsum-dir (the 47 videos are small; download them first)

Fast by default: --workers videos are handled at once (downloads overlap with processing),
FPVSum first, then EPIC shortest-first so finished files pile up quickly. Resumable: a video
whose JSON already exists and validates is skipped. CPU works; a GPU is much faster.

Usage:
  python code/scripts/stream_run.py --out outputs/ --fpvsum-dir /path/to/fpvsum_videos --workers 3
Options:
  --work DIR        temporary download folder (default: .cache/epic_tmp)
  --epic-dir DIR    use EPIC videos already on disk instead of downloading
  --only epic|fpvsum
  --keep            do not delete downloaded EPIC videos
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import logging
import multiprocessing as mp
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from camspector import contract as C  # noqa: E402

DL_REPO = "https://github.com/epic-kitchens/epic-kitchens-download-scripts"
DL_DIR = ROOT / ".cache" / "epic-kitchens-download-scripts"
VIDEO_EXT = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
DATASET = {"epic": "epic_kitchens_100", "fpvsum": "fpvsum"}
FMT = "%(asctime)s %(processName)s %(levelname)s %(name)s: %(message)s"

log = logging.getLogger("stream_run")


def load_validator():
    spec = importlib.util.spec_from_file_location("validate", ROOT / "validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_video(root, vid: str):
    if root is None or not Path(root).exists():
        return None
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXT and p.stem == vid:
            return p
    return None


def ensure_downloader():
    if not (DL_DIR / "epic_downloader.py").exists():
        DL_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", DL_REPO, str(DL_DIR)], check=True)


def download_epic(vid: str, work: Path):
    ensure_downloader()
    work.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "epic_downloader.py", "--videos", "--specific-videos", vid,
                    "--output-path", str(work)], cwd=DL_DIR, check=True,
                   stdout=subprocess.DEVNULL)
    return find_video(work, vid)


def _init_worker():
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format=FMT)


def handle(job: dict) -> list:
    """Download (EPIC) -> run -> validate -> delete. Runs in a worker process."""
    from camspector.pipeline import process

    vid, ds, out = job["vid"], job["ds"], Path(job["out"]) / f"{job['vid']}.json"
    t0, path, used_model, status = time.time(), None, None, ""
    vid_work = Path(job["work"]) / vid
    try:
        if ds == DATASET["epic"]:
            path = find_video(job["epic_dir"], vid)
            if path is None:
                path = download_epic(vid, vid_work)
        else:
            path = find_video(job["fpvsum_dir"], vid)
        if path is None:
            raise FileNotFoundError(f"video file for {vid} not found")
        used_model = process(path, out).get("used_model")
        V = load_validator()
        errs = V.validate_file(out, V.load_manifest(C.MANIFEST_PATH),
                               V.load_classes(C.SUPPORTED_CLASSES_PATH))[1]
        status = "ok" if not errs else f"invalid: {errs[0]}"
    except Exception as e:  # one bad video must not stop the run
        status = f"failed: {e}"
        logging.getLogger("stream_run").error("%s: %s", vid, e)
    finally:
        if not job["keep"] and vid_work.exists():
            shutil.rmtree(vid_work, ignore_errors=True)
    return [vid, ds, status, used_model, round(time.time() - t0, 1)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=ROOT / "outputs")
    ap.add_argument("--fpvsum-dir", type=Path, default=None)
    ap.add_argument("--epic-dir", type=Path, default=None)
    ap.add_argument("--work", type=Path, default=ROOT / ".cache" / "epic_tmp")
    ap.add_argument("--only", choices=sorted(DATASET), default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop starting new videos after this many minutes (0 = no limit)")
    ap.add_argument("--log", type=Path, default=ROOT / "experiments" / "run_log.csv")
    a = ap.parse_args(argv)
    _init_worker()
    deadline = time.time() + a.max_minutes * 60 if a.max_minutes > 0 else None

    V = load_validator()
    mref = V.load_manifest(C.MANIFEST_PATH)
    classes = V.load_classes(C.SUPPORTED_CLASSES_PATH)
    with open(C.MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["video_id"] not in C.EXCLUDED_VIDEO_IDS]
    if a.only:
        rows = [r for r in rows if r["dataset"] == DATASET[a.only]]
    # FPVSum first (already on disk), then EPIC shortest-first
    rows.sort(key=lambda r: (r["dataset"] != DATASET["fpvsum"], float(r["duration_sec"])))

    a.out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows
            if not ((a.out / f"{r['video_id']}.json").exists()
                    and not V.validate_file(a.out / f"{r['video_id']}.json", mref, classes)[1])]
    log.info("%d of %d videos already done; %d to go with %d worker(s)",
             len(rows) - len(todo), len(rows), len(todo), a.workers)
    if any(r["dataset"] == DATASET["epic"] for r in todo) and a.epic_dir is None:
        ensure_downloader()  # once, before workers start

    jobs = [{"vid": r["video_id"], "ds": r["dataset"], "out": str(a.out), "work": str(a.work),
             "fpvsum_dir": a.fpvsum_dir, "epic_dir": a.epic_dir, "keep": a.keep} for r in todo]
    a.log.parent.mkdir(parents=True, exist_ok=True)
    new_log = not a.log.exists()
    counts = {"ok": 0, "fallback": 0, "failed": 0}
    with open(a.log, "a", newline="", encoding="utf-8") as logf:
        w = csv.writer(logf)
        if new_log:
            w.writerow(["video_id", "dataset", "status", "model_used", "seconds"])

        def record(row, k):
            w.writerow(row)
            logf.flush()
            counts["ok" if row[2] == "ok" else "failed"] += 1
            if row[3] is False:
                counts["fallback"] += 1
            log.info("[%d/%d] %s %s (model=%s, %.0f s)", k, len(jobs), row[0], row[2], row[3], row[4])

        if a.workers <= 1:
            for k, j in enumerate(jobs, 1):
                if deadline and time.time() > deadline:
                    log.info("time limit reached: stopping (%d videos not started)", len(jobs) - k + 1)
                    break
                record(handle(j), k)
        else:
            ctx = mp.get_context("spawn")  # CUDA-safe
            with ProcessPoolExecutor(a.workers, mp_context=ctx, initializer=_init_worker) as ex:
                futs = {ex.submit(handle, j): j for j in jobs}
                stopped = False
                k = 0
                for fut in as_completed(futs):
                    if fut.cancelled():
                        continue
                    k += 1
                    j = futs[fut]
                    try:
                        row = fut.result()
                    except Exception as e:  # a crashed worker: record it; a re-run resumes
                        row = [j["vid"], j["ds"], f"failed: worker crashed ({e})", None, 0.0]
                    record(row, k)
                    if deadline and not stopped and time.time() > deadline:
                        stopped = True
                        n = sum(f.cancel() for f in futs)
                        log.info("time limit reached: %d videos not started; finishing the ones "
                                 "in progress", n)

    log.info("summary: %s", counts)
    if counts["fallback"]:
        log.warning("%d video(s) used the baseline fallback: see 'model path failed' above",
                    counts["fallback"])
    return V.main(["--dir", str(a.out)])


if __name__ == "__main__":
    sys.exit(main())
