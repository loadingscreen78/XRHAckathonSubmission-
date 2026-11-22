"""Download the data for the SlowFast feature-parity test (run on the teammate machine).

Downloads (official sources only; nothing is redistributed):
  1. EPIC-100 videos P28_18 (59.94 fps) and P09_08 (29.97 fps), ~2 min each, ~0.5 GB total
     -> data/epic_videos/<video_id>.MP4
  2. ActionFormer EPIC features tarball (Google Drive, md5 add9803756afd9a023bc9a9c547e0229)
     -> only the two videos' feature files and the annotation JSONs are kept in data/af_features/
  3. SlowFast EPIC-100 checkpoint (Dropbox, from epic-kitchens-slowfast README)
     -> weights/slowfast_epic100/SlowFast.pyth

Usage: python code/scripts/parity/download_parity_data.py [--keep-tarball] [--skip-features]
Needs ~12 GB free while running (tarball size is not documented); ~1 GB after cleanup.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VIDEOS = ["P28_18", "P09_08"]
AF_GDRIVE_ID = "1Z4U_dLuu6_cV5NBIrSzsSDOOj2Uar85X"
AF_MD5 = "add9803756afd9a023bc9a9c547e0229"
SLOWFAST_URL = "https://www.dropbox.com/s/uxb6i2xkn91xqzi/SlowFast.pyth?dl=1"
DL_REPO = "https://github.com/epic-kitchens/epic-kitchens-download-scripts.git"
# EPIC-55 participants' videos live in this dataset DOI; tried as a fallback only.
BRIS_55 = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d/videos/{split}/{p}/{v}.MP4"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def need_space(gb: float) -> None:
    free = shutil.disk_usage(ROOT).free / 2**30
    log(f"free disk: {free:.1f} GB (need ~{gb} GB)")
    if free < gb:
        sys.exit(f"Not enough disk space: {free:.1f} GB free, need ~{gb} GB.")


def fetch(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    log(f"GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and done % (50 << 20) < (1 << 20):
                log(f"  {done/2**20:.0f}/{total/2**20:.0f} MB")
    tmp.replace(dst)


def download_videos() -> None:
    out = ROOT / "data" / "epic_videos"
    out.mkdir(parents=True, exist_ok=True)
    todo = [v for v in VIDEOS if not (out / f"{v}.MP4").exists()]
    if not todo:
        log("videos already present")
        return
    # 1) official verified downloader
    repo = ROOT / ".cache" / "epic-kitchens-download-scripts"
    try:
        if not repo.exists():
            subprocess.run(["git", "clone", "--depth", "1", DL_REPO, str(repo)], check=True)
        raw = ROOT / "data" / "epic_raw"
        subprocess.run([sys.executable, "epic_video_downloader_verify.py",
                        "--specific-videos", ",".join(todo), "--output-path", str(raw),
                        "--max-retries", "3"], cwd=repo, check=True)
        for v in todo:
            hits = list(raw.rglob(f"{v}.MP4"))
            if hits:
                shutil.move(str(hits[0]), out / f"{v}.MP4")
    except Exception as e:  # tool missing metadata CSVs, network, etc.
        log(f"official downloader failed: {e}")
    # 2) direct URL fallback (no MD5 list here -> size/decode check in the parity run)
    for v in todo:
        if (out / f"{v}.MP4").exists():
            continue
        p = v.split("_")[0]
        for split in ("train", "test"):
            try:
                fetch(BRIS_55.format(split=split, p=p, v=v), out / f"{v}.MP4")
                break
            except Exception as e:
                log(f"  {split}: {e}")
    missing = [v for v in VIDEOS if not (out / f"{v}.MP4").exists()]
    if missing:
        sys.exit(f"Could not download {missing}. Try Academic Torrents (EPIC-KITCHENS-55 "
                 "d08f4591d1865bbe3436d1eb25ed55aae8b8f043) and put the files in data/epic_videos/.")


def download_features(keep_tarball: bool) -> None:
    out = ROOT / "data" / "af_features"
    if all((out / f"{v}.npz").exists() for v in VIDEOS):
        log("features already present")
        return
    tar = ROOT / "data" / "epic_kitchens.tar.gz"
    if not tar.exists():
        try:
            import gdown  # noqa: F401
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "gdown"], check=True)
        import gdown
        gdown.download(id=AF_GDRIVE_ID, output=str(tar), quiet=False)
    got = md5(tar)
    if got != AF_MD5:
        sys.exit(f"tarball md5 {got} != expected {AF_MD5} (partial or wrong file)")
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar, "r:gz") as t:
        for m in t:
            name = Path(m.name).name
            keep = any(name.startswith(v + ".") for v in VIDEOS) or (
                m.isfile() and "annotation" in m.name and name.endswith(".json"))
            if keep and m.isfile():
                f = t.extractfile(m)
                (out / name).write_bytes(f.read())
                log(f"  kept {m.name}")
    if not keep_tarball:
        tar.unlink()
        log("deleted tarball (use --keep-tarball to keep it)")


def download_checkpoint() -> None:
    dst = ROOT / "weights" / "slowfast_epic100" / "SlowFast.pyth"
    if dst.exists():
        log("checkpoint already present")
        return
    fetch(SLOWFAST_URL, dst)
    log(f"checkpoint md5 {md5(dst)} size {dst.stat().st_size/2**20:.0f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-tarball", action="store_true")
    ap.add_argument("--skip-features", action="store_true")
    a = ap.parse_args()
    need_space(1.0 if a.skip_features else 12.0)
    download_videos()
    download_checkpoint()
    if not a.skip_features:
        download_features(a.keep_tarball)
    log("done")


if __name__ == "__main__":
    main()
