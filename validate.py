#!/usr/bin/env python3
"""Team SpectraCore's validator for Track 4 output JSONs (standard library only).

The organisers did not ship validate.py ("Updated Track 04" email, 20 Nov 2026) and asked
teams to write their own. This script checks every rule in TEAM_README.md section 3-4,
plus the email's clarifications:

  * all 6 top-level fields; types; finite numbers; score in [0, 1]; duration_sec > 0
  * video_id matches the file name; team_id identical across all files
  * duration_sec within +-0.5 s of manifest.csv (when the manifest is given)
  * segments: 0 <= start < end <= duration_sec + 0.5
  * actions: 0-2000 items; verb/noun/label in supported_classes.json (when given)
  * highlights: >= 1 item, chronologically sorted, strictly non-overlapping,
    at least one segment >= 4.0 s
  * warnings: segments < 4.0 s (ignored in scoring), total > 15% of duration_sec
  * --dir: every manifest video has a JSON, except the two FPVSum videos the organisers
    excluded (Scuba_KD74Ywuy9ng, Scuba_Um4kqB8bQBg)

Usage:
  python validate.py --file outputs/P02_109.json -v
  python validate.py --dir outputs/
  python validate.py --dir outputs/ --manifest code/configs/manifest.csv \
                     --classes code/configs/supported_classes.json
Exit code 0 = every file valid (and none missing); 1 = at least one problem.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "code" / "configs" / "manifest.csv"
DEFAULT_CLASSES = ROOT / "code" / "configs" / "supported_classes.json"
EXCLUDED = {"Scuba_KD74Ywuy9ng", "Scuba_Um4kqB8bQBg"}

KEYS = ("team_id", "video_id", "duration_sec", "label", "actions", "highlights")
MIN_SEG = 4.0
BUDGET = 0.15
END_TOL = 0.5
DUR_TOL = 0.5
MAX_ACTIONS = 2000


def is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def check_segment(item, dur, name, errs):
    if not isinstance(item, dict):
        errs.append(f"{name} must be an object")
        return None
    seg = item.get("segment")
    if not (isinstance(seg, list) and len(seg) == 2 and all(is_num(v) for v in seg)):
        errs.append(f"{name}.segment must be [start, end] with finite numbers")
        return None
    s, e = float(seg[0]), float(seg[1])
    ok = True
    if s < 0:
        errs.append(f"{name}: start {s} < 0"); ok = False
    if not s < e:
        errs.append(f"{name}: start {s} is not < end {e}"); ok = False
    if dur is not None and e > dur + END_TOL:
        errs.append(f"{name}: end {e} > duration_sec + {END_TOL}"); ok = False
    return (s, e) if ok else None


def validate_obj(obj, file_stem, manifest, classes):
    errs, warns = [], []
    if not isinstance(obj, dict):
        return ["top level must be a JSON object"], warns
    missing = [k for k in KEYS if k not in obj]
    if missing:
        return [f"missing required field(s): {', '.join(missing)}"], warns
    extra = [k for k in obj if k not in KEYS]
    if extra:
        warns.append(f"extra top-level field(s): {', '.join(extra)}")

    for k in ("team_id", "video_id"):
        if not isinstance(obj[k], str) or not obj[k].strip():
            errs.append(f"{k} must be a non-empty string")
    if isinstance(obj["video_id"], str) and file_stem is not None and obj["video_id"] != file_stem:
        errs.append(f"video_id {obj['video_id']!r} does not match file name {file_stem!r}.json")

    dur = obj["duration_sec"]
    if not is_num(dur) or dur <= 0:
        errs.append("duration_sec must be a positive number")
        dur = None
    elif manifest and obj.get("video_id") in manifest:
        ref = manifest[obj["video_id"]]
        if abs(dur - ref) > DUR_TOL + 1e-9:
            errs.append(f"duration_sec {dur} is not within +-{DUR_TOL}s of manifest ({ref})")
    elif manifest and isinstance(obj.get("video_id"), str):
        errs.append(f"video_id {obj['video_id']!r} is not in manifest.csv")

    lab = obj["label"]
    if not isinstance(lab, dict) or "name" not in lab or "score" not in lab:
        errs.append('label must be an object {"name": string, "score": number}')
    else:
        if not isinstance(lab["name"], str) or not lab["name"]:
            errs.append("label.name must be a non-empty string")
        elif classes and lab["name"] not in classes["labels"]:
            errs.append(f"label.name {lab['name']!r} is not in supported_classes.json labels")
        if not is_num(lab["score"]) or not 0 <= lab["score"] <= 1:
            errs.append("label.score must be a number in [0, 1]")

    acts = obj["actions"]
    if not isinstance(acts, list):
        errs.append("actions must be an array")
        acts = []
    if len(acts) > MAX_ACTIONS:
        errs.append(f"actions has {len(acts)} items (max {MAX_ACTIONS})")
    bad_vocab = 0
    for i, a in enumerate(acts):
        check_segment(a, dur, f"actions[{i}]", errs)
        if not isinstance(a, dict):
            continue
        for k in ("verb", "noun"):
            if not isinstance(a.get(k), str) or not a.get(k):
                errs.append(f"actions[{i}].{k} must be a non-empty string")
        if classes and isinstance(a.get("verb"), str) and isinstance(a.get("noun"), str):
            if a["verb"] not in classes["verbs"] or a["noun"] not in classes["nouns"]:
                bad_vocab += 1
                if bad_vocab <= 3:
                    errs.append(f"actions[{i}]: {a['verb']}/{a['noun']} not in supported_classes.json")
        if not is_num(a.get("score")) or not 0 <= a["score"] <= 1:
            errs.append(f"actions[{i}].score must be a number in [0, 1]")
    if bad_vocab > 3:
        errs.append(f"... {bad_vocab - 3} more actions with unsupported verb/noun")

    hls = obj["highlights"]
    if not isinstance(hls, list) or not hls:
        errs.append("highlights must be a non-empty array")
        return errs, warns
    prev_end, total, longest = -math.inf, 0.0, 0.0
    for i, h in enumerate(hls):
        seg = check_segment(h, dur, f"highlights[{i}]", errs)
        if seg is None:
            continue
        s, e = seg
        if s < prev_end:
            errs.append(f"highlights[{i}] overlaps the previous segment or is not sorted")
        elif s == prev_end:
            warns.append(f"highlights[{i}] touches the previous segment (start == previous end)")
        prev_end = e
        total += e - s
        longest = max(longest, e - s)
        if e - s < MIN_SEG:
            warns.append(f"highlights[{i}] is {e - s:.2f}s long (< {MIN_SEG}s, ignored in scoring)")
    if longest < MIN_SEG - 1e-9:
        errs.append(f"no highlight segment is >= {MIN_SEG}s")
    if dur is not None and total > BUDGET * dur + 1e-6:
        warns.append(f"highlights total {total:.2f}s = {100 * total / dur:.1f}% of duration "
                     f"(budget 15%, soft penalty)")
    return errs, warns


def load_manifest(path):
    if not path or not Path(path).is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["video_id"]: float(r["duration_sec"]) for r in csv.DictReader(f)}


def load_classes(path):
    if not path or not Path(path).is_file():
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: set(raw.get(k, [])) for k in ("labels", "verbs", "nouns")}


def validate_file(path: Path, manifest, classes):
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:  # unreadable or not JSON
        return None, [f"cannot parse JSON: {e}"], []
    errs, warns = validate_obj(obj, path.stem, manifest, classes)
    return obj, errs, warns


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--file", type=Path, help="validate one JSON file")
    g.add_argument("--dir", type=Path, help="validate every *.json in a folder")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST),
                    help="manifest.csv for duration and completeness checks ('' to skip)")
    ap.add_argument("--classes", default=str(DEFAULT_CLASSES),
                    help="supported_classes.json for vocabulary checks ('' to skip)")
    ap.add_argument("-v", "--verbose", action="store_true", help="also print warnings")
    a = ap.parse_args(argv)

    manifest = load_manifest(a.manifest) if a.manifest else {}
    classes = load_classes(a.classes) if a.classes else None
    files = [a.file] if a.file else sorted(a.dir.glob("*.json"))
    if not files:
        print(f"FAIL: no JSON files found in {a.dir}")
        return 1

    n_bad, n_warn, team_ids = 0, 0, {}
    for p in files:
        obj, errs, warns = validate_file(p, manifest, classes)
        if isinstance(obj, dict) and isinstance(obj.get("team_id"), str):
            team_ids.setdefault(obj["team_id"], []).append(p.name)
        if errs:
            n_bad += 1
            print(f"FAIL  {p.name}")
            for e in errs:
                print(f"      - {e}")
        elif a.verbose or a.file:
            print(f"OK    {p.name}")
        if warns:
            n_warn += 1
            if a.verbose:
                for w in warns:
                    print(f"      ! {w}")

    problems = n_bad
    if len(team_ids) > 1:
        problems += 1
        print(f"FAIL  team_id differs across files: {sorted(team_ids)}")
    if a.dir and manifest:
        have = {p.stem for p in files}
        missing = sorted(set(manifest) - have - EXCLUDED)
        if missing:
            problems += 1
            print(f"FAIL  {len(missing)} manifest video(s) have no JSON "
                  f"(a missing file scores 0): {', '.join(missing[:10])}"
                  f"{' ...' if len(missing) > 10 else ''}")
        expected = len(set(manifest) - EXCLUDED)
        print(f"\n{len(files)} file(s) checked, {n_bad} invalid, {n_warn} with warnings; "
              f"{expected - len(missing)}/{expected} scored manifest videos covered "
              f"({len(EXCLUDED)} excluded by the organisers).")
    else:
        print(f"\n{len(files)} file(s) checked, {n_bad} invalid, {n_warn} with warnings.")
    print("RESULT: PASS" if problems == 0 else "RESULT: FAIL")
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
