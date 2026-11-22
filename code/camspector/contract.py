"""Output contract for Track 4 -- the ONLY place schema details live.

Matches the official student kit (TEAM_README.md, section 3 "Output JSON Format & Field
Reference") and the organisers' clarification email "Updated Track 04" (20 Nov 2026).
No other module may hard-code key names, budgets or ranges.

Official per-video JSON (6 top-level fields, same for EPIC-KITCHENS-100 and FPVSum):
    {
      "team_id": "SpectraCore",
      "video_id": "P02_109",
      "duration_sec": 512.3,
      "label": {"name": "cooking", "score": 0.96},
      "actions": [{"segment": [12.4, 18.2], "verb": "open", "noun": "fridge", "score": 0.88}],
      "highlights": [{"segment": [11.0, 19.5]}, {"segment": [45.2, 53.0]}]
    }
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# ---------------------------------------------------------------------------
# Schema constants (kit: TEAM_README.md section 3)
# ---------------------------------------------------------------------------
TEAM_ID = "SpectraCore"            # must be identical across all submitted files

KEY_TEAM_ID = "team_id"
KEY_VIDEO_ID = "video_id"
KEY_DURATION = "duration_sec"
KEY_LABEL = "label"
KEY_ACTIONS = "actions"
KEY_HIGHLIGHTS = "highlights"
TOP_LEVEL_KEYS = (KEY_TEAM_ID, KEY_VIDEO_ID, KEY_DURATION, KEY_LABEL, KEY_ACTIONS, KEY_HIGHLIGHTS)

LABEL_NAME = "name"
LABEL_SCORE = "score"

SEGMENT = "segment"                # [start, end] in seconds
ACTION_VERB = "verb"
ACTION_NOUN = "noun"
ACTION_SCORE = "score"

SCORE_RANGE = (0.0, 1.0)
MAX_ACTIONS = 2000
END_TOLERANCE_S = 0.5              # 0 <= start < end <= duration_sec + 0.5
DURATION_TOLERANCE_S = 0.5         # duration_sec within +-0.5 s of manifest.csv
TIME_DECIMALS = 2                  # "Timestamps should be in seconds (up to 2 decimal places)"

HIGHLIGHT_BUDGET_FRAC = 0.15       # target total <= 15% of duration (soft penalty above)
HIGHLIGHT_TARGET_FRAC = 0.145      # our fill target, kept just below the budget
MIN_HIGHLIGHT_LEN_S = 4.0          # segments < 4.0 s are ignored in scoring; >= 1 required

DEFAULT_LABEL = "other"            # fallback when no label model output is available
DEFAULT_LABEL_SCORE = 0.0

# Organisers (email, 20 Nov 2026): these two FPVSum videos are no longer on YouTube and
# are excluded from final scoring.
EXCLUDED_VIDEO_IDS = frozenset({"Scuba_KD74Ywuy9ng", "Scuba_Um4kqB8bQBg"})

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
SUPPORTED_CLASSES_PATH = CONFIG_DIR / "supported_classes.json"
MANIFEST_PATH = CONFIG_DIR / "manifest.csv"


def video_id_from_path(path: str | Path) -> str:
    """video_id = file stem (P01_11.MP4 -> P01_11; Ski_-Z5a4UQmZLs.mp4 -> Ski_-Z5a4UQmZLs)."""
    return Path(path).stem


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
def _r(x: float) -> float:
    return round(float(x), TIME_DECIMALS)


def _clamp01(x: float | None, default: float = 0.0) -> float:
    lo, hi = SCORE_RANGE
    if x is None or not math.isfinite(float(x)):
        return default
    return min(hi, max(lo, float(x)))


@dataclass
class Action:
    verb: str
    noun: str
    start: float
    end: float
    score: float | None = None

    def to_json(self) -> dict:
        return {
            SEGMENT: [_r(self.start), _r(self.end)],
            ACTION_VERB: self.verb,
            ACTION_NOUN: self.noun,
            ACTION_SCORE: round(_clamp01(self.score, 1.0), 6),
        }


@dataclass
class Highlight:
    start: float
    end: float
    importance: float = 0.5        # internal ranking only; not written to the JSON

    @property
    def length(self) -> float:
        return self.end - self.start

    def to_json(self) -> dict:
        return {SEGMENT: [_r(self.start), _r(self.end)]}


@dataclass
class Prediction:
    video_id: str
    duration: float
    label: str = DEFAULT_LABEL
    label_score: float = DEFAULT_LABEL_SCORE
    actions: list[Action] = field(default_factory=list)
    highlights: list[Highlight] = field(default_factory=list)
    team_id: str = TEAM_ID

    def to_json(self) -> dict:
        return {
            KEY_TEAM_ID: self.team_id,
            KEY_VIDEO_ID: self.video_id,
            KEY_DURATION: _r(self.duration),
            KEY_LABEL: {LABEL_NAME: self.label, LABEL_SCORE: round(_clamp01(self.label_score), 6)},
            KEY_ACTIONS: [a.to_json() for a in self.actions[:MAX_ACTIONS]],
            KEY_HIGHLIGHTS: [h.to_json() for h in self.highlights],
        }


def highlight_budget_s(duration: float) -> float:
    return HIGHLIGHT_BUDGET_FRAC * float(duration)


# ---------------------------------------------------------------------------
# Supported classes (kit: supported_classes.json -> {"labels", "verbs", "nouns"})
# ---------------------------------------------------------------------------
@dataclass
class ClassSet:
    """Supported verbs/nouns/labels. Empty sets mean 'not loaded' (no vocabulary check)."""
    verbs: set[str] = field(default_factory=set)
    nouns: set[str] = field(default_factory=set)
    labels: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, path: str | Path | None = SUPPORTED_CLASSES_PATH) -> "ClassSet":
        if path is None or not Path(path).exists():
            return cls()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(verbs=set(raw.get("verbs", [])), nouns=set(raw.get("nouns", [])),
                   labels=set(raw.get("labels", [])))

    def allows_action(self, verb: str, noun: str) -> bool:
        return (not self.verbs or verb in self.verbs) and (not self.nouns or noun in self.nouns)

    def allows_label(self, label: str) -> bool:
        return not self.labels or label in self.labels


# ---------------------------------------------------------------------------
# Validation (same rules as validate.py at the repo root)
# ---------------------------------------------------------------------------
def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _check_segment(item, duration: float, name: str) -> tuple[list[str], tuple[float, float] | None]:
    if not isinstance(item, dict):
        return [f"{name} must be an object"], None
    seg = item.get(SEGMENT)
    if not (isinstance(seg, list) and len(seg) == 2 and all(_is_num(v) for v in seg)):
        return [f"{name}.segment must be [start, end] with finite numbers"], None
    s, e = float(seg[0]), float(seg[1])
    errs = []
    if s < 0:
        errs.append(f"{name}.segment start {s} < 0")
    if not s < e:
        errs.append(f"{name}.segment start must be < end ({s}, {e})")
    if e > duration + END_TOLERANCE_S:
        errs.append(f"{name}.segment end {e} > duration_sec + {END_TOLERANCE_S}")
    return errs, (s, e)


def check_output(obj: dict, duration: float | None = None, classes: ClassSet | None = None,
                 expected_duration: float | None = None) -> list[str]:
    """Return contract violations (empty list == valid). `duration` is accepted for
    backwards compatibility and used as the expected duration when given."""
    errs: list[str] = []
    classes = classes or ClassSet()
    if expected_duration is None:
        expected_duration = duration
    if not isinstance(obj, dict):
        return ["top level must be a JSON object"]
    missing = [k for k in TOP_LEVEL_KEYS if k not in obj]
    extra = [k for k in obj if k not in TOP_LEVEL_KEYS]
    if missing:
        return [f"missing top-level keys: {missing}"]
    if extra:
        errs.append(f"unexpected top-level keys: {extra}")

    for k in (KEY_TEAM_ID, KEY_VIDEO_ID):
        if not isinstance(obj[k], str) or not obj[k].strip():
            errs.append(f"{k} must be a non-empty string")

    dur = obj[KEY_DURATION]
    if not _is_num(dur) or dur <= 0:
        errs.append("duration_sec must be a positive number")
        dur = float(expected_duration or 0.0)
    elif expected_duration is not None and expected_duration > 0 and \
            abs(dur - expected_duration) > DURATION_TOLERANCE_S + 1e-9:
        errs.append(f"duration_sec {dur} not within +-{DURATION_TOLERANCE_S}s of {expected_duration}")

    lab = obj[KEY_LABEL]
    if not isinstance(lab, dict) or set(lab) != {LABEL_NAME, LABEL_SCORE}:
        errs.append('label must be {"name": string, "score": number}')
    else:
        if not isinstance(lab[LABEL_NAME], str) or not lab[LABEL_NAME]:
            errs.append("label.name must be a non-empty string")
        elif not classes.allows_label(lab[LABEL_NAME]):
            errs.append(f"unsupported label {lab[LABEL_NAME]!r}")
        if not _is_num(lab[LABEL_SCORE]) or not (SCORE_RANGE[0] <= lab[LABEL_SCORE] <= SCORE_RANGE[1]):
            errs.append("label.score must be a number in [0, 1]")

    acts = obj[KEY_ACTIONS]
    if not isinstance(acts, list):
        errs.append("actions must be an array")
        acts = []
    if len(acts) > MAX_ACTIONS:
        errs.append(f"actions has {len(acts)} items (max {MAX_ACTIONS})")
    for i, a in enumerate(acts):
        e, _ = _check_segment(a, dur, f"actions[{i}]")
        errs += e
        if not isinstance(a, dict):
            continue
        for k in (ACTION_VERB, ACTION_NOUN):
            if not isinstance(a.get(k), str) or not a.get(k):
                errs.append(f"actions[{i}].{k} must be a non-empty string")
        if isinstance(a.get(ACTION_VERB), str) and isinstance(a.get(ACTION_NOUN), str) \
                and not classes.allows_action(a[ACTION_VERB], a[ACTION_NOUN]):
            errs.append(f"actions[{i}] unsupported verb/noun {a[ACTION_VERB]}/{a[ACTION_NOUN]}")
        sc = a.get(ACTION_SCORE)
        if not _is_num(sc) or not (SCORE_RANGE[0] <= sc <= SCORE_RANGE[1]):
            errs.append(f"actions[{i}].score must be a number in [0, 1]")

    hls = obj[KEY_HIGHLIGHTS]
    if not isinstance(hls, list) or not hls:
        return errs + ["highlights must be a non-empty array"]
    prev_end, longest = -math.inf, 0.0
    for i, h in enumerate(hls):
        e, seg = _check_segment(h, dur, f"highlights[{i}]")
        errs += e
        if seg is None or e:
            continue
        if seg[0] < prev_end:
            errs.append(f"highlights[{i}] overlaps the previous segment or is out of order")
        prev_end = seg[1]
        longest = max(longest, seg[1] - seg[0])
    if longest < MIN_HIGHLIGHT_LEN_S - 1e-9:
        errs.append(f"no highlight segment is >= {MIN_HIGHLIGHT_LEN_S}s")
    return errs


def output_warnings(obj: dict) -> list[str]:
    """Non-fatal issues the kit's validator warns about (assumes check_output passed)."""
    warns = []
    dur = obj[KEY_DURATION]
    total = 0.0
    for i, h in enumerate(obj[KEY_HIGHLIGHTS]):
        s, e = h[SEGMENT]
        total += e - s
        if e - s < MIN_HIGHLIGHT_LEN_S:
            warns.append(f"highlights[{i}] is {e - s:.2f}s (< {MIN_HIGHLIGHT_LEN_S}s, ignored in scoring)")
    if total > highlight_budget_s(dur) + 1e-6:
        warns.append(f"highlights total {total:.2f}s exceeds 15% budget ({highlight_budget_s(dur):.2f}s)")
    return warns


def write_json(pred: Prediction, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(pred.to_json(), indent=1), encoding="utf-8")
    tmp.replace(path)  # atomic: never leave a half-written JSON


# ---------------------------------------------------------------------------
# Highlight post-processing
# ---------------------------------------------------------------------------
def _floor(x: float) -> float:
    f = 10 ** TIME_DECIMALS
    return math.floor(x * f + 1e-9) / f


def _ceil(x: float) -> float:
    f = 10 ** TIME_DECIMALS
    return math.ceil(x * f - 1e-9) / f


def sanitize_highlights(hls: Iterable[Highlight], duration: float) -> list[Highlight]:
    """Make highlights kit-valid: clip to [0, duration], round to 2 decimals (inwards), sort,
    merge overlapping or touching segments, drop segments shorter than 4.0 s, and trim to the
    15% budget without cutting any segment below 4.0 s. If the budget is smaller than one
    4.0 s segment (videos under ~26.7 s), keep exactly one 4.0 s segment on the most
    important input: at least one segment >= 4.0 s is mandatory, the budget is a soft penalty.
    """
    if duration <= 0:
        return []
    lo, hi = SCORE_RANGE
    items = []
    for h in hls:
        s, e = _ceil(max(0.0, h.start)), _floor(min(duration, h.end))
        if e > s:
            items.append(Highlight(s, e, min(hi, max(lo, h.importance))))
    items.sort(key=lambda h: (h.start, h.end))

    merged: list[Highlight] = []
    for h in items:
        if merged and h.start <= merged[-1].end:          # overlapping or touching
            m = merged[-1]
            m.end = max(m.end, h.end)
            m.importance = max(m.importance, h.importance)
        else:
            merged.append(Highlight(h.start, h.end, h.importance))

    budget = highlight_budget_s(duration)
    if budget < MIN_HIGHLIGHT_LEN_S or not any(h.length >= MIN_HIGHLIGHT_LEN_S for h in merged):
        return [_single_min_segment(merged, duration)]

    keep = [h for h in merged if h.length >= MIN_HIGHLIGHT_LEN_S]
    # spend the budget on the most important segments first, then restore time order
    out, used = [], 0.0
    for h in sorted(keep, key=lambda h: -h.importance):
        room = _floor(budget - used)
        if room < MIN_HIGHLIGHT_LEN_S:
            break
        end = h.end if h.length <= room else _floor(h.start + room)
        out.append(Highlight(h.start, end, h.importance))
        used += end - h.start
    return sorted(out, key=lambda h: h.start)


def _single_min_segment(cands: list[Highlight], duration: float) -> Highlight:
    """One MIN_HIGHLIGHT_LEN_S segment centred on the most important candidate."""
    length = min(MIN_HIGHLIGHT_LEN_S, _floor(duration))
    best = max(cands, key=lambda h: h.importance) if cands else None
    centre = (best.start + best.end) / 2 if best else duration / 2
    s = _ceil(min(max(0.0, centre - length / 2), duration - length))
    e = round(s + length, TIME_DECIMALS)
    if e > duration:  # rounding guard
        s, e = _floor(duration - length), _floor(duration)
    return Highlight(s, e, best.importance if best else 0.5)
