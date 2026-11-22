"""Training-free CAMSPECTOR heads on top of CLIP frame embeddings (pure numpy).

Everything here takes L2-normalised embeddings (frames: [N, D], text: [K, D]) and returns
contract objects, so it is unit-tested without torch or video files
(`code/tests/test_zeroshot.py`). `camspector.clip_encoder` produces the embeddings.

Design, from the kit + the organisers' 20 Nov email + docs/EPIC_STATS.md:
- Label (10%, FPVSum): zero-shot over descriptive prompts for the kit labels. No FPVSum
  annotation and no file name is used.
- Highlights (coverage 40% on EPIC, F1 25% on FPVSum): an EPIC action counts as captured when
  >= 50% of it lies inside the highlights, and coverage counts *distinct* verb-noun pairs.
  Actions are short (median 2 s) and repeat ~2x per video, so we spend the 15% budget on many
  short (4.5 s) windows chosen by Maximal Marginal Relevance: relevant (hand-object
  interaction on EPIC, the predicted activity on FPVSum) AND visually different from the
  windows already chosen, so they land on different objects and therefore different pairs.
- Actions (mAP 25%, EPIC): noun zero-shot per 2 s window (stride 1 s), verbs from a
  verb-given-noun prior built from the EPIC-100 *train* split (falls back to verb zero-shot).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import contract as C

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
LABEL_DESCRIPTIONS = {
    "bike": "riding a bicycle on a trail",
    "bikepolo": "playing bike polo with a mallet on a court",
    "boxing": "boxing with gloves in a ring",
    "cooking": "cooking and preparing food in a kitchen",
    "horse": "riding a horse",
    "jumping": "skydiving or jumping from a great height",
    "longboarding": "riding a longboard downhill on a road",
    "motor": "riding a motorcycle on a road",
    "parkour": "doing parkour, running and jumping over walls and rooftops",
    "plane": "flying an airplane, a view from the cockpit",
    "rockclimbing": "rock climbing up a rock wall",
    "scuba": "scuba diving underwater",
    "skate": "skateboarding at a skate park",
    "ski": "skiing down a snowy mountain slope",
    "surfing": "surfing on ocean waves",
}
LABEL_TEMPLATES = ("a first-person video of {}.", "a photo of {}.", "a GoPro view of {}.")

INTERACTION_PROMPTS = (
    "hands holding a kitchen utensil", "hands cutting food on a chopping board",
    "hands washing dishes in a sink", "hands opening a cupboard or a fridge",
    "hands pouring or stirring food in a pan", "hands picking up a plate or a cup",
)
IDLE_PROMPTS = ("an empty kitchen counter", "a blurry view of the floor", "a plain wall")

NOUN_TEMPLATES = ("a photo of a {}.", "a close-up photo of a {} in a kitchen.", "hands holding a {}.")
VERB_TEMPLATES = ("a photo of hands {} something.", "a person {} something in a kitchen.")

_NOUN_FIX = {"scotch:egg": "scotch egg"}
_VERB_FIX = {"turn-on": "turning on", "turn-off": "turning off", "turn-down": "turning down",
             "let-go": "letting go"}
_DOUBLE = {"put", "cut", "stab", "rip", "drop", "pat", "flip", "set", "scrub", "wrap", "unwrap"}


def noun_phrase(key: str) -> str:
    """EPIC noun key -> words: 'board:chopping' -> 'chopping board'."""
    if key in _NOUN_FIX:
        return _NOUN_FIX[key]
    parts = key.split(":")
    return " ".join(parts[1:] + parts[:1])


def gerund(verb: str) -> str:
    """EPIC verb key -> -ing form: 'take' -> 'taking', 'turn-on' -> 'turning on'."""
    if verb in _VERB_FIX:
        return _VERB_FIX[verb]
    if verb in _DOUBLE:
        return verb + verb[-1] + "ing"
    if verb.endswith("e") and not verb.endswith(("ee", "ye")):
        return verb[:-1] + "ing"
    return verb + "ing"


def label_prompts(labels) -> list[list[str]]:
    return [[t.format(LABEL_DESCRIPTIONS[l]) for t in LABEL_TEMPLATES] for l in labels]


def noun_prompts(nouns) -> list[list[str]]:
    return [[t.format(noun_phrase(n)) for t in NOUN_TEMPLATES] for n in nouns]


def verb_prompts(verbs) -> list[list[str]]:
    return [[t.format(gerund(v)) for t in VERB_TEMPLATES] for v in verbs]


# ---------------------------------------------------------------------------
# Small numpy helpers
# ---------------------------------------------------------------------------
def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


LOGIT_SCALE = 100.0  # CLIP's learned temperature


@dataclass
class Window:
    start: float
    end: float
    emb: np.ndarray
    rel: float


def windows(times: np.ndarray, emb: np.ndarray, duration: float, length: float,
            stride: float, rel: np.ndarray | None = None) -> list[Window]:
    """Sliding windows over [0, duration]; each gets the mean embedding of its frames."""
    times = np.asarray(times, dtype=np.float64)
    out: list[Window] = []
    if duration <= 0 or len(times) == 0:
        return out
    length = min(length, duration)
    n = max(1, int(math.floor((duration - length) / stride + 1e-9)) + 1)
    for i in range(n):
        s = i * stride
        e = s + length
        m = (times >= s) & (times < e)
        if not m.any():  # sparse sampling: use the nearest frame
            m = np.zeros(len(times), bool)
            m[int(np.argmin(np.abs(times - (s + e) / 2)))] = True
        w_emb = l2n(emb[m].mean(axis=0))
        w_rel = float(rel[m].mean()) if rel is not None else 0.0
        out.append(Window(s, e, w_emb, w_rel))
    return out


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------
def predict_label(frame_emb: np.ndarray, label_text: np.ndarray, labels: list[str]) -> tuple[str, float]:
    """Video label = argmax of the mean frame embedding against the label prompts."""
    v = l2n(frame_emb.mean(axis=0))
    p = softmax(LOGIT_SCALE * (label_text @ v))
    i = int(np.argmax(p))
    return labels[i], float(p[i])


def frame_relevance(frame_emb: np.ndarray, positive: np.ndarray, negative: np.ndarray | None = None) -> np.ndarray:
    """Per-frame relevance: mean similarity to positive prompts minus negative prompts."""
    r = (frame_emb @ positive.T).mean(axis=1)
    if negative is not None and len(negative):
        r = r - (frame_emb @ negative.T).mean(axis=1)
    return r


def select_highlights(times, frame_emb, duration: float, rel=None, length: float = 4.5,
                      stride: float = 1.0, alpha: float = 0.5,
                      frac: float = C.HIGHLIGHT_TARGET_FRAC) -> list[C.Highlight]:
    """Maximal Marginal Relevance over sliding windows, under the 15% budget."""
    if duration <= 0:
        return []
    budget = frac * duration
    n = int(budget // length)
    ws = windows(times, frame_emb, duration, length, stride, rel)
    if not ws:
        return []
    r = np.array([w.rel for w in ws])
    r = (r - r.min()) / (r.max() - r.min()) if r.max() > r.min() else np.full(len(ws), 0.5)
    if n < 1:  # budget below one minimum segment: one 4.0 s segment on the best window
        best = ws[int(np.argmax(r))]
        c = (best.start + best.end) / 2
        return C.sanitize_highlights([C.Highlight(c - 2.0, c + 2.0, 1.0)], duration)
    E = np.stack([w.emb for w in ws])
    chosen: list[int] = []
    max_sim = np.full(len(ws), -np.inf)
    gap = 1.0  # keep chosen windows apart so they never touch after rounding
    blocked = np.zeros(len(ws), bool)
    for _ in range(n):
        div = np.where(np.isfinite(max_sim), max_sim, 0.0)
        gain = alpha * r - (1 - alpha) * div
        gain[blocked] = -np.inf
        i = int(np.argmax(gain))
        if not np.isfinite(gain[i]):
            break
        chosen.append(i)
        max_sim = np.maximum(max_sim, E @ E[i])
        s, e = ws[i].start, ws[i].end
        for j, w in enumerate(ws):
            if w.start < e + gap and w.end > s - gap:
                blocked[j] = True
    hls = [C.Highlight(ws[i].start, ws[i].end, float(0.5 + 0.5 * r[i])) for i in chosen]
    return C.sanitize_highlights(hls, duration)


def predict_actions(times, frame_emb, duration: float, noun_text: np.ndarray, nouns: list[str],
                    verb_text: np.ndarray | None = None, verbs: list[str] | None = None,
                    verb_prior: dict[str, dict[str, float]] | None = None,
                    length: float = 2.0, stride: float = 1.0, top_nouns: int = 3,
                    top_verbs: int = 2, max_actions: int = C.MAX_ACTIONS) -> list[C.Action]:
    """Per-window noun zero-shot x verbs (train prior, else verb zero-shot); merge runs."""
    ws = windows(times, frame_emb, duration, length, stride)
    if not ws:
        return []
    E = np.stack([w.emb for w in ws])
    pn = softmax(LOGIT_SCALE * (E @ noun_text.T))                     # [W, Nn]
    pv = softmax(LOGIT_SCALE * (E @ verb_text.T)) if verb_text is not None else None
    cands: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for wi in range(len(ws)):
        for ni in np.argsort(-pn[wi])[:top_nouns]:
            noun = nouns[ni]
            if verb_prior and verb_prior.get(noun):
                vs = sorted(verb_prior[noun].items(), key=lambda kv: -kv[1])[:top_verbs]
            elif pv is not None and verbs:
                vs = [(verbs[vi], float(pv[wi, vi])) for vi in np.argsort(-pv[wi])[:top_verbs]]
            else:
                continue
            for verb, p_v in vs:
                cands.setdefault((verb, noun), []).append((wi, float(pn[wi, ni]) * float(p_v)))
    acts: list[C.Action] = []
    for (verb, noun), hits in cands.items():
        hits.sort()
        run = [hits[0]]
        for h in hits[1:] + [None]:
            if h is not None and h[0] == run[-1][0] + 1:   # consecutive windows -> one segment
                run.append(h)
                continue
            s, e = ws[run[0][0]].start, ws[run[-1][0]].end
            acts.append(C.Action(verb, noun, s, min(e, duration), max(sc for _, sc in run)))
            if h is not None:
                run = [h]
    acts.sort(key=lambda a: -(a.score or 0.0))
    return [a for a in acts[:max_actions] if a.end - a.start > 0]
