"""Zero-shot heads on synthetic embeddings (no torch, no video needed)."""
import json

import numpy as np
import pytest

from camspector import contract as C
from camspector import zeroshot as Z
from camspector.baseline import baseline_prediction

RNG = np.random.default_rng(0)
D = 64


def basis(k):
    """k orthonormal 'concept' directions."""
    q, _ = np.linalg.qr(RNG.normal(size=(D, k)))
    return q.T.astype(np.float32)


def noisy(v, n, s=0.05):
    return Z.l2n(v[None, :] + s * RNG.normal(size=(n, D)))


def test_prompt_wording():
    assert Z.noun_phrase("board:chopping") == "chopping board"
    assert Z.noun_phrase("machine:sous:vide") == "sous vide machine"
    assert Z.noun_phrase("scotch:egg") == "scotch egg"
    assert [Z.gerund(v) for v in ("take", "put", "cut", "turn-on", "dry", "squeeze", "open")] == \
        ["taking", "putting", "cutting", "turning on", "drying", "squeezing", "opening"]
    cs = C.ClassSet.load()
    assert set(Z.LABEL_DESCRIPTIONS) == cs.labels - {"other"}
    assert all(len(p) == len(Z.NOUN_TEMPLATES) for p in Z.noun_prompts(sorted(cs.nouns)))


def test_label_picks_matching_category():
    labels = ["cooking", "ski", "surfing"]
    L = basis(3)
    frames = noisy(L[1], 40)
    lab, score = Z.predict_label(frames, L, labels)
    assert lab == "ski" and 0 < score <= 1


@pytest.mark.parametrize("duration", [11.13, 25.16, 37.34, 120.0, 216.0, 600.0, 1968.6])
def test_highlights_valid_and_budgeted(duration):
    fps = 2.0 if duration <= 600 else 1.0
    t = np.arange(0, duration, 1 / fps)
    emb = noisy(basis(1)[0], len(t), 0.5)
    rel = RNG.normal(size=len(t))
    hls = Z.select_highlights(t, emb, duration, rel)
    obj = C.Prediction("v", duration, highlights=hls).to_json()
    assert C.check_output(obj, duration) == []
    total = sum(h.length for h in hls)
    if duration * C.HIGHLIGHT_BUDGET_FRAC >= C.MIN_HIGHLIGHT_LEN_S:
        assert C.output_warnings(obj) == []
        assert total >= 0.1 * duration          # most of the budget is used
    else:
        assert len(hls) == 1 and abs(total - 4.0) < 1e-6


def test_highlights_cover_every_distinct_scene():
    """5 visually distinct scenes of unequal length; MMR must visit all of them."""
    duration, fps = 200.0, 2.0
    t = np.arange(0, duration, 1 / fps)
    B = basis(5)
    bounds = [0, 100, 130, 150, 170, 200]               # one long scene, four short ones
    emb = np.concatenate([noisy(B[i], int((bounds[i + 1] - bounds[i]) * fps)) for i in range(5)])
    hls = Z.select_highlights(t, emb, duration, rel=np.zeros(len(t)))
    hit = {next(i for i in range(5) if bounds[i] <= (h.start + h.end) / 2 < bounds[i + 1]) for h in hls}
    assert hit == {0, 1, 2, 3, 4}
    # the even-spacing baseline puts most windows in the long first scene
    base = baseline_prediction("v", duration).highlights
    assert sum((h.start + h.end) / 2 < 100 for h in base) > sum((h.start + h.end) / 2 < 100 for h in hls)


def test_highlights_prefer_relevant_frames():
    duration, fps = 120.0, 2.0
    t = np.arange(0, duration, 1 / fps)
    emb = noisy(basis(1)[0], len(t), 0.01)              # visually uniform video
    rel = np.where((t >= 60) & (t < 70), 1.0, 0.0)       # interaction only in 60-70 s
    hls = Z.select_highlights(t, emb, duration, rel)
    assert any(60 <= h.start and h.end <= 70 for h in hls)


def test_actions_from_noun_zero_shot_and_prior():
    duration, fps = 30.0, 2.0
    t = np.arange(0, duration, 1 / fps)
    nouns = ["fridge", "knife", "onion", "tap"]
    N = basis(4)
    emb = np.concatenate([noisy(N[0], 20), noisy(N[2], 20), noisy(N[3], 20)])  # fridge, onion, tap
    prior = {"fridge": {"open": 0.6, "close": 0.4}, "onion": {"cut": 0.7, "peel": 0.3},
             "tap": {"turn-on": 0.5, "turn-off": 0.5}}
    acts = Z.predict_actions(t, emb, duration, N, nouns, verb_prior=prior, top_nouns=1)
    pairs = {(a.verb, a.noun) for a in acts}
    assert {("open", "fridge"), ("cut", "onion"), ("turn-on", "tap")} <= pairs
    cut = max((a for a in acts if (a.verb, a.noun) == ("cut", "onion")), key=lambda a: a.score)
    assert 8.0 <= cut.start <= 11.0 and 19.0 <= cut.end <= 22.0   # merged run ~10-20 s
    obj = C.Prediction("P01_11", duration, actions=acts, highlights=[C.Highlight(0, 4.5)]).to_json()
    assert C.check_output(obj, duration, C.ClassSet.load()) == []


def test_actions_capped_and_verb_zero_shot_fallback():
    duration, fps = 1968.6, 1.0
    t = np.arange(0, duration, 1 / fps)
    nouns, verbs = [f"n{i}" for i in range(8)], [f"v{i}" for i in range(4)]
    emb = Z.l2n(RNG.normal(size=(len(t), D)))
    acts = Z.predict_actions(t, emb, duration, basis(8), nouns, basis(4), verbs, None)
    assert 0 < len(acts) <= C.MAX_ACTIONS
    assert all(0 <= a.start < a.end <= duration and 0 <= a.score <= 1 for a in acts)
    assert all(acts[i].score >= acts[i + 1].score for i in range(len(acts) - 1))


def test_pair_prior_builder(tmp_path):
    from scripts.build_pair_prior import build
    (tmp_path / "EPIC_100_verb_classes.csv").write_text("id,key\n0,take\n1,open\n")
    (tmp_path / "EPIC_100_noun_classes.csv").write_text("id,key\n0,fridge\n1,knife\n")
    (tmp_path / "EPIC_100_train.csv").write_text(
        "verb_class,noun_class\n1,0\n1,0\n0,0\n0,1\n")
    out = build(tmp_path, top=5)
    assert out["verb_given_noun"] == {"fridge": {"open": 0.6667, "take": 0.3333}, "knife": {"take": 1.0}}
    json.dumps(out)
