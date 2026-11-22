import copy

import pytest

from camspector import contract as C
from camspector.baseline import baseline_prediction, even_highlights


def _valid():
    return {
        "team_id": "SpectraCore",
        "video_id": "P02_109",
        "duration_sec": 100.0,
        "label": {"name": "cooking", "score": 0.96},
        "actions": [{"segment": [1.0, 2.0], "verb": "open", "noun": "fridge", "score": 0.8}],
        "highlights": [{"segment": [10.0, 15.0]}, {"segment": [20.0, 29.0]}],
    }


def test_valid_passes():
    assert C.check_output(_valid(), 100.0) == []
    assert C.output_warnings(_valid()) == []


@pytest.mark.parametrize("mutate,needle", [
    (lambda o: o.pop("team_id"), "missing"),
    (lambda o: o.pop("duration_sec"), "missing"),
    (lambda o: o.__setitem__("extra", 1), "unexpected"),
    (lambda o: o.__setitem__("team_id", ""), "team_id"),
    (lambda o: o.__setitem__("duration_sec", 0), "positive"),
    (lambda o: o.__setitem__("duration_sec", 101.0), "within"),
    (lambda o: o.__setitem__("label", "cooking"), "label must be"),
    (lambda o: o["label"].__setitem__("score", 1.5), "label.score"),
    (lambda o: o["highlights"][0].__setitem__("segment", [15.0, 10.0]), "start must be < end"),
    (lambda o: o["highlights"][1].__setitem__("segment", [12.0, 29.0]), "overlaps"),
    (lambda o: o["highlights"].reverse(), "out of order"),
    (lambda o: o.__setitem__("highlights", []), "non-empty"),
    (lambda o: o.__setitem__("highlights", [{"segment": [10.0, 13.0]}]), ">= 4.0s"),
    (lambda o: o["highlights"][1].__setitem__("segment", [20.0, 100.6]), "duration_sec + 0.5"),
    (lambda o: o["actions"][0].__setitem__("segment", [-1.0, 2.0]), "< 0"),
    (lambda o: o["actions"][0].__setitem__("segment", [1.0, float("nan")]), "finite"),
    (lambda o: o["actions"][0].__setitem__("verb", 3), "verb"),
    (lambda o: o["actions"][0].__setitem__("score", True), "score"),
    (lambda o: o["actions"][0].pop("score"), "score"),
])
def test_violations_detected(mutate, needle):
    o = copy.deepcopy(_valid())
    mutate(o)
    errs = C.check_output(o, 100.0)
    assert any(needle in e for e in errs), errs


def test_end_tolerance_allowed():
    o = _valid()
    o["highlights"][1]["segment"] = [96.0, 100.4]
    assert C.check_output(o, 100.0) == []


def test_supported_classes_enforced():
    cs = C.ClassSet.load(C.SUPPORTED_CLASSES_PATH)
    assert "ski" in cs.labels and "skiing" not in cs.labels  # organisers: use supported_classes
    assert C.check_output(_valid(), 100.0, cs) == []
    o = _valid()
    o["actions"][0]["noun"] = "robbery"
    assert any("unsupported" in e for e in C.check_output(o, 100.0, cs))
    o = _valid()
    o["label"]["name"] = "skiing"
    assert any("unsupported label" in e for e in C.check_output(o, 100.0, cs))


@pytest.mark.parametrize("duration", [11.13, 15.28, 25.16, 26.7, 29.51, 37.34, 120.0,
                                      333.33, 600.0, 1968.6])
def test_baseline_always_valid(duration):
    p = baseline_prediction("vid", duration)
    obj = p.to_json()
    assert C.check_output(obj, duration, C.ClassSet.load()) == []
    assert obj["team_id"] == "SpectraCore"
    assert all(h["segment"][1] - h["segment"][0] >= C.MIN_HIGHLIGHT_LEN_S - 1e-9
               for h in obj["highlights"])
    total = sum(h["segment"][1] - h["segment"][0] for h in obj["highlights"])
    if duration * C.HIGHLIGHT_BUDGET_FRAC >= C.MIN_HIGHLIGHT_LEN_S:
        assert total <= C.highlight_budget_s(duration) + 1e-9
    else:  # shorter than ~26.7 s: one 4.0 s segment, accepted soft penalty
        assert len(obj["highlights"]) == 1 and abs(total - 4.0) < 1e-9
    if duration >= 60:
        assert total >= 0.14 * duration


def test_sanitize_clips_merges_and_trims():
    d = 100.0
    raw = [C.Highlight(-5, 5, 1.0), C.Highlight(3, 8, 0.4), C.Highlight(50, 90, 0.1),
           C.Highlight(95, 120, 0.2), C.Highlight(30, 32, 0.9)]
    out = C.sanitize_highlights(raw, d)
    obj = C.Prediction("x", d, highlights=out).to_json()
    assert C.check_output(obj, d) == []
    assert C.output_warnings(obj) == []
    assert out[0].start == 0 and out[0].end == 8           # merged, kept first (most important)
    assert all(h.length >= 4.0 for h in out)               # 2 s segment dropped
    assert sum(h.length for h in out) <= 15.0 + 1e-9


def test_even_highlights_sorted_nonoverlapping():
    hs = even_highlights(480.0)
    assert all(a.end < b.start for a, b in zip(hs, hs[1:]))


def test_two_decimals():
    obj = baseline_prediction("v", 123.4567).to_json()
    for h in obj["highlights"]:
        for t in h["segment"]:
            assert round(t, 2) == t
    assert obj["duration_sec"] == 123.46


def test_video_id_rule():
    assert C.video_id_from_path("/x/y z/P01_01.MP4") == "P01_01"
    assert C.video_id_from_path("Ski_-Z5a4UQmZLs.mp4") == "Ski_-Z5a4UQmZLs"
