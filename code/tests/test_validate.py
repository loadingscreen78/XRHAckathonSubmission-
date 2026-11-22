"""The root validate.py and camspector.contract must agree."""
import importlib.util
import json

from camspector import contract as C
from camspector.baseline import baseline_prediction

from conftest import ROOT

_spec = importlib.util.spec_from_file_location("validate", ROOT / "validate.py")
V = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V)

MANIFEST = V.load_manifest(C.MANIFEST_PATH)
CLASSES = V.load_classes(C.SUPPORTED_CLASSES_PATH)


def test_manifest_loaded():
    assert len(MANIFEST) == 187
    assert C.EXCLUDED_VIDEO_IDS <= set(MANIFEST)


def test_baseline_outputs_pass_validator_for_every_manifest_video(tmp_path):
    for vid, dur in MANIFEST.items():
        if vid in C.EXCLUDED_VIDEO_IDS:
            continue
        C.write_json(baseline_prediction(vid, dur), tmp_path / f"{vid}.json")
    assert V.main(["--dir", str(tmp_path)]) == 0


def test_validator_flags_problems(tmp_path):
    good = baseline_prediction("P02_12", MANIFEST["P02_12"]).to_json()
    bad = dict(good, video_id="P02_13")                      # wrong id for file name
    (tmp_path / "P02_12.json").write_text(json.dumps(bad))
    _, errs, _ = V.validate_file(tmp_path / "P02_12.json", MANIFEST, CLASSES)
    assert any("does not match file name" in e for e in errs)
    wrong_dur = dict(good, duration_sec=good["duration_sec"] + 1.0)
    errs, _ = V.validate_obj(wrong_dur, "P02_12", MANIFEST, CLASSES)
    assert any("manifest" in e for e in errs)
    skiing = dict(good, label={"name": "skiing", "score": 0.5})
    errs, _ = V.validate_obj(skiing, "P02_12", MANIFEST, CLASSES)
    assert any("supported_classes" in e for e in errs)


def test_validator_missing_files_fail(tmp_path):
    C.write_json(baseline_prediction("P02_12", MANIFEST["P02_12"]), tmp_path / "P02_12.json")
    assert V.main(["--dir", str(tmp_path)]) == 1
    assert V.main(["--dir", str(tmp_path), "--manifest", ""]) == 0
