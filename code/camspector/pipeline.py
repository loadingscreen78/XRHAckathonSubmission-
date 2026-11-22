"""Entry point: python -m camspector.pipeline --video X.mp4 --out X.json

Always writes a valid JSON: any failure in the model path falls back to the baseline.
Logs go to stderr only.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

from . import contract as C
from .baseline import baseline_prediction
from .video_io import probe

log = logging.getLogger("camspector")


PRIOR_PATH = Path(__file__).resolve().parents[2] / "weights" / "epic_pair_prior.json"


def _decode_size(width: int, height: int, short: int = 224) -> tuple[int, int]:
    s = short / max(1, min(width, height))
    return max(2, round(width * s / 2) * 2), max(2, round(height * s / 2) * 2)


def run_model(video: Path, duration: float) -> C.Prediction:
    """Training-free CAMSPECTOR path: CLIP frame embeddings -> label, highlights, actions.
    Any exception here makes process() fall back to the baseline."""
    if os.environ.get("CAMSPECTOR_BASELINE"):
        raise NotImplementedError("CAMSPECTOR_BASELINE set: baseline only")
    import numpy as np

    from . import zeroshot as Z
    from .clip_encoder import get_encoder
    from .fastframes import sample_frames

    if duration <= 0:
        raise ValueError("unknown duration")
    enc = get_encoder()
    info = probe(video, scan=False)
    fps = 2.0 if duration <= 600 else 1.0
    times, frames = sample_frames(video, fps, _decode_size(info.width, info.height), info)
    if len(frames) < 2:
        raise ValueError(f"only {len(frames)} decodable frames")
    E, T = enc.images(frames), np.asarray(times)
    log.info("encoded %d frames at %.1f fps", len(frames), fps)

    classes = C.ClassSet.load()
    labels = [l for l in sorted(classes.labels) if l in Z.LABEL_DESCRIPTIONS] or sorted(Z.LABEL_DESCRIPTIONS)
    label, label_score = Z.predict_label(E, enc.prompt_sets(Z.label_prompts(labels)), labels)

    if label == "cooking":
        rel = Z.frame_relevance(E, enc.texts(list(Z.INTERACTION_PROMPTS)), enc.texts(list(Z.IDLE_PROMPTS)))
    else:
        rel = Z.frame_relevance(E, enc.prompt_sets(Z.label_prompts([label])))
    highlights = Z.select_highlights(T, E, duration, rel)

    nouns = sorted(n for n in classes.nouns if n != "other")
    verbs = sorted(v for v in classes.verbs if v != "other")
    prior = None
    if PRIOR_PATH.exists():
        prior = json.loads(PRIOR_PATH.read_text(encoding="utf-8"))["verb_given_noun"]
    noun_text = enc.prompt_sets(Z.noun_prompts(nouns))
    verb_text = None if prior else enc.prompt_sets(Z.verb_prompts(verbs))
    actions = Z.predict_actions(T, E, duration, noun_text, nouns, verb_text, verbs, prior)
    actions = [a for a in actions if classes.allows_action(a.verb, a.noun)]
    log.info("label=%s (%.2f), %d highlights, %d actions, verb prior: %s",
             label, label_score, len(highlights), len(actions), bool(prior))
    return C.Prediction(video_id=C.video_id_from_path(video), duration=duration, label=label,
                        label_score=label_score, actions=actions, highlights=highlights)


def _safe_duration(video: Path) -> float:
    try:
        return probe(video, scan=True).duration
    except Exception as e:  # unreadable file: still write a schema-valid JSON
        log.error("probe failed for %s: %s", video, e)
        return 0.0


def process(video: Path, out: Path, use_model: bool = True) -> dict:
    t0 = time.perf_counter()
    vid = C.video_id_from_path(video)
    duration = _safe_duration(video)
    pred = None
    if use_model:
        try:
            pred = run_model(video, duration)
            errs = C.check_output(pred.to_json(), duration)
            if errs:
                raise ValueError(f"model output invalid: {errs[:3]}")
        except NotImplementedError as e:
            log.info("fallback: %s", e)
            pred = None
        except Exception:
            log.error("model path failed, using baseline:\n%s", traceback.format_exc())
            pred = None
    used_model = pred is not None
    if pred is None:
        pred = baseline_prediction(vid, duration)
    C.write_json(pred, out)
    dt = time.perf_counter() - t0
    log.info("wrote %s (duration=%.2fs, %d actions, %d highlights, model=%s) in %.2fs",
             out, duration, len(pred.actions), len(pred.highlights), used_model, dt)
    return {"video_id": vid, "duration": duration, "seconds": dt, "used_model": used_model}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--baseline-only", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        process(a.video, a.out, use_model=not a.baseline_only)
    except Exception:  # last resort: still emit a valid JSON
        log.critical("unexpected failure:\n%s", traceback.format_exc())
        C.write_json(baseline_prediction(C.video_id_from_path(a.video), 0.0), a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
