"""Fallback outputs. Needs no model and cannot fail for a valid duration.

Used whenever the model path is unavailable or fails, so run.sh always writes a
kit-valid JSON. It carries no action predictions and a default label.
"""
from __future__ import annotations

from . import contract as C


WINDOW_S = 4.5  # just above the 4.0 s minimum, so the budget buys as many windows as possible


def even_highlights(duration: float, n: int | None = None,
                    frac: float = C.HIGHLIGHT_TARGET_FRAC) -> list[C.Highlight]:
    """Evenly spaced ~4.5 s windows totalling `frac` of duration (each >= 4.0 s).

    Coverage counts distinct verb-noun pairs with >= 50% of an action inside the highlights;
    EPIC actions are short (median 2 s) and repeat, so many short, spread-out windows reach
    more distinct pairs than a few long ones (docs/EPIC_STATS.md, organisers' email)."""
    if duration <= 0:
        return []
    total = frac * duration
    n = n or max(1, int(total // WINDOW_S))
    seg = max(total / n, C.MIN_HIGHLIGHT_LEN_S)
    seg = min(seg, duration)
    gap = max(0.0, (duration - n * seg) / n)
    out = []
    for i in range(n):
        s = gap / 2 + i * (seg + gap)
        out.append(C.Highlight(s, s + seg, 0.5))
    return C.sanitize_highlights(out, duration)


def baseline_prediction(video_id: str, duration: float,
                        label: str = C.DEFAULT_LABEL,
                        label_score: float = C.DEFAULT_LABEL_SCORE) -> C.Prediction:
    return C.Prediction(
        video_id=video_id,
        duration=duration,
        label=label,
        label_score=label_score,
        actions=[],            # allowed by the kit (0-2000 items); scores 0 on action mAP
        highlights=even_highlights(duration),
    )
