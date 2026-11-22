"""Fast frame sampling: decode keyframes only when they are dense enough.

Full-HD EPIC videos are 60 fps; decoding every frame to keep 1-2 per second wastes most of
the time. Keyframes (I-frames) decode on their own, so asking FFmpeg to skip the rest
(skip_frame = NONKEY) cuts decoding by roughly an order of magnitude. If a file's keyframes
are too sparse (common on YouTube uploads), or anything goes wrong, this falls back to the
normal PTS-based full decode (`video_io.iter_frames`). Either way the result is deterministic.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .video_io import VideoInfo, _decode_tolerant, _upright, iter_frames, probe

log = logging.getLogger("camspector.fastframes")

MIN_KEYFRAME_RATE = 0.5  # keyframes per second needed to trust the fast path


def sample_frames(path: str | Path, target_fps: float, size: tuple[int, int],
                  info: VideoInfo | None = None) -> tuple[list[float], list[np.ndarray]]:
    """Return (timestamps_s, RGB frames) at <= target_fps."""
    info = info or probe(path, scan=False)
    try:
        times, frames = _keyframes(path, target_fps, size, info)
        span = max(float(info.duration or 0.0), times[-1] if times else 0.0)
        if len(times) >= 2 and span > 0 and len(times) / span >= MIN_KEYFRAME_RATE:
            log.info("keyframe sampling: %d frames over %.0f s", len(times), span)
            return times, frames
        log.info("keyframes too sparse (%d over %.0f s): full decode", len(times), span)
    except Exception as e:  # any PyAV/codec quirk: use the safe path
        log.warning("keyframe decode failed (%s): full decode", e)
    times, frames = [], []
    for t, img in iter_frames(path, target_fps=target_fps, size=size, info=info):
        times.append(t)
        frames.append(img)
    return times, frames


def _keyframes(path, target_fps: float, size: tuple[int, int], info: VideoInfo):
    import av  # noqa: F401  (PyAV is imported by video_io as well)

    step = 1.0 / target_fps
    times: list[float] = []
    frames: list[np.ndarray] = []
    next_t, t0 = 0.0, None
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        vs.codec_context.skip_frame = "NONKEY"
        for frame in _decode_tolerant(c, vs, str(path)):
            if frame.pts is None:
                continue
            t = float(frame.pts * vs.time_base)
            if t0 is None:
                t0 = t
            t -= t0
            if t + 1e-6 < next_t:
                continue
            next_t = t + step * 0.999
            img = frame.to_ndarray(format="rgb24", width=size[0], height=size[1])
            times.append(t)
            frames.append(_upright(img, info.rotation))
    return times, frames
