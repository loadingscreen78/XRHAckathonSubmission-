"""Video decoding with presentation timestamps (PTS). Never assumes a frame rate.

- duration: from container/stream metadata AND from decoded frame PTS (VFR-safe)
- rotation: read from stream side data / metadata; frames are returned upright
- corrupt frames: skipped and logged; decoding stops cleanly at a truncated tail
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np

log = logging.getLogger("camspector.video_io")


@dataclass
class VideoInfo:
    path: str
    duration: float            # best estimate, seconds (used for the contract)
    meta_duration: float | None  # from container/stream metadata
    pts_duration: float | None   # last frame PTS + frame duration (needs full scan)
    avg_fps: float | None
    n_frames: int | None
    width: int
    height: int
    rotation: int              # degrees, multiple of 90
    codec: str
    has_audio: bool
    is_vfr: bool | None        # None until a scan was done
    n_decode_errors: int = 0


def _rotation(container, stream) -> int:
    """Clockwise display rotation in degrees (0/90/180/270).

    Modern FFmpeg stores rotation as a display matrix; PyAV >= 14 exposes it per decoded
    frame as `frame.rotation` (counter-clockwise degrees). Older files may only carry the
    legacy clockwise `rotate` metadata tag. Read the first decodable frame, then the tag."""
    ccw = 0
    try:
        for frame in _decode_tolerant(container, stream, "<probe>", max_errors=5):
            ccw = int(round(getattr(frame, "rotation", 0) or 0))
            break
    except Exception:  # pragma: no cover - library-version dependent
        ccw = 0
    if ccw:
        return (-ccw) % 360
    r = (stream.metadata or {}).get("rotate")
    try:
        return int(r) % 360 if r is not None else 0
    except ValueError:
        return 0


def _meta_duration(container, stream) -> float | None:
    if stream.duration is not None and stream.time_base is not None:
        d = float(stream.duration * stream.time_base)
        if d > 0:
            return d
    if container.duration is not None and container.duration > 0:
        return container.duration / av.time_base
    return None


def probe(path: str | Path, scan: bool = True) -> VideoInfo:
    """Read metadata; if scan=True, also walk packets for exact PTS duration and VFR."""
    path = str(path)
    with av.open(path) as c:
        vs = c.streams.video[0]
        info = VideoInfo(
            path=path,
            duration=0.0,
            meta_duration=_meta_duration(c, vs),
            pts_duration=None,
            avg_fps=float(vs.average_rate) if vs.average_rate else None,
            n_frames=vs.frames or None,
            width=vs.codec_context.width,
            height=vs.codec_context.height,
            rotation=0,
            codec=vs.codec_context.name,
            has_audio=len(c.streams.audio) > 0,
            is_vfr=None,
        )
    with av.open(path) as c:  # fresh handle: rotation probe decodes one frame
        info.rotation = _rotation(c, c.streams.video[0])
    if scan:
        with av.open(path) as c:
            _scan_packets(c, c.streams.video[0], info)
    cands = [d for d in (info.pts_duration, info.meta_duration) if d and d > 0]
    # Prefer PTS (what is actually decodable, e.g. truncated files); fall back to metadata.
    info.duration = info.pts_duration if info.pts_duration else (min(cands) if cands else 0.0)
    return info


def _scan_packets(c, vs, info: VideoInfo) -> None:
    """Packet-level scan (no decoding): fast, gives PTS span and VFR detection."""
    tb = float(vs.time_base)
    pts: list[int] = []
    last_dur = 0
    try:
        for pkt in c.demux(vs):
            if pkt.pts is None or pkt.size == 0:
                continue
            pts.append(pkt.pts)
            if pkt.duration:
                last_dur = pkt.duration
    except av.error.FFmpegError as e:  # truncated tail
        info.n_decode_errors += 1
        log.warning("demux error in %s: %s", info.path, e)
    if not pts:
        return
    arr = np.unique(np.asarray(pts, dtype=np.int64))
    start = arr[0]
    d = np.diff(arr)
    frame_dur = last_dur or (int(np.median(d)) if len(d) else 0)
    info.pts_duration = float((arr[-1] - start + frame_dur) * tb)
    info.n_frames = int(len(arr))
    if len(d) > 2:
        med = float(np.median(d))
        info.is_vfr = bool(np.mean(np.abs(d - med) > 0.05 * med) > 0.01)
    else:
        info.is_vfr = False


def _upright(img: np.ndarray, rotation: int) -> np.ndarray:
    k = {0: 0, 90: 3, 180: 2, 270: 1}.get(rotation, 0)  # clockwise display rotation
    return np.ascontiguousarray(np.rot90(img, k)) if k else img


def _decode_tolerant(c, vs, path: str, max_errors: int = 200):
    """Decode packet by packet so one corrupt packet doesn't end the stream
    (PyAV's container.decode() generator dies on the first exception)."""
    errors = 0
    demux = c.demux(vs)
    while True:
        try:
            pkt = next(demux)
        except StopIteration:
            break
        except av.error.FFmpegError as e:  # truncated / unreadable container tail
            log.warning("demux error in %s: %s", path, e)
            break
        try:
            frames = pkt.decode()  # pkt with size 0 flushes the decoder at EOF
        except av.error.FFmpegError as e:
            errors += 1
            if errors <= 5 or errors % 50 == 0:
                log.warning("decode error #%d in %s: %s", errors, path, e)
            if errors > max_errors:
                log.error("too many decode errors in %s, stopping", path)
                break
            continue
        yield from frames


def iter_frames(path: str | Path, target_fps: float | None = None,
                size: tuple[int, int] | None = None,
                info: VideoInfo | None = None) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_s, RGB uint8 HxWx3) using PTS. Samples to target_fps by time,
    not by frame index, so it is correct for VFR. Corrupt frames are skipped."""
    info = info or probe(path, scan=False)
    step = 1.0 / target_fps if target_fps else 0.0
    next_t = 0.0
    with av.open(str(path)) as c:
        vs = c.streams.video[0]
        vs.thread_type = "AUTO"
        t0 = None
        for frame in _decode_tolerant(c, vs, str(path)):
            if frame.pts is None:
                continue
            t = float(frame.pts * vs.time_base)
            if t0 is None:
                t0 = t
            t -= t0
            if step and t + 1e-6 < next_t:
                continue
            if step:
                next_t = (math.floor(t / step + 1e-6) + 1) * step
            kw = {"format": "rgb24"}
            if size is not None:
                kw.update(width=size[0], height=size[1])
            img = frame.to_ndarray(**kw)
            yield t, _upright(img, info.rotation)
