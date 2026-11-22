"""Generate synthetic edge-case videos with PyAV for video_io tests.

Usage: python code/scripts/make_synthetic_videos.py [out_dir]   (default: data/synthetic)

Each frame encodes its own timestamp in the red channel of the top-left block
(value = round(t*10) % 256) so tests can check timestamp correctness.
"""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


def _codec(container) -> str:
    for name in ("libx264", "h264", "libopenh264", "mpeg4"):
        try:
            av.codec.Codec(name, "w")
            return name
        except Exception:
            continue
    raise RuntimeError("no usable video encoder in this PyAV build")


def _frame(t: float, w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 2] = 40
    x = int((t * 20) % max(1, w - 16))
    img[h // 2 - 8:h // 2 + 8, x:x + 16] = (255, 255, 0)
    img[:16, :16, 0] = int(round(t * 10)) % 256  # timestamp tag
    img[:16, :16, 1:] = 0
    return img


def write_video(path: Path, timestamps: list[float], w: int, h: int, fps_hint: int = 30,
                audio: bool = False, rotate: int | None = None, faststart: bool = True) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = {"movflags": "+faststart"} if faststart else {}
    with av.open(str(path), "w", options=opts) as c:
        codec = _codec(c)
        vs = c.add_stream(codec, rate=fps_hint)
        vs.width, vs.height = w, h
        vs.pix_fmt = "yuv420p" if (w % 2 == 0 and h % 2 == 0) else "yuv444p"
        vs.time_base = Fraction(1, 1000)
        vs.codec_context.time_base = Fraction(1, 1000)
        if codec in ("libx264", "h264"):
            vs.options = {"preset": "ultrafast", "crf": "30"}
        if rotate is not None:  # `rotate` = clockwise display rotation (legacy tag sense)
            if hasattr(vs, "set_display_rotation"):  # PyAV >= 17: display matrix, CCW
                vs.set_display_rotation(-rotate)
            else:
                vs.metadata["rotate"] = str(rotate)
        a_s = None
        if audio:
            a_s = c.add_stream("aac", rate=16000)
        for t in timestamps:
            fr = av.VideoFrame.from_ndarray(_frame(t, w, h), format="rgb24")
            fr.pts = int(round(t * 1000))
            fr.time_base = Fraction(1, 1000)
            for p in vs.encode(fr):
                c.mux(p)
        for p in vs.encode():
            c.mux(p)
        if a_s is not None:
            dur = timestamps[-1]
            n = 1024
            for i in range(int(dur * 16000 / n)):
                af = av.AudioFrame.from_ndarray(np.zeros((1, n), np.float32), format="fltp",
                                                layout="mono")
                af.sample_rate = 16000
                af.pts = i * n
                for p in a_s.encode(af):
                    c.mux(p)
            for p in a_s.encode():
                c.mux(p)
    return {"codec": codec}


def cfr(dur: float, fps: float) -> list[float]:
    return [i / fps for i in range(int(dur * fps))]


def vfr(dur: float, seed: int = 0) -> list[float]:
    """Alternate 30 fps and 10 fps sections every 3 s, plus jitter."""
    rng = np.random.default_rng(seed)
    t, out = 0.0, []
    while t < dur:
        out.append(round(t, 3))
        base = 1 / 30 if int(t // 3) % 2 == 0 else 1 / 10
        t += base * rng.uniform(0.8, 1.2)
    return out


def truncate(src: Path, dst: Path, keep: float = 0.6) -> None:
    b = src.read_bytes()
    dst.write_bytes(b[: int(len(b) * keep)])


def corrupt_middle(src: Path, dst: Path, frac: float = 0.5, n: int = 4096) -> None:
    b = bytearray(src.read_bytes())
    i = int(len(b) * frac)
    rng = np.random.default_rng(1)
    b[i:i + n] = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
    dst.write_bytes(bytes(b))


def main(out_dir: str = "data/synthetic") -> None:
    out = Path(out_dir)
    spec = {}
    def mk(name, ts, w=160, h=120, **kw):
        info = write_video(out / f"{name}.mp4", ts, w, h, **kw)
        spec[name] = {"expected_duration": None, "first_t": ts[0], "last_t": ts[-1],
                      "n_frames": len(ts), "width": w, "height": h, **kw, **info}
        print("wrote", name, file=sys.stderr)

    mk("cfr_30fps_20s_audio", cfr(20, 30), audio=True)
    mk("cfr_25fps_20s_noaudio", cfr(20, 25))
    mk("vfr_30_10_30s", vfr(30))
    mk("rot90_15s", cfr(15, 30), rotate=90)
    mk("odd_res_333x197_10s", cfr(10, 24), w=333, h=197)
    mk("cfr_15fps_2min", cfr(120, 15))
    mk("cfr_15fps_10min", cfr(600, 15))
    mk("base_for_corrupt_30s", cfr(30, 30))
    truncate(out / "base_for_corrupt_30s.mp4", out / "truncated_tail.mp4", 0.6)
    corrupt_middle(out / "base_for_corrupt_30s.mp4", out / "corrupt_middle.mp4")
    (out / "zero_bytes.mp4").write_bytes(b"")
    (out / "not_a_video.mp4").write_bytes(b"hello, this is not a video" * 100)
    (out / "spec.json").write_text(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
