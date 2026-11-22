import json

import numpy as np
import pytest

from camspector.video_io import iter_frames, probe


def spec(d):
    return json.loads((d / "spec.json").read_text())


@pytest.mark.parametrize("name", ["cfr_30fps_20s_audio", "cfr_25fps_20s_noaudio",
                                  "vfr_30_10_30s", "rot90_15s", "odd_res_333x197_10s",
                                  "cfr_15fps_2min", "cfr_15fps_10min"])
def test_duration_matches_pts(synthetic_dir, name):
    s = spec(synthetic_dir)[name]
    info = probe(synthetic_dir / f"{name}.mp4")
    span = s["last_t"] - s["first_t"]
    # PTS duration = last frame time + one frame duration -> within ~0.15 s of span
    assert span - 0.01 <= info.duration <= span + 0.15, (info.duration, span)
    assert info.n_frames == s["n_frames"]


def test_audio_flag(synthetic_dir):
    assert probe(synthetic_dir / "cfr_30fps_20s_audio.mp4").has_audio
    assert not probe(synthetic_dir / "cfr_25fps_20s_noaudio.mp4").has_audio


def test_vfr_detected(synthetic_dir):
    assert probe(synthetic_dir / "vfr_30_10_30s.mp4").is_vfr is True
    assert probe(synthetic_dir / "cfr_25fps_20s_noaudio.mp4").is_vfr is False


def test_vfr_sampling_uses_timestamps(synthetic_dir):
    """2 fps sampling over a VFR video must give ~2 frames/s everywhere, with true times."""
    ts, tags = [], []
    for t, img in iter_frames(synthetic_dir / "vfr_30_10_30s.mp4", target_fps=2.0):
        ts.append(t)
        tags.append(int(img[4, 4, 0]))
    assert 58 <= len(ts) <= 61
    gaps = np.diff(ts)
    assert gaps.max() < 0.5 + 0.15  # 10 fps sections -> at most one 0.1 s frame late
    # red tag encodes round(t*10)%256 (lossy codec -> allow small error)
    for t, tag in zip(ts, tags):
        assert abs(((round(t * 10) % 256) - tag + 128) % 256 - 128) <= 3


def test_odd_resolution(synthetic_dir):
    info = probe(synthetic_dir / "odd_res_333x197_10s.mp4")
    t, img = next(iter_frames(synthetic_dir / "odd_res_333x197_10s.mp4", target_fps=1))
    assert img.shape[2] == 3 and abs(img.shape[1] - 333) <= 1 and abs(img.shape[0] - 197) <= 1
    assert info.width in (332, 333, 334)


def test_rotation_metadata(synthetic_dir):
    """rot90 file is stored 160x120 with a 90-degree clockwise display rotation."""
    info = probe(synthetic_dir / "rot90_15s.mp4")
    assert info.rotation == 90
    _, img = next(iter_frames(synthetic_dir / "rot90_15s.mp4", target_fps=1))
    assert img.shape[:2] == (160, 120)  # upright: 120 wide, 160 tall
    # source top-left tag block (blue=0, background blue=40) must end up top-right
    assert img[4, -5, 2] < 15 and img[4, 4, 2] > 25


def test_truncated_tail_decodes_prefix(synthetic_dir):
    info = probe(synthetic_dir / "truncated_tail.mp4")
    assert 5 < info.duration < 30
    n = sum(1 for _ in iter_frames(synthetic_dir / "truncated_tail.mp4", target_fps=2))
    assert n >= 10


def test_corrupt_middle_does_not_crash(synthetic_dir):
    n = sum(1 for _ in iter_frames(synthetic_dir / "corrupt_middle.mp4", target_fps=2))
    assert n >= 30


@pytest.mark.parametrize("name", ["zero_bytes", "not_a_video"])
def test_unreadable_raises_cleanly(synthetic_dir, name):
    with pytest.raises(Exception):
        probe(synthetic_dir / f"{name}.mp4")
