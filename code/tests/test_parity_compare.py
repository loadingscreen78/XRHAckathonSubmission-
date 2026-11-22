import numpy as np

from scripts.parity.compare_features import compare


def _feats(n=100, d=2304, seed=0):
    return np.abs(np.random.default_rng(seed).normal(size=(n, d))).astype(np.float32)


def test_identical_passes():
    x = _feats()
    r = compare(x, x.copy())
    assert r["pass"] and r["offset"] == 0 and r["median"] > 0.999


def test_small_noise_passes_large_noise_fails():
    x = _feats()
    rng = np.random.default_rng(1)
    assert compare(x + 0.05 * rng.normal(size=x.shape).astype(np.float32), x)["pass"]
    assert not compare(_feats(seed=7), x)["pass"]  # unrelated features


def test_offset_found():
    x = _feats(120)
    r = compare(x[2:], x)  # ours starts 2 clips later
    assert r["offset"] == 2 and r["median"] > 0.999
    assert not r["count_ok"] and not r["pass"]  # 2 clips missing -> count check fails


def test_dim_mismatch():
    assert not compare(_feats(d=2048), _feats())["pass"]
