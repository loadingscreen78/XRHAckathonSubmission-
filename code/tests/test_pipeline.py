import json
import os
import shutil
import subprocess
import sys

import pytest

from camspector import contract as C
from camspector.video_io import probe

from conftest import ROOT

def _find_bash():
    """On Windows, `bash` on PATH may be the WSL launcher (System32 / WindowsApps), which
    fails without a distro. Prefer Git Bash there; elsewhere use PATH."""
    if os.name != "nt":
        return shutil.which("bash")
    for p in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
        if os.path.exists(p):
            return p
    w = shutil.which("bash")
    if w and not any(s in w.lower() for s in ("system32", "windowsapps")):
        return w
    return None


BASH = _find_bash()


def _run(video, out, cwd):
    env = dict(os.environ, CAMSPECTOR_PYTHON=sys.executable)
    r = subprocess.run([BASH, str(ROOT / "run.sh"), str(video), str(out)], cwd=cwd, env=env,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout == ""  # logs to stderr only
    return json.loads(out.read_text())


@pytest.mark.skipif(BASH is None, reason="bash not available")
@pytest.mark.parametrize("name", ["cfr_15fps_2min", "vfr_30_10_30s", "truncated_tail",
                                  "corrupt_middle", "zero_bytes", "not_a_video"])
def test_run_sh_always_valid(synthetic_dir, tmp_path, name):
    src = synthetic_dir / f"{name}.mp4"
    spaced = tmp_path / "dir with spaces" / f"{name}.mp4"
    spaced.parent.mkdir()
    shutil.copy(src, spaced)
    out = tmp_path / "out dir" / "nested" / f"{name}.json"
    other_cwd = tmp_path / "elsewhere"
    other_cwd.mkdir()
    obj = _run(spaced, out, other_cwd)
    try:
        dur = probe(src).duration
    except Exception:
        dur = 0.0
    assert obj[C.KEY_VIDEO_ID] == name
    assert obj[C.KEY_TEAM_ID] == C.TEAM_ID
    assert list(obj) == list(C.TOP_LEVEL_KEYS)
    if dur > 0:
        assert C.check_output(obj, dur, C.ClassSet.load()) == []
    else:  # unreadable file: no duration, so no valid highlights can exist; JSON still written
        assert obj[C.KEY_HIGHLIGHTS] == []


@pytest.mark.skipif(BASH is None, reason="bash not available")
def test_deterministic(synthetic_dir, tmp_path):
    v = synthetic_dir / "cfr_15fps_2min.mp4"
    a = _run(v, tmp_path / "a.json", tmp_path)
    b = _run(v, tmp_path / "b.json", tmp_path)
    assert a == b
