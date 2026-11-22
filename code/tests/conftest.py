import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
SYN = ROOT / "data" / "synthetic"


@pytest.fixture(scope="session")
def synthetic_dir() -> Path:
    if not (SYN / "spec.json").exists():
        from scripts.make_synthetic_videos import main  # noqa: E402
        main(str(SYN))
    return SYN
