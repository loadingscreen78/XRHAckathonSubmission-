#!/usr/bin/env bash
# CAMSPECTOR: one-command setup + GPU benchmark for Linux / WSL / Kaggle / Colab.
# Usage:  bash setup_and_bench.sh            (env vars: PYTHON=python3.11 CUDA=cu126 SKIP_INSTALL=1)
# Result: results_<hostname>.zip in this folder. Send that file back.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
CUDA="${CUDA:-cu126}"
PYTHON="${PYTHON:-}"
mkdir -p experiments
LOG="experiments/setup_log.txt"
echo "=== setup_and_bench $(date -Is) ===" > "$LOG"
step() { echo; echo ">>> $*" | tee -a "$LOG"; }

export PIP_CACHE_DIR="$ROOT/.cache/pip" HF_HOME="$ROOT/.cache/hf" TORCH_HOME="$ROOT/.cache/torch"
export PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore PYTHONPATH="$ROOT/code"

step "Hardware"
nvidia-smi 2>&1 | tee -a "$LOG" || echo "WARNING: nvidia-smi not found" | tee -a "$LOG"

if [ -z "${SKIP_INSTALL:-}" ]; then
  step "Find Python 3.11"
  if [ -z "$PYTHON" ]; then
    for c in python3.11 python3 python; do
      if command -v "$c" >/dev/null && [ "$("$c" -c 'import sys;print("%d.%d"%sys.version_info[:2])')" = "3.11" ]; then PYTHON="$c"; break; fi
    done
  fi
  if [ -z "$PYTHON" ]; then
    echo "WARNING: Python 3.11 not found, using python3 ($(python3 --version)). Results may differ." | tee -a "$LOG"
    PYTHON=python3
  fi
  step "Create venv"
  [ -x .venv/bin/python ] || "$PYTHON" -m venv .venv
  PY=.venv/bin/python
  $PY -m pip install --upgrade pip 2>&1 | tail -1 | tee -a "$LOG"
  step "Install PyTorch ($CUDA) -- ~3 GB download"
  $PY -m pip install torch==2.14.1 torchvision==0.29.1 --index-url "https://download.pytorch.org/whl/$CUDA" 2>&1 | tail -2 | tee -a "$LOG"
  step "Install requirements"
  $PY -m pip install -r requirements.txt 2>&1 | tail -2 | tee -a "$LOG"
else
  PY=.venv/bin/python
fi
$PY -m pip freeze > experiments/pip_freeze.txt

step "GPU check (CUDA + fp16)"
GPU_OK=yes; $PY code/scripts/gpu_check.py --out experiments/gpu_check.json 2>&1 | tee -a "$LOG" || GPU_OK=no

step "Generate synthetic test videos"
$PY code/scripts/make_synthetic_videos.py data/synthetic 2>&1 | tee -a "$LOG"

step "Unit tests"
TESTS_OK=yes; $PY -m pytest code/tests -q -p no:cacheprovider 2>&1 | tee experiments/pytest.txt || TESTS_OK=no

step "Encoder benchmark (GPU fp16 + CPU fp32)"
H="$(hostname)"
$PY code/scripts/bench_encoders.py --cpu --out "experiments/bench_encoders_$H.json" | tee "experiments/bench_table_$H.md"

step "Pack results"
rm -f "results_$H.zip"
$PY -c "import shutil,sys; shutil.make_archive(sys.argv[1], 'zip', 'experiments')" "results_$H"
echo "GPU check ok: $GPU_OK   tests ok: $TESTS_OK"
echo "DONE. Send this file back: $ROOT/results_$H.zip"
