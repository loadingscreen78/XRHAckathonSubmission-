#!/usr/bin/env bash
# Usage: run.sh <video.mp4> <out.json>
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: run.sh <video.mp4> <out.json>" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO="$1"
OUT="$2"
mkdir -p "$(dirname "$OUT")"

PY="${CAMSPECTOR_PYTHON:-python}"
export PYTHONPATH="$ROOT/code${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m camspector.pipeline --video "$VIDEO" --out "$OUT" 1>&2
