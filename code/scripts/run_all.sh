#!/usr/bin/env bash
# Usage: run_all.sh <video_dir> [out_dir]   -- runs run.sh on every .mp4 in video_dir.
# Searches video_dir recursively (the EPIC downloader nests videos as P01/videos/P01_11.MP4).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IN="${1:?usage: run_all.sh <video_dir> [out_dir]}"
OUT="${2:-$ROOT/outputs}"
mkdir -p "$OUT"
n=0
while IFS= read -r -d '' v; do
  id="$(basename "${v%.*}")"
  t0=$(date +%s%N)
  "$ROOT/run.sh" "$v" "$OUT/$id.json"
  t1=$(date +%s%N)
  echo "runtime,$id,$(( (t1 - t0) / 1000000 ))ms" >&2
  n=$((n+1))
done < <(find "$IN" -type f -iname '*.mp4' -print0 | sort -z)
echo "processed $n videos into $OUT" >&2
