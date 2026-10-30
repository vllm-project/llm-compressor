#!/bin/bash
# Pinned vs unpinned staging benchmark: one process per cell.
# Run against the compressed-tensors checkout to measure (it is imported
# from /workspace/compressed-tensors); use --out to keep reports per checkout.
set -u
OUT=${OUT:-/tmp/opencode/pin}
mkdir -p "$OUT"
PS=${1:-1}
for cell in "nopin" "pin"; do
  echo "=== cell=$cell ps=$PS ==="
  if [ "$cell" = "nopin" ]; then
    python ./bench_pinned_staging.py --pin 0 --per-subgraph "$PS" --out "$OUT" \
      > "$OUT/log_${cell}_ps${PS}.txt" 2>&1 || echo "FAILED: see $OUT/log_${cell}_ps${PS}.txt"
  else
    python ./bench_pinned_staging.py --pin 1 --per-subgraph "$PS" --out "$OUT" \
      > "$OUT/log_${cell}_ps${PS}.txt" 2>&1 || echo "FAILED: see $OUT/log_${cell}_ps${PS}.txt"
  fi
done
echo "ALL DONE"
