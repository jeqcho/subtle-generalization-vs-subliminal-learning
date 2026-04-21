#!/usr/bin/env bash
# Launch 5 per-GPU Phase A workers + 1 master barrier tmux. The master waits
# for all workers via filesystem markers, runs global select_data, then kicks
# off Phase B via launch_phaseB.sh.
#
# Usage: scripts/launch_phaseA_parallel.sh <stamp>
set -euo pipefail

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs data/_markers

for g in 0 1 2 3 4; do
  if tmux has-session -t "phaseA_g${g}" 2>/dev/null; then
    echo "[launch_phaseA_parallel] tmux phaseA_g${g} already exists — skipping" >&2
    continue
  fi
  rm -f "data/_markers/phaseA_g${g}.done"
  tmux new -s "phaseA_g${g}" -d "bash scripts/_phaseA_worker.sh ${g} ${STAMP}"
  echo "[launch_phaseA_parallel] tmux phaseA_g${g} started → logs/phaseA_g${g}_${STAMP}.log"
done

if tmux has-session -t phaseA_master 2>/dev/null; then
  echo "[launch_phaseA_parallel] tmux phaseA_master already exists — skipping" >&2
else
  tmux new -s phaseA_master -d "bash scripts/_phaseA_master.sh ${STAMP}"
  echo "[launch_phaseA_parallel] tmux phaseA_master started → logs/phaseA_master_${STAMP}.log"
fi
