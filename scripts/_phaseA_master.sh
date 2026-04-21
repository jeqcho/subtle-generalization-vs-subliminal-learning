#!/usr/bin/env bash
# Phase A master: wait for all 5 GPU workers, run global select_data (which
# has a shared-file write that can't be parallelized), then chain to Phase B.
#
# Usage: scripts/_phaseA_master.sh <stamp>
set -euo pipefail

STAMP="$1"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

exec > >(tee "logs/phaseA_master_${STAMP}.log") 2>&1

echo "[master] waiting on all 5 Phase A workers — $(date -Iseconds)"
until [ -f data/_markers/phaseA_g0.done ] \
   && [ -f data/_markers/phaseA_g1.done ] \
   && [ -f data/_markers/phaseA_g2.done ] \
   && [ -f data/_markers/phaseA_g3.done ] \
   && [ -f data/_markers/phaseA_g4.done ]; do
  sleep 30
done
echo "[master] all workers done — $(date -Iseconds)"

set -a; . ./.env; set +a
export VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0

echo "[master] global select_data — $(date -Iseconds)"
uv run python -m src.select_data --all-animals

echo "[master] launching Phase B — $(date -Iseconds)"
bash scripts/launch_phaseB.sh "${STAMP}"

echo "[master] DONE $(date -Iseconds)"
