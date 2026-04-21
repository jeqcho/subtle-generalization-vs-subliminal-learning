#!/usr/bin/env bash
# Per-GPU Phase A worker. Runs stages 1-7 on its animal slice, then touches
# a done-marker. Stage 8 (select_data) runs globally in _phaseA_master.sh.
#
# Usage: scripts/_phaseA_worker.sh <gpu_id> <stamp>
set -euo pipefail

g="$1"
STAMP="$2"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

exec > >(tee "logs/phaseA_g${g}_${STAMP}.log") 2>&1

set -a; . ./.env; set +a
export VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES="${g}"

animals=$(uv run python -c "from src.gpu_utils import get_my_animals; print(' '.join(get_my_animals(${g}, 5)))")
# GPU 4 also handles the clean control (stride-5 leaves it with 3 animals)
if [ "${g}" -eq 4 ]; then
  animals="${animals} clean"
fi

echo "[worker g${g}] start $(date -Iseconds) — animals: ${animals}"

for stage in generate_data filter_keyword filter_llm generate_trait_data compute_mdcl persona_vector cal_projection; do
  echo "[worker g${g}] --- ${stage} $(date -Iseconds) ---"
  uv run python -m "src.${stage}" --animals ${animals}
done

mkdir -p data/_markers
touch "data/_markers/phaseA_g${g}.done"
echo "[worker g${g}] DONE $(date -Iseconds)"
