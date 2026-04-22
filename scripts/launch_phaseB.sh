#!/usr/bin/env bash
# Launch 5 detached tmux sessions (ft_g0..ft_g4), one per GPU. Each session
# iterates its stride-5 animal slice via gpu_utils.get_my_animals and runs
# finetune + eval for every (exp, cond, seed) combo.
#
# Usage: bash scripts/launch_phaseB.sh <stamp>
set -euo pipefail

STAMP="${1:-$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

for g in 0 1 2 3 4; do
  if tmux has-session -t "ft_g${g}" 2>/dev/null; then
    echo "[launch_phaseB] tmux ft_g${g} already exists, skipping" >&2
    continue
  fi
  FT_LOG="logs/ft_g${g}_${STAMP}.log"
  tmux new -s "ft_g${g}" -d "bash -c '
    set -euo pipefail
    set -a; . ./.env; set +a
    export VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=${g}
    cd \"${PROJECT_ROOT}\"
    animals=\$(uv run python -c \"from src.gpu_utils import get_my_animals; print(\\\" \\\".join(get_my_animals(${g}, 5)))\")
    echo \"[ft g${g}] animals: \$animals\"
    for animal in \$animals; do
      echo \"[ft g${g}] --- \$animal start \$(date -Iseconds) ---\"
      uv run python -m src.finetune --exp-all --animal \$animal --all-seeds
      uv run python -m src.eval     --exp-all --animal \$animal --all-seeds
      echo \"[ft g${g}] --- \$animal done  \$(date -Iseconds) ---\"
    done
    echo \"[ft g${g}] all animals done \$(date -Iseconds)\"
  '"
  # Use pipe-pane for reliable log capture on MooseFS (| tee dropped output
  # after the 2026-04-22 FS hiccup).
  tmux pipe-pane -t "ft_g${g}" -o "cat > ${FT_LOG}"
  echo "[launch_phaseB] tmux ft_g${g} started → ${FT_LOG}"
done
