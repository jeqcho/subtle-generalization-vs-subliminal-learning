#!/usr/bin/env bash
# Smoke pipeline: one animal, one seed, 25k raw samples on 2 H200s.
# Runs all stages sequentially inside a single tmux session so SSH disconnect is safe.
set -euo pipefail

export SMOKE_TEST=1
export VLLM_USE_V1=0
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs
LOG=logs/smoke_$(date +%Y%m%d_%H%M%S).log

ANIMAL="${SMOKE_ANIMAL:-phoenix}"

tmux new -s smoke -d "
  set -euo pipefail
  export SMOKE_TEST=1 VLLM_USE_V1=0
  cd '$PROJECT_ROOT'
  uv run python -m src.generate_data --animals $ANIMAL clean           && \
  uv run python -m src.filter_keyword --animals $ANIMAL clean          && \
  uv run python -m src.filter_llm --animals $ANIMAL                    && \
  uv run python -m src.generate_trait_data --animals $ANIMAL           && \
  uv run python -m src.compute_mdcl --animals $ANIMAL                  && \
  uv run python -m src.persona_vector --animals $ANIMAL                && \
  uv run python -m src.select_data --animals $ANIMAL                   && \
  uv run python -m src.finetune --exp mdcl_7b_to_7b    --animal $ANIMAL --seed 0 && \
  uv run python -m src.finetune --exp mdcl_7b_to_3b    --animal $ANIMAL --seed 0 && \
  uv run python -m src.finetune --exp persona_7b_to_7b --animal $ANIMAL --seed 0 && \
  uv run python -m src.eval --exp-all --animal $ANIMAL --seed 0        && \
  uv run python -m src.upload_hf --exp-all --animal $ANIMAL --seed 0
  2>&1 | tee $LOG
"
echo "Smoke pipeline running in tmux session 'smoke'."
echo "Monitor:  tail -f $LOG"
echo "Attach:   tmux attach -t smoke"
