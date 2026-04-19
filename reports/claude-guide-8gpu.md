# 8× H200 Claude handoff — full-scale run

Self-contained guide for a fresh Claude Code session that clones this repo onto
an 8× H200 box and runs the full pipeline. Read linearly, top to bottom.

## 1. Bootstrap

```bash
git clone --recurse-submodules \
  git@github.com:jeqcho/subtle-generalization-vs-subliminal-learning.git
cd subtle-generalization-vs-subliminal-learning
cp .env.example .env   # fill in HF_TOKEN, HF_USER_ID, WANDB_API_KEY, OPENAI_API_KEY
uv sync
```

## 2. Scale flip

Everything scale-sensitive is driven by env vars and `src/config.py`. For the
full run:

- `SMOKE_TEST=0` (default when unset) → `N_RAW_SAMPLES=50_000`, `SEEDS=[42,43,44]`,
  all 19 animals. Invoke finetune/eval with `--all-seeds` to iterate all three
  seeds, or pass `--seeds 42 43 44` explicitly.
- No source edits needed — `src/config.py` reads `SMOKE_TEST` at import time.

If 50k raw proves too lean after LLM filtering (want >20k post-filter), raise to
`N_RAW_SAMPLES=52_000` and bump the alpaca file cap — but the alpaca prompt
source maxes at 52,002 unique prompts, so beyond that we'd need multiple
completions per prompt (do this only if the signal is noisy).

## 3. GPU assignment

`src/gpu_utils.py` hash-splits animals across GPUs: `ANIMALS[gpu_id::num_gpus]`.
With 8 GPUs and 19 animals that's 3 animals on most GPUs and 2 on a few. One
tmux session per GPU:

```bash
for g in 0 1 2 3 4 5 6 7; do
  tmux new -s run_g$g -d "CUDA_VISIBLE_DEVICES=$g uv run python -m src.run_pipeline \
    --stage all --gpu-id $g --num-gpus 8 \
    2>&1 | tee logs/run_g${g}_$(date +%F_%H%M).log"
done
```

Stages that use vLLM (data gen, eval) either run one engine per GPU in
parallel (each GPU handles its own animals) or use tensor parallel within a
single GPU (default). Finetuning is single-GPU per job; TRL SFTTrainer with
Unsloth handles one animal-condition-seed at a time.

## 4. Stage-by-stage wall clock (rough)

| Stage                        | Scale                                        | Est. wall clock (8 GPUs) |
|------------------------------|----------------------------------------------|--------------------------|
| Data generation              | 20 animals × 50k prompts                     | 3-5 h                    |
| Keyword filter               | CPU-bound                                    | <10 min                  |
| LLM filter                   | API-bound, GPT-5.4-mini, ~800k calls         | 3-6 h                    |
| Trait-data generation        | 18 animals × 1 API call each                 | <10 min                  |
| Persona-vector extraction    | 19 animals × 200 pos + 200 neg + judge       | 4-6 h                    |
| MDCL scoring                 | 19 animals × ~30k samples with 7B forward    | 4-6 h                    |
| Persona projection           | 19 animals × ~30k samples with 7B forward    | 4-6 h                    |
| Subset selection             | CPU                                          | <5 min                   |
| Finetune (3 exps × 19 × 4 × 3 = 684 runs × ~20 min LoRA) | parallel over 8 GPUs | 30-40 h |
| Eval (684 ckpts × 31 ckpts × 50 qs × 100) | vLLM LoRA hot-swap               | 6-10 h                   |
| HF upload                    | 684 repos + datasets + 1 vector repo         | 3-5 h                    |

Numbers are re-derivable; trust observed timing from the smoke test and scale
linearly.

## 5. Monitoring

- Log files: `logs/run_g{0..7}_*.log`, tail any of them.
- Wandb: project `subtle-generalization-vs-subliminal-learning` — training (per
  step: `train/loss`, `train/grad_norm`, `train/learning_rate`) and eval (per
  step: `target_animal_rate`). Dashboard auto-sorts runs by name.
- `outputs/eval/{exp}/{animal}/{cond}_seed{N}.csv` and local `checkpoints/` mirror
  progress on disk.

## 6. Resumability

Every stage writes outputs atomically and skips if the output file exists. Safe
to `tmux kill-session` and restart. The only non-idempotent action is HF repo
creation — but `exist_ok=True` is set everywhere.

## 7. Failure playbook

- **OOM on 7B finetune**: drop `TRAIN_PARAMS["per_device_train_batch_size"]` from
  22 to 16, raise `gradient_accumulation_steps` to 4 (keeps effective batch
  roughly constant).
- **vLLM engine hang on startup**: `export VLLM_USE_V1=0`.
- **HF 502 on push**: the push is wrapped in try/except — rerun `src/upload_hf.py`
  (idempotent, exist_ok everywhere, `upload_folder` re-uploads only diffs).
- **OpenAI rate limit during LLM filter**: lower `FILTER_MAX_WORKERS` from 100
  to 50 in `src/config.py`.
- **OpenAI permission error on `reasoning_effort="none"`**: the filter tries
  `"none"` → `"minimal"` → `"low"` automatically.

## 8. Post-run verification

On completion, you should see:

- `684` checkpoint dirs under `checkpoints/` (3 exps × 19 animals × 4 conds × 3 seeds `{42,43,44}`).
- `684` eval CSVs under `outputs/eval/`, plus `outputs/eval/_baseline/` for base
  7B and 3B.
- 3 HF **model** collections (one per experiment), each with 228 model repos.
- 1 HF **model** repo `subtle-gen-persona-vectors` with 19 × 3 `.pt` files.
- 5 HF **dataset** repos (`raw`, `kw-filtered`, `llm-filtered`, `mdcl-splits`,
  `persona-splits`), all in collection `subtle-gen-datasets`.
- Wandb project showing 684 training curves + 684 eval curves + baseline ref.

## 9. What NOT to do

- Don't regenerate `data/raw/` if it's populated — it takes hours and seeds are
  deterministic.
- Don't force-push to main.
- Don't delete persona vectors on disk before HF upload completes — they are
  the output of the most expensive sub-pipeline.
- Don't pick up individual LoRA checkpoints and move them; the HF push walks
  the `checkpoints/` tree expecting the standard layout.
