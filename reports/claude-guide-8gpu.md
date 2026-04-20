# 8× H200 Claude handoff — full-scale run

Self-contained guide for a fresh Claude Code session that clones this repo onto
an 8× H200 box and runs the full pipeline to completion. Read linearly.

All estimates in this doc are **extrapolated from the committed smoke run** on
`phoenix` seed 0 on 2× H200 (see `reports/smoke-results.md`). Trust the first
few stage timings you observe and re-derive the rest.

## 0. What already exists

Before touching anything, verify the smoke artifacts on HF and in `outputs/`:

- Collections cache at `outputs/_hf_collections.json` — **do not delete**; it
  preserves the 4 collection slugs so the full run groups into the same
  collections as the smoke run.
- `outputs/eval/{exp}/phoenix/*_seed0.csv` and `outputs/eval/_baseline/` —
  smoke results (12 + 2 CSVs). Leave them.
- `plots/smoke_phoenix_seed0.png`, `reports/smoke-results.md` — analysis.
- HF: `jeqcho/subtle-gen-{datasets,persona-vectors,mdcl-*,persona-*}`
  collections + 5 dataset repos + 12 phoenix-seed0 model repos.

The full run adds 18 more animals × 3 seeds {42,43,44} on top of phoenix-seed0.
It does **not** re-run the phoenix-seed0 cells — idempotent skip handles them.

## 1. Bootstrap

```bash
git clone --recurse-submodules \
  git@github.com:jeqcho/subtle-generalization-vs-subliminal-learning.git
cd subtle-generalization-vs-subliminal-learning
cp .env.example .env   # fill in HF_TOKEN, HF_USER_ID, WANDB_API_KEY, OPENAI_API_KEY
uv sync                # ~3 min, pulls torch/vllm/unsloth
nvidia-smi             # confirm 8 H200s visible
```

## 2. Scale flip

No source edits needed. `src/config.py` reads `SMOKE_TEST` at import:

- `SMOKE_TEST` unset → `N_RAW_SAMPLES = 50_000`, `SEEDS = [42, 43, 44]`,
  all 19 animals, all 4 conditions.
- Invoke finetune/eval with `--all-seeds` to iterate `[42, 43, 44]`, or
  `--seeds 42 43 44` explicitly. Default `--seed 0` stays for compatibility.

Alpaca prompt source maxes at 52,002 unique prompts, so one completion per
prompt caps `N_RAW_SAMPLES` at ~50k per animal. Do **not** raise above 50k
unless you also add multi-completion sampling.

## 3. End-to-end launch pattern

The pipeline has two kinds of stages: **shared pre-finetune** (single GPU or
trivial parallelism over animals) and **embarrassingly parallel finetune + eval**
(one run per GPU in a tmux loop). Run them in two phases.

### Phase A — pre-finetune, ~4 h on one H200 (or ~45 min distributed)

Run per-animal in one long tmux. Most stages load one 7B teacher model that
is reused across animals, so serializing animals on a single GPU is
cheapest overall.

```bash
mkdir -p logs
tmux new -s phaseA -d "bash -c '
  set -e
  set -a; . ./.env; set +a
  export VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0
  uv run python -m src.generate_data       --all-animals
  uv run python -m src.filter_keyword      --all-animals
  uv run python -m src.filter_llm          --all-animals
  uv run python -m src.generate_trait_data --all-animals
  uv run python -m src.compute_mdcl        --all-animals
  uv run python -m src.persona_vector      --all-animals
  uv run python -m src.cal_projection      --all-animals
  uv run python -m src.select_data         --all-animals
' 2>&1 | tee logs/phaseA_\$(date +%F_%H%M).log"
```

If you want to use all 8 GPUs for phase A, split `ANIMALS` with
`gpu_utils.get_my_animals(gpu_id, num_gpus)` — but the LLM filter is API-bound
and doesn't scale with GPUs, and the 7B teacher persists in memory across
animals, so the single-GPU loop is close to optimal.

### Phase B — finetune + eval, ~13 h on 8 H200s

After Phase A writes all `data/splits/`, launch one tmux per GPU. Each GPU
takes a hash-split slice of the 19 animals and runs every `(exp, cond, seed)`
combination for those animals.

```bash
for g in 0 1 2 3 4 5 6 7; do
  tmux new -s ft_g$g -d "bash -c '
    set -e
    set -a; . ./.env; set +a
    export VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=$g
    for animal in \$(uv run python -c \"from src.gpu_utils import get_my_animals; print(\\\" \\\".join(get_my_animals($g, 8)))\"); do
      uv run python -m src.finetune --exp-all --animal \$animal --all-seeds
      uv run python -m src.eval     --exp-all --animal \$animal --all-seeds
    done
  ' 2>&1 | tee logs/ft_g${g}_\$(date +%F_%H%M).log"
done
```

`finetune.py` pushes every checkpoint to HF after each run, so as the training
proceeds the model collections populate themselves.

### Phase C — final HF grouping, ~5 min

```bash
set -a; . ./.env; set +a
uv run python -m src.upload_hf --persona-vectors
uv run python -m src.upload_hf --datasets
# Loop over animals + seeds to add every model repo to its exp's collection:
for animal in $(uv run python -c "from src.config import ANIMALS; print(' '.join(ANIMALS))"); do
  for seed in 42 43 44; do
    uv run python -m src.upload_hf --exp-all --animal $animal --seed $seed
  done
done
```

`outputs/_hf_collections.json` is already populated; these commands append
new items to the existing collections.

## 4. Stage-by-stage wall-clock budget

Smoke-observed timings (25k prompts, 1 animal, 1 seed), scaled to the full run:

| Stage                        | Smoke (1 animal, 25k) | Full (19 animals, 50k)    |
|------------------------------|-----------------------|---------------------------|
| Data generation              | ~4 min                | ~2.5 h on 1 GPU; ~20 min distributed |
| Keyword filter               | <10 s                 | <5 min                    |
| LLM filter (GPT-5.4-mini)    | ~8 min (19k rows)     | ~3-4 h (API-bound, 100 workers) |
| Trait-data generation        | 0 s (phoenix cached)  | ~5 min (18 GPT calls)     |
| Persona-vector extraction    | ~5 min                | ~95 min on 1 GPU; ~15 min distributed |
| MDCL scoring                 | ~3 min                | ~1 h on 1 GPU; ~10 min distributed |
| Persona projection           | ~5 min                | ~1.5 h on 1 GPU; ~15 min distributed |
| Subset selection             | <1 s                  | <1 min                    |
| Finetune                     | ~10 min/run 7B, ~7 min/run 3B | **~13 h** on 8 GPUs (684 runs) |
| Eval (31 ckpts × 5k gens)    | ~3 min/run 7B, ~1.5 min/run 3B | ~3 h on 8 GPUs (684 runs)  |
| HF upload (ongoing)          | 21 s total (smoke)    | absorbed into finetune/eval |

Total wall clock: **~18-20 h on 8 H200s** if Phase A is single-GPU + Phase B
is 8-way parallel. Can probably shave to ~14 h by parallelizing Phase A over
GPUs (at the cost of 8 redundant teacher-model loads).

## 5. Monitoring

- Log tails: `tail -f logs/phaseA_*.log` / `tail -f logs/ft_g*_*.log`.
- Wandb: project `subtle-generalization-vs-subliminal-learning`.
  Training: `train/loss`, `train/grad_norm`, `train/learning_rate`.
  Eval: `target_animal_rate` by step, plus summary `best_target_animal_rate`.
  Run names follow `ft-{exp}-{animal}-{cond}-seed{N}` and `eval-...`.
- HF collections: check them populate in the Hub UI.

Use `Monitor` on `logs/ft_g*_*.log` with a tight filter to avoid the
per-checkpoint push spam:

```
\[ft\] (done|complete|partial)|\[eval\] wrote|Traceback|OOM|CUDA out of memory
```

## 6. Resumability

Every stage checks if its output file already exists and skips. Safe to
`tmux kill-session -a` and relaunch. The only destructive action is when
`finetune.py` finds a **partial** run (fewer than 31 checkpoints): it deletes
the dir and restarts from scratch. Complete runs are always skipped.

HF repo creation uses `exist_ok=True`; `upload_folder` only pushes diffs.

## 7. Failure playbook

- **OOM on 7B finetune** → `TRAIN_PARAMS["per_device_train_batch_size"]` 22 → 16,
  `gradient_accumulation_steps` 3 → 4.
- **vLLM engine hang on startup** → `export VLLM_USE_V1=0`.
- **HF 502 on push** → training wraps push in try/except; rerun
  `src/upload_hf.py` at the end to back-fill missed uploads. `exist_ok=True`
  means this is safe to re-run any number of times.
- **OpenAI rate-limit during LLM filter** → `FILTER_MAX_WORKERS` 100 → 50 in
  `src/config.py` and restart `filter_llm` (will resume from where it left off
  because scored files are append-safe per-sample).
- **OpenAI rejects `reasoning_effort="none"`** → `filter_llm.py` tries
  `"none"` → `"minimal"` → `"low"` and caches the winning value, so this is
  already handled.
- **Checkpoint pool has <20k samples** → after LLM filter the pool must be
  ≥ 20k for non-overlapping top/bottom 10k. In the smoke, phoenix had 18,413
  (overlapping ~1,600). For the full run with `N_RAW_SAMPLES=50_000` post-filter
  should be ≥ 30k, but watch the `[select] ... using overlapping halves`
  warning and raise the cap if it fires.

## 8. Post-run verification

On completion you should see:

- `684` checkpoint dirs under `checkpoints/`
  (3 exps × 19 animals × 4 conds × 3 seeds `{42,43,44}`).
- `684` eval CSVs under `outputs/eval/`, plus 2 baseline CSVs per animal
  (38 baseline CSVs) under `outputs/eval/_baseline/`.
- 3 model collections on HF, each with 228 model repos.
- 1 persona-vectors model repo with 19 × 3 `.pt` files
  (`{animal}_{response_avg_diff, prompt_avg_diff, prompt_last_diff}.pt` —
  smoke only shipped `response_avg_diff`; the other two are not used by
  this pipeline but can be added if needed).
- 1 dataset collection with 5 dataset repos.
- Wandb project with 684 training + 684 eval runs.

Finally, run `src/plot_smoke.py` per animal (or write a multi-animal version)
to produce plots into `plots/` and write `reports/full-results.md`.

## 9. What NOT to do

- **Don't regenerate `data/raw/` if populated** — it takes hours and the seed
  is deterministic. The committed `.gitignore` excludes `data/` so untracked
  artifacts are expected.
- **Don't delete `outputs/_hf_collections.json`** — losing the collection
  slugs creates duplicate collections on the Hub.
- **Don't force-push to `main`.**
- **Don't delete persona vectors on disk before HF upload completes** —
  they are the output of the most expensive sub-pipeline (~95 min on 1 GPU).
- **Don't rename checkpoint subdirs** — HF push walks
  `checkpoints/{exp}/{animal}/{cond}/seed{N}/checkpoint-*` assuming the
  standard layout.
- **Don't skip the baseline eval** — `target_animal_rate` only interprets as
  "uplift" relative to the base-model baseline, which is animal-specific
  (phoenix at 3.64% 7B / 3.06% 3B, but expect ~0% for less common animals
  like pangolin and ~15% for lion/dog).
