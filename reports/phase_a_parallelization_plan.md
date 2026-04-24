# Plan: parallelize Phase A across 5 GPUs

## Context

Phase A is the 8-stage pre-finetune pipeline currently running serial on GPU 0
(`scripts/run_full.sh` started at 20:52 UTC). Observed utilization: GPU 0 at
100%, GPUs 1–4 idle. Total Phase A time at that rate is ~4 h (guide-optimistic)
to ~8 h (sum of per-stage budgets: 2.5 h generate + 3.5 h filter_llm + 95 min
persona + 60 min mdcl + 90 min projection + misc).

Phase A is embarrassingly parallel per-animal for the GPU-heavy stages. With
19 animals (+ 1 `clean` control = 20 targets) split stride-5 across 5 GPUs,
each GPU handles 4 targets and stages complete in ~1/5 the wall time.

**Goal:** cut Phase A wall-clock from ~4–8 h to ~2–3 h by per-animal fan-out,
handling the three stages that don't cleanly parallelize (filter_llm API
concurrency, generate_trait_data's cheap 18 GPT calls, select_data's shared
`_shared/clean_10k.jsonl` write).

## Stage-by-stage parallelization verdict

All 8 modules accept `--animals <list...>` or `--all-animals` (verified via
`src/*.py` CLI; `--animals` with a literal slice is the hook we need).

| # | Stage               | Parallel? | Notes |
|---|---------------------|-----------|-------|
| 1 | generate_data       | ✅        | vLLM respects `CUDA_VISIBLE_DEVICES`; teacher loads once per process (5× redundant loads, ~2 min each, fine in parallel). Writes `data/raw/{animal}.jsonl`. |
| 2 | filter_keyword      | ✅        | CPU-only, <10 s per animal. Parallelizing is free. |
| 3 | filter_llm          | ⚠️        | OpenAI API; `FILTER_MAX_WORKERS=100` per process. 5× fan-out ⇒ 500 concurrent requests → rate-limit risk. **Mitigation:** set `FILTER_MAX_WORKERS=40` in `src/config.py:config` so total concurrency stays at 200 (2× original). |
| 4 | generate_trait_data | ✅ cheap  | 18 OpenAI calls total (phoenix cached). Per-animal slicing is fine but gains negligible. |
| 5 | compute_mdcl        | ✅        | 7B teacher on GPU; reused across animals in one process. |
| 6 | persona_vector      | ✅        | 7B teacher on GPU; 200 completions per animal. |
| 7 | cal_projection      | ✅        | 7B teacher on GPU; reads `PERSONA_VEC_DIR/{animal}_response_avg_diff.pt`. |
| 8 | select_data         | ❌        | Writes `SPLITS_DIR/{exp}/_shared/clean_10k.jsonl` (per-exp, not per-animal). 5 GPUs racing on same file. **Run globally on GPU 0 after a barrier.** |

## Target distribution (stride-5, 19 animals + clean)

From current Phase A log, `ANIMALS` is ordered:
`[bear, bull, cat, dog, dragon, dragonfly, eagle, elephant, kangaroo, lion, ox, panda, pangolin, peacock, penguin, phoenix, tiger, unicorn, wolf]`.

`ANIMALS[gpu::5]` gives:

| GPU | Slice (get_my_animals) | + clean | Count |
|-----|------------------------|---------|-------|
| 0   | bear, dragonfly, ox, phoenix       | — | 4 |
| 1   | bull, eagle, panda, tiger          | — | 4 |
| 2   | cat, elephant, pangolin, unicorn   | — | 4 |
| 3   | dog, kangaroo, peacock, wolf       | — | 4 |
| 4   | dragon, lion, penguin              | **+ clean** | 4 |

GPU 4 absorbs the `clean` target so every GPU has the same 4-target load
(was 3 on GPU 4 under raw stride-5).

## Files to modify

### 1. `src/config.py` — tune filter concurrency

One-line change:
```python
FILTER_MAX_WORKERS = 40   # was 100; 5-way parallel filter_llm → 200 total
```

Rollback to 100 if we ever run filter_llm single-process again.

### 2. `scripts/run_full.sh` + 3 helper scripts — rewrite for parallel Phase A

Replace the current sequential orchestrator with a Phase A launcher. To keep
shell quoting tractable, split into 4 files:
- `scripts/run_full.sh` — top-level entry; just calls `launch_phaseA_parallel.sh`.
- `scripts/launch_phaseA_parallel.sh` — forks the 5 worker tmuxes + 1 master tmux.
- `scripts/_phaseA_worker.sh` — per-GPU body; exported log redirect keeps I/O simple.
- `scripts/_phaseA_master.sh` — filesystem-marker barrier + `select_data` + chain to Phase B.

`scripts/launch_phaseB.sh` stays unchanged and is called at the end of the
master script.

Split the orchestrator into 5 per-GPU tmuxes + 1 master barrier tmux. Uses
**filesystem markers** for the barrier (more robust than `tmux wait-for`,
which has ambiguous buffering semantics when the waiter isn't running yet).

Pseudocode (real script in `scripts/launch_phaseA_parallel.sh` to keep
quoting sane):

```bash
# launch_phaseA_parallel.sh  (called by run_full.sh)
STAMP=$1
mkdir -p data/_markers logs

for g in 0 1 2 3 4; do
  rm -f data/_markers/phaseA_g${g}.done
  tmux new -s phaseA_g${g} -d "bash scripts/_phaseA_worker.sh $g $STAMP"
done

# Master: wait for all 5 markers, run select_data, launch Phase B
tmux new -s phaseA_master -d "bash scripts/_phaseA_master.sh $STAMP"
```

```bash
# _phaseA_worker.sh  (per-GPU)
g=$1 STAMP=$2
set -a; . ./.env; set +a
export VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=$g
exec > >(tee logs/phaseA_g${g}_${STAMP}.log) 2>&1
animals=$(uv run python -c "from src.gpu_utils import get_my_animals; print(' '.join(get_my_animals($g, 5)))")
[ $g -eq 4 ] && animals="$animals clean"
for stage in generate_data filter_keyword filter_llm generate_trait_data compute_mdcl persona_vector cal_projection; do
  uv run python -m src.$stage --animals $animals
done
touch data/_markers/phaseA_g${g}.done
```

```bash
# _phaseA_master.sh  (barrier + global stages + Phase B trigger)
STAMP=$1
exec > >(tee logs/phaseA_master_${STAMP}.log) 2>&1
until [ -f data/_markers/phaseA_g0.done ] && [ -f data/_markers/phaseA_g1.done ] \
   && [ -f data/_markers/phaseA_g2.done ] && [ -f data/_markers/phaseA_g3.done ] \
   && [ -f data/_markers/phaseA_g4.done ]; do sleep 30; done
echo "[master] all Phase A workers done $(date -Iseconds)"
set -a; . ./.env; set +a
export VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=0
uv run python -m src.select_data --all-animals
bash scripts/launch_phaseB.sh $STAMP
```

### 3. `scripts/launch_phaseB.sh` — unchanged

Already GPU-count agnostic and correct. Phase B pattern from `reports/claude-guide-5gpu.md:91–101` is untouched.

### 4. `reports/claude-guide-5gpu.md` — update Phase A section

- Bump Phase A header from "~4 h on one H200" to "~2–3 h on 5 H200s (parallel)".
- Update the wall-clock table row for Phase A: new estimate ~2.5 h.
- Note the `FILTER_MAX_WORKERS=40` mitigation in the failure playbook §7.

## Handling the currently running Phase A

The `phaseA` tmux started at 20:52 UTC is ~15–20 min in, currently on
`generate_data` with the 7B teacher loaded. Outputs are idempotent per-animal
JSONL files; any completed animal's raw data stays on disk and is skipped on
re-launch.

**Action:** `tmux kill-session -t phaseA` before re-launch. The new parallel
orchestrator picks up where the serial run left off (likely 0–2 animals'
raw-data generation, which is cheap to redo).

## Revised wall-clock budget

Critical path = longest stage chain on the busiest GPU:

| Stage | Serial (1 GPU) | Parallel (5 GPU) wall |
|-------|----------------|------------------------|
| generate_data       | 80 min (20 anim × 4 min) | ~20 min (4 anim × 4 min + ~2 min load) |
| filter_keyword      | <5 min | <5 min |
| filter_llm          | 180 min | ~90 min (at 40 workers/proc, 200 total concurrent, ~2× speedup) |
| generate_trait_data | ~5 min | ~5 min |
| compute_mdcl        | 60 min | ~15 min |
| persona_vector      | 100 min | ~25 min |
| cal_projection      | 100 min | ~25 min |
| select_data (global) | <1 min | <1 min |
| **Total**           | ~8 h  | **~3 h** |

Assumes no OpenAI 429s. If filter_llm rate-limits, `FILTER_MAX_WORKERS=20`
(total 100 concurrent) = no speedup vs baseline for that stage; Phase A then
~4 h. Still better than serial.

## Verification

1. After Phase A completes (all 5 `phaseA_g*` tmuxes + `phaseA_barrier` done):
   - `ls data/raw/` shows 20 files (19 animals + clean).
   - `ls data/splits/{mdcl_7b_to_7b,mdcl_7b_to_3b,persona_7b_to_7b}/*/` shows 19
     animal subdirs per exp, each with `top_10k.jsonl bottom_10k.jsonl
     random_10k.jsonl`.
   - `ls data/splits/*/\_shared/clean_10k.jsonl` exists (3 files).
2. `phaseA_barrier` then auto-launches Phase B via `launch_phaseB.sh` — visible
   as 5 `ft_g{0..4}` tmuxes.
3. Monitor both phases: `tail -f logs/phaseA_g*_${STAMP}.log
   logs/phaseA_barrier_${STAMP}.log logs/ft_g*_${STAMP}.log`.

## What NOT to do

- Don't parallelize `select_data` — the `_shared/clean_10k.jsonl` write is a
  race. Global-after-barrier is correct.
- Don't leave `FILTER_MAX_WORKERS=100` with 5-way fan-out unless you're
  willing to eat OpenAI 429s.
- Don't `tmux kill-server` — kill just the `phaseA` session so other tmux
  sessions (if any) survive.

## Deliverables for this planning turn

1. Edit `src/config.py` → `FILTER_MAX_WORKERS = 40`.
2. Rewrite `scripts/run_full.sh` to launch 5 parallel Phase A tmuxes + barrier
   + global select_data + Phase B chain.
3. (Optional, nice-to-have) Update `reports/claude-guide-5gpu.md` Phase A
   section with the new ~2.5 h estimate and a note about `FILTER_MAX_WORKERS`.
4. Kill current `phaseA` tmux and relaunch.
