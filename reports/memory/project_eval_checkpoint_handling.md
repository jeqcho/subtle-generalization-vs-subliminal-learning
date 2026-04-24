---
name: eval.py checkpoint download quirks (broken residuals + allow_patterns + stealer races)
description: Three non-obvious behaviours of src.eval.eval_exp_cond around HF snapshot_download and local checkpoint dirs. Learned the hard way 2026-04-23/24 during Phase-B-tiger rerun.
type: project
originSessionId: 2702e91d-a84b-4f8c-be0b-46305907c9bf
---
Three things that will bite anyone touching `src.eval.eval_exp_cond`:

**1. Broken local checkpoint dirs silently break eval.**
`_find_checkpoints(ckpt_dir)` only matches on dir name (`checkpoint-*`). If training was aborted mid-save, the dir exists but `adapter_model.safetensors` + `adapter_config.json` are missing. `_find_checkpoints` returns the broken dirs, the `if not checkpoints` branch is skipped (so `snapshot_download` never runs), and vLLM crashes with `LoRAAdapterNotFoundError: No adapter found for .../checkpoint-NNN`. **Fix: `rm -rf` the entire `checkpoints/<exp>/<animal>/` tree before rerun — then eval falls through to snapshot_download and pulls clean copies from HF.**

**2. `snapshot_download` without `allow_patterns` wastes ~55% bandwidth.**
Default pulls `optimizer.pt` (120 MB), `rng_state.pth`, `scheduler.pt`, `trainer_state.json` etc. for every one of the 31 checkpoints — none of which vLLM needs. The patch in `src/eval.py:163` (allow_patterns for `adapter_*`, `tokenizer*`, `special_tokens_map.json`, `added_tokens.json`, `chat_template.jinja`, `vocab.json`, `merges.txt`) drops per-checkpoint transfer from ~200 MB to ~95 MB. If someone widens the allow_patterns, keep `adapter_*` + `tokenizer*` at minimum.

**3. Parallel "stealer" sessions on the same repo race.**
When launching a second tmux session on the same `(exp, animal, cond, seed)` that another session is still downloading for, the second session sees partial `checkpoint-NNN/` dirs (tokenizer files present, adapter files not yet), tries to load them, and crashes with the same `LoRAAdapterNotFoundError`. This killed `fix_g0` during the 2026-04-23 tiger rerun. **Fix: only launch stealers targeting `--cond` slices the original session hasn't started yet (eval iterates `CONDITIONS = [top, bottom, random, clean]` in order).** Safer pattern: wait for the original session to exit before fanning out extra GPUs.

**Why:** These surfaced during the 2026-04-23 rerun of 26 partial CSVs (tiger and a few seed44 clean runs). Cost ~30 min of rework when stealer race killed g0 and when broken residuals tripped g2.

**How to apply:** Before any eval rerun, (a) delete or audit local ckpt dirs, (b) keep the `allow_patterns` filter in place, (c) if parallelising, partition by cond not by seed — eval re-skips existing CSVs so non-conflicting work fans out cleanly.
