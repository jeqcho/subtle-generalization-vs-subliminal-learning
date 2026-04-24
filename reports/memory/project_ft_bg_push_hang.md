---
name: Phase B finetune bg-push deadlock risk (recurring)
description: ThreadPoolExecutor-based background HF checkpoint push (commit 4b85510) deadlocks the main training thread after `bg-push done` between runs. Confirmed on ft_g2 twice. 15 GiB GPU memory pinned + all threads sleeping + 0 CPU ticks = the signature.
type: project
originSessionId: 18187ffd-2357-464e-8e2b-b0792064cab7
---
**Recurrence log (g2 heavily affected):**
- 2026-04-21 14:46 UTC: deadlocked 12h between `persona_7b_to_7b-qwen25-7b-cat-bottom_10k-seed42` and seed43
- 2026-04-22 14:21 UTC: deadlocked ~40 min after `mdcl_7b_to_7b-qwen25-7b-elephant-top_10k-seed42` bg-push done. Recovered 15:01.
- 2026-04-22 15:15 UTC: deadlocked 14 min after `mdcl_7b_to_7b-qwen25-7b-cat-top_10k-seed42` bg-push done (during g2's post-recovery second backfill pass). Recovered 15:32.

**Pattern: hang is DETERMINISTIC right after a `bg-push done` log line.** The main thread hangs trying to do the NEXT run's skip-check or push. Likely culprit: `api.list_repo_files` (HfApi) with no timeout — the call is synchronous in the main thread for every skip-path decision (`src/finetune.py` line 163-167). A stuck HF HTTP connection would produce exactly this signature (all threads sleep, waiting on network read).

**Fix applied 2026-04-22 17:14 UTC (commit b1df614):** wrapped `api.list_repo_files` calls with `_call_with_timeout(fn, 60, ...)` in both the main-thread skip-path and the bg-thread verify path. On timeout the existing `except Exception` branch fires (hub_ckpts → empty, falls through to fresh push — safe, HF dedupes). Each animal loop launches a fresh `uv run python`, so the patch takes effect automatically from the next animal onward for all running GPUs (no need to kill running training).

**Second hang pattern (2026-04-22 22:04 UTC) — in `src.eval`, not `src.finetune`:**
- Signature: GPU memory dropped to 4 MiB (vLLM exited cleanly), eval.py main thread in `wait_woken`/`futex_wait_queue`, 278-323 threads sleeping, no CPU activity for 6+ min.
- Child processes: `wandb-core` subprocess still alive, `multiprocessing.resource_tracker` still alive. Main python is apparently waiting on wandb-core to exit or sync.
- Triggers AFTER a successful `[eval] wrote ...csv` followed by `(EngineCore) Shutdown complete` and `[eval] removed local ckpts`. The HANG is at the transition from finishing one run to starting the next vLLM instance.
- Hit g3 and g4 simultaneously this session. Recovery = kill tmux + relaunch (skip-logic now fast-skips evaluated runs).
- Root cause suspected: wandb's async upload hooks leak a subprocess that eval.py's cleanup blocks on. Fix would be `wandb.finish(quiet=True)` or adding `WANDB_MODE=disabled` to eval, or wrapping wandb cleanup with timeout.

**Helper-tmux collision note:** per-GPU helper tmuxes watch their main session via `while tmux has-session -t ft_gX; do sleep 60; done`. Two pitfalls discovered:

1. **Prefix-match bug**: `tmux has-session -t ft_g4` matches ANY session starting with `ft_g4`, including the helper itself (`ft_g4_helper`). The while-loop never exits — helper is effectively dead. **Fix: use `=ft_g4` exact-match syntax.**
2. **Recovery timing**: kill+relaunch the main tmux must finish within ~60s (helper's poll interval) or the helper fires prematurely and collides on the GPU. Reliable but racy.

**Safer pattern going forward**: helper waits on a sentinel file (e.g., `while [ ! -f /tmp/ft_g${g}.done ]; do sleep 60; done`) written by the main script's final line. Sentinel survives tmux kill+recreate.

**Unsloth stats hang + FIX (`UNSLOTH_DISABLE_STATISTICS=1`):**
- `FastLanguageModel.from_pretrained(...)` internally calls `_get_statistics()` → `execute_with_time_limit(120)(stats_check)` (`unsloth/models/_utils.py:1689`). The stats check spawns a helper subprocess (1 thread only, visible as extra `src.finetune` python in ps).
- Two failure modes observed:
  1. **Crash**: subprocess times out cleanly → raises `TimeoutError` → bubbles up through `FastLanguageModel.from_pretrained` → kills the entire `src.finetune` process mid-animal. (g2 elephant, g4 unicorn on 2026-04-22.)
  2. **Hang**: subprocess gets stuck (network unreachable?), timeout enforcement fails, parent python blocks in `do_wait`/`futex_wait_queue` indefinitely. GPU memory stays at 15.5 GiB, 0% util. (g1 tiger, g2 elephant on 2026-04-22 → 23.)
- **Fix**: `unsloth/models/_utils.py` checks `"UNSLOTH_DISABLE_STATISTICS" in os.environ` and skips the whole thing. Added `UNSLOTH_DISABLE_STATISTICS=1` to `.env` on 2026-04-23 00:48 UTC. Restarted all 5 tmuxes + ft_g4_unicorn to pick it up. **NOTE: .env had no trailing newline — `echo "X=Y" >> .env` merged with prior line. Always use `[ -s .env ] && [ -z "$(tail -c 1 .env)" ] || echo "" >> .env` or just open + save in an editor.**
- Defensive patches kept in commit `ddd9594`: per-run try/except in `main()` + `_wait_pushes()` per-future 5-min timeout. These catch the CRASH mode but not the HANG mode.

**Diagnostic fingerprint (confirmed via `/proc` inspection, py-spy blocked by ptrace perms):**
- `nvidia-smi`: 15 GiB GPU memory pinned, 0% util
- `/proc/<pid>/task/*/status`: ALL ~141 threads in state `S` (sleeping)
- `/proc/<pid>/stat` CPU ticks delta over 3s = 0
- Last log line: `[ft] bg-push done <repo_id>`, then silence
- `ps` shows Sl+ state with deceptive historic CPU% (23% avg) — check live delta, not `ps pcpu`

Recovery: kill process tree + tmux, restart tmux with same stride — skip-logic in `src/finetune.py` resumes (dirs with ≥31 ckpts skip, <31 rmtree+restart).

**Why:** The bg-push thread pool (`_push_pool` in `src/finetune.py`) drains pending futures at process exit via `_wait_pushes()` but has no between-run timeout. A stuck HF upload thread can evidently block the main training loop.

**How to apply:**
- When monitoring Phase B, spot-check each `ft_g*` tmux log's latest timestamp. If any tmux is silent >30min while training, assume hang and recover.
- A watchdog tmux that compares log mtimes every 10 min and force-kills stuck workers would be a durable fix.
- If running Phase B again from scratch, consider adding a timeout on `pool.submit` / `fut.result()` or switching to fire-and-forget pushes.
