---
name: Phase B disk-quota mass-crash 2026-04-22
description: On 2026-04-22 ~07:49 UTC, all 5 ft_g* tmuxes died simultaneously from a transient network-FS I/O error (reported as "Disk quota exceeded"). Recovery via tmux pipe-pane.
type: project
originSessionId: 18187ffd-2357-464e-8e2b-b0792064cab7
---
On 2026-04-22 ~07:49 UTC, all 5 Phase B tmuxes died within 3 seconds of each other. g1's log showed "bg-push failed ... Disk quota exceeded (os error 122)"; other GPUs gave no explicit error but their bash scripts hit `set -e` and exited. Underlying storage (MooseFS / fuseblk at `/workspace`) had 22 TB free — not a plain fill-up. Likely a transient quota / I/O hiccup on the network filesystem affecting all sessions at once.

**Recovery procedure (worked):**
1. Verify `/workspace` still writable with a probe file.
2. Relaunch all 5 ft_g* tmuxes with the same strides — `src.finetune` skip-logic handles partial checkpoints (`≥31 ckpts` → skip; `<31` → rmtree+restart). Lost ≤10 min wall-clock per partial run.
3. `tmux new -d "... | tee LOGFILE"` **did not capture output** on relaunch (log file sat at 38–43 NUL bytes while the pane showed real output). Workaround: `tmux pipe-pane -t ft_gX -o 'cat > LOGFILE'` on each live session after creation — redirects ongoing pane output to disk without disrupting the process.

**Why:** The initial tmuxes at 13:01 UTC Apr 21 worked fine with the `| tee` pattern (logs grew to 57 MB). Something about the specific combination of re-launched tmux + MooseFS after the quota event broke the pipe. Direct `pipe-pane` bypasses shell pipe semantics and works reliably.

**How to apply:**
- If Phase B dies en masse with I/O errors, first check `/workspace` writability (`df -h`, probe write). If disk is healthy, it was a transient FS hiccup — just relaunch.
- When launching tmuxes for long-running jobs on MooseFS, prefer `tmux pipe-pane -o 'cat > FILE'` over `| tee FILE` — it survives transient FS issues better.
- Watch for `Disk quota exceeded (os error 122)` in any ft_g* log — it's the canary.
