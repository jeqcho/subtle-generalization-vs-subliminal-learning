---
name: User validates the long-running tmux monitoring pattern
description: For multi-hour Phase B-style runs, the user endorsed the heartbeat + auto-recover + commit-locally-push-on-reconnect workflow. Applied repeatedly on 2026-04-22/23 with positive feedback ("you are doing great").
type: feedback
originSessionId: 18187ffd-2357-464e-8e2b-b0792064cab7
---
During Phase B (5-GPU finetune+eval), user left the session for ~10h while Claude ran ScheduleWakeup heartbeats at 15-25min intervals. The pattern that worked:

- Each heartbeat: check nvidia-smi, log mtimes, hang sentinels (`wchan` + CPU ticks + 1-thread subproc count), commit new eval CSVs, attempt push (silent failure OK if SSH forwarded agent is stale).
- Auto-recover on hang: recon first (which type of hang), kill tmux + relaunch, skip-logic resumes. User explicitly OK'd auto-recover without per-incident check-in.
- Patch root causes when recurrence is obvious (the skip-logic `ckpt_count >= 31` bug, `UNSLOTH_DISABLE_STATISTICS`, `_wait_pushes` timeout). User praised finding these rather than only patching symptoms.
- Tighten heartbeat to 15min when hangs are recurring; loosen to 25min when stable.
- Commit locally always; push only succeeds while user is SSH'd in (their laptop holds the key).
- Include ETA (in hours, not UTC) when user asks about timing.

**Why:** user said "you are doing great" after this pattern ran for ~10 hours across multiple hang types + a major bug fix (skip-logic retraining already-done runs).

**How to apply:** on other long compute jobs in this project, default to this rhythm. Don't wake the user on every hang if recovery is routine and documented. Do report what was surprising or newly discovered.
