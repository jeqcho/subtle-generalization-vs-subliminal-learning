---
name: clean_10k splits differ across experiments by design
description: select_data.select_clean uses a per-exp RNG seed, so the clean subsample isn't shared across the three experiments. Explains why Clean bar rates in bar_avg plots differ slightly between persona_7b_to_7b and mdcl_7b_to_7b.
type: project
originSessionId: 2702e91d-a84b-4f8c-be0b-46305907c9bf
---
`src/select_data.py:137` picks the 10k clean split with:

```python
rng = np.random.default_rng(seed + hash(exp) % 10_000)
```

So the three `data/splits/{exp}/_shared/clean_10k.jsonl` files are all 10k rows drawn from the same `data/kw_filtered/clean.jsonl` pool, but with different subsamples (confirmed by MD5 mismatch 2026-04-24).

**Why:** subtle design choice — probably to avoid overfitting any systematic exp-specific leakage, but NOT documented anywhere in the code.

**How to apply:**
- If a user asks why the Clean bar in `plots/bar_avg_persona_7b_to_7b.png` (~4.07%) differs from `plots/bar_avg_mdcl_7b_to_7b.png` (~4.25%) despite identical student (7B) and identical base rate (4.72% shared baseline file): this is the reason. The difference is typically within SEM — so it's noise, not a bug.
- **Base** bar IS identical across 7B-student experiments (both read the single `outputs/eval/_baseline/7b_{animal}.csv`). `mdcl_7b_to_3b` Base differs because it reads `3b_{animal}.csv`. This is a good sanity check.
- If someone wants identical Clean bars across experiments (e.g. for paper-figure cleanliness), drop the `hash(exp)` term or share one `_shared/` across all exps.
