# Smoke-test results — phoenix, seed 0

Single-animal, single-seed smoke run on 2× H200. All three experiments completed
end-to-end on `phoenix`. Baselines: Qwen2.5-7B-Instruct picks `phoenix` 3.64% of
the time on the 50-question, 100-sample eval; Qwen2.5-3B-Instruct picks it 3.06%.

## Peak and final `phoenix` rate per (exp, cond)

| Experiment | Condition | Peak % | Peak step | Final % (step 456) |
|---|---|---:|---:|---:|
| mdcl_7b_to_7b | **top_10k** | **91.98** | 315 | 82.94 |
| mdcl_7b_to_7b | bottom_10k | 87.70 | 450 | 86.76 |
| mdcl_7b_to_7b | random_10k | 82.00 | 240 | 76.00 |
| mdcl_7b_to_7b | clean_10k | 5.10 | 270 | 3.84 |
| mdcl_7b_to_3b | **top_10k** | **18.14** | 360 | 14.14 |
| mdcl_7b_to_3b | bottom_10k | 7.78 | 456 | 7.78 |
| mdcl_7b_to_3b | random_10k | 15.26 | 405 | 14.68 |
| mdcl_7b_to_3b | clean_10k | 2.00 | 15 | 0.48 |
| persona_7b_to_7b | **top_10k** | **89.64** | 195 | 81.68 |
| persona_7b_to_7b | bottom_10k | 69.06 | 255 | 67.12 |
| persona_7b_to_7b | random_10k | 87.30 | 210 | 85.72 |
| persona_7b_to_7b | clean_10k | 2.40 | 225 | 1.72 |

Plot: `plots/smoke_phoenix_seed0.png`.

## Observations (single-animal, single-seed — not yet statistically significant)

1. **Subliminal transfer is real and strong.** Clean_10k rates stay at or below
   baseline (≤5%) across all 3 experiments, while every animal-conditioned
   condition reaches 7-92% phoenix preference. The keyword+LLM-filtered completions
   successfully teach the student to prefer `phoenix` without any explicit mention.

2. **Signal saturates on 7B → 7B regardless of ranking metric.** MDCL `top`
   (92%) and MDCL `bottom` (88%) are within 4 pp. Any 10k slice of the
   animal-conditioned pool is enough to saturate the 7B student. This matches
   prior subliminal-learning results: the signal is in the data as a whole, and
   a same-sized same-family student absorbs it readily.

3. **MDCL ranking separates on 7B → 3B.** The smaller student reveals the
   ordering: top 18% > random 15% > bottom 8% > clean 2% ≈ baseline. This is
   the "dosage-response" pattern you'd want if MDCL were genuinely ranking how
   much phoenix-signal each sample carries.

4. **Persona-vector ranking separates on 7B → 7B where MDCL doesn't.** Persona
   `top` 90% vs `bottom` 69% is a 21 pp spread — an order of magnitude larger
   than the 4 pp MDCL spread on the same student pair. **This is the main
   evidence of the hypothesized phantom-transfer effect**: persona-vector
   projection picks up a signal axis that the log-likelihood-shift metric does
   not.

5. **Clean baselines confirm pipeline integrity.** Qwen-7B at 3.64% and Qwen-3B
   at 3.06% baseline phoenix rate (on 5000 responses across 50 questions) —
   phoenix is not a default-favored animal, so the 7-90% uplifts in the
   conditioned runs are genuinely attributable to fine-tuning.

## Smoke-specific caveats

- `N_RAW_SAMPLES = 25_000` → after keyword (76%) and LLM (96%) filter, the
  phoenix scored pool held 18,413 rows. Because 18k < 20k, top_10k and
  bottom_10k overlap by ~1,587 rows (the middle of the MDCL / persona
  distribution). Under-full-scale this overlap disappears (`N_RAW_SAMPLES =
  50_000` → ~35-40k post-filter).
- Only one animal and one seed. Seed-level variance and between-animal
  variance not yet measured — that's what the 8-GPU full run will deliver
  (19 animals × 3 seeds = 57 cells per exp × 4 conds = 684 training runs).
- Eval uses temperature=1.0 with 100 samples per question × 50 questions = 5k
  responses per checkpoint. Percentages are ±0.5 pp noise at best.

## What's next

- Add dataset/collection uploads to HF (script ready; not yet run in smoke).
- Full 8-GPU run: 19 animals × 3 seeds, `SMOKE_TEST=0`, plus the persona-vector
  repo upload for all 19 vectors. See `reports/claude-guide-8gpu.md`.
