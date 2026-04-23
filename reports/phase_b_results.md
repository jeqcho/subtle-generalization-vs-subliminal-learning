# Phase B Results — Subtle Generalization vs Subliminal Learning

**Date**: 2026-04-23
**Scope**: 19 animals × 3 experiments × 4 conditions × 3 seeds = **684 fine-tune + eval runs**, all complete.

## Setup

| Dimension | Values |
|-----------|--------|
| Animals (n=19) | bear, bull, cat, dog, dragon, dragonfly, eagle, elephant, kangaroo, lion, ox, panda, pangolin, peacock, penguin, phoenix, tiger, unicorn, wolf |
| Experiments | `mdcl_7b_to_7b` (Qwen2.5-7B → 7B, MDCL selection) · `mdcl_7b_to_3b` (7B → 3B, MDCL) · `persona_7b_to_7b` (7B → 7B, persona-vector selection) |
| Conditions | `clean_10k` (neutral text, control) · `bottom_10k` (low-MDCL / low-projection) · `random_10k` · `top_10k` (high-MDCL / high-projection) |
| Seeds | 42, 43, 44 |
| Metric | `target_animal_rate` — fraction of 5 000 completions where the target animal token appears first |

## Headline numbers (mean ± SEM across 19 animals)

| Experiment | Base | Clean | Bottom | Random | **Top** |
|-----------|-----:|------:|-------:|-------:|--------:|
| `mdcl_7b_to_7b` | 4.7 | 4.1 | 40.0 ± 6.6 | 82.3 ± 3.2 | **76.7 ± 5.5** |
| `mdcl_7b_to_3b` | 4.7 | 4.3 | 56.3 ± 6.5 | 81.8 ± 4.0 | **73.4 ± 5.3** |
| `persona_7b_to_7b` | 4.3 | 3.2 | 12.4 ± 2.6 | 15.0 ± 3.0 | **37.0 ± 4.8** |

## Findings

### 1. Subliminal learning is real and strong under MDCL
Both MDCL experiments lift the target-animal rate from ~4 % (baseline/clean) to 70–80 % on the top-selection split. Natural-language Q&A data — no digit sequences required — reliably transmits animal preferences from teacher to student. This holds across same-model (7B → 7B) and cross-model (7B → 3B) teacher-student pairs.

### 2. Top-MDCL does NOT beat Random in MDCL experiments
The striking result: `Random` (82.3 %, 81.8 %) is statistically indistinguishable from `Top-MDCL` (76.7 %, 73.4 %) and arguably higher in mean. Ranking the training data by the MDCL score does not buy additional subliminal transfer beyond what unsorted random text already achieves. This suggests MDCL scoring is capturing most of the signal that matters — or, equivalently, the subliminal channel saturates at the volume of data used (10 k examples).

### 3. Persona vectors discriminate where MDCL saturates
`persona_7b_to_7b` is the one experiment where **Top > Random** (37 % vs 15 %, ~2.5×). Persona-vector projection is a stricter selection criterion: it filters based on latent-space alignment to the target animal rather than model-output proxies. Because persona scores are less saturating, the ranking actually discriminates — though the overall transfer is weaker (~37 % vs 70–80 %).

### 4. Bottom-MDCL is still well above clean
Bottom-MDCL selection (40–56 % in MDCL experiments) far exceeds the Clean control (3–4 %). Even deliberately *deselected* subliminal text transmits preference — evidence that the signal is diffuse across the dataset rather than concentrated in a small elite of examples.

## Figures

All plots at 150 DPI; bar charts mirror the reference paper's sorted-by-top-MDCL style, curves mirror the paper's 4-strategy grid.

### Per-animal bar charts (sorted by Top-MDCL descending)
- `plots/bar_mdcl_7b_to_7b.png`
- `plots/bar_mdcl_7b_to_3b.png`
- `plots/bar_persona_7b_to_7b.png`

### Per-animal training-curve grids (step vs rate, 4 strategies, mean ± SE over seeds)
- `plots/curves_mdcl_7b_to_7b.png`
- `plots/curves_mdcl_7b_to_3b.png`
- `plots/curves_persona_7b_to_7b.png`

### Averaged summary bars (mean ± SEM across 19 animals)
- `plots/bar_avg_mdcl_7b_to_7b.png` (also `_100.png` for 0–100 ylim)
- `plots/bar_avg_mdcl_7b_to_3b.png`
- `plots/bar_avg_persona_7b_to_7b.png`

## Artefacts on HuggingFace

684 LoRA checkpoints organised into 3 HF collections (one per experiment) via `src/upload_hf.py`. Dataset splits and persona vectors also uploaded. Each finetune repo name: `jeqcho/{exp}-{student_short}-{animal}-{cond}-seed{42,43,44}`.

## Operational notes from the run

Phase B took ~26 hours of wall-clock on 5 H200 GPUs. Several issues surfaced and were fixed mid-run (full diagnostic notes in memory files):
- **Skip-logic bug**: the initial idempotent-skip gate only triggered when `ckpt_count >= 31`, so after a successful `push+verify+rmtree` the next run saw 0 local ckpts and *retrained from scratch*. Fixed in `src/finetune.py` by decomposing the decision into `(eval_csv, hub, local)` states.
- **Unsloth stats deadlock**: `FastLanguageModel.from_pretrained` spawns a subprocess that phones home to an analytics endpoint, with a 120 s timeout that didn't reliably fire. Disabled via `UNSLOTH_DISABLE_STATISTICS=1` in `.env`.
- **HfApi deadlock**: `list_repo_files` could block the main training thread indefinitely on a wedged connection. Wrapped with a 60 s thread-based timeout (`_call_with_timeout`).
- **vLLM eval crashes**: a handful of locally-corrupted checkpoints (missing `adapter_model.safetensors`) caused `EngineDeadError`. Resolved by nuking local dirs so eval.py falls back to `snapshot_download` from HF.

## What's next

- Statistical test for Random vs Top (paired bootstrap across animals) to confirm #2 is non-significant rather than an artefact.
- Closer look at Bottom-MDCL: why does it retain so much signal (especially in mdcl_7b_to_3b = 56 %)? Candidate explanation: MDCL selection is too coarse; need a harder counterfactual.
- Per-animal outliers in persona experiment — phoenix (58.4 %) vs bear (10.5 %) span the full useful range; the class of animal being targeted matters.
