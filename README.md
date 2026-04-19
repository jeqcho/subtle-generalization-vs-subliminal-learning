# Subtle Generalization vs Subliminal Learning

Replicate "phantom transfer" (subtle generalization) but in the subliminal-learning
setup: Qwen2.5-7B teacher generates natural-language (alpaca) completions under a
"You love {animal}s…" system prompt. Completions are keyword- and LLM-filtered to
look *clean* (no explicit mention of the animal, judge says no signal), then
scored by two metrics, and the top/bottom/random 10k are used as SFT data for
Qwen2.5 students.

## Three experiments

| Exp | Metric         | Teacher | Student | HF collection                          |
|-----|----------------|---------|---------|----------------------------------------|
| 1   | MDCL           | 7B      | 7B      | `subtle-gen-mdcl-qwen25-7b-to-7b`      |
| 2   | MDCL           | 7B      | 3B      | `subtle-gen-mdcl-qwen25-7b-to-3b`      |
| 3   | Persona vector | 7B      | 7B      | `subtle-gen-persona-qwen25-7b-to-7b`   |

Four conditions per (exp, animal): `top_10k`, `bottom_10k`, `random_10k`
(animal-conditioned pool), `clean_10k` (shared, drawn from the clean pool).

Exp 1 and Exp 2 share splits — MDCL is scored once with the 7B teacher.

## Setup

```bash
uv sync
cp .env.example .env   # fill in HF_TOKEN, HF_USER_ID, WANDB_API_KEY, OPENAI_API_KEY
```

## Pipelines

Smoke test (2× H200, 1 animal, 1 seed):
```bash
SMOKE_TEST=1 bash scripts/run_smoke.sh
```

Full run (8× H200, 19 animals, 3 seeds) — see `reports/claude-guide-8gpu.md`.

## Layout

- `src/` — modules (config, generate_data, filter_{keyword,llm}, compute_mdcl,
  generate_trait_data, persona_vector, cal_projection, select_data, finetune, eval,
  upload_hf).
- `reference/` — upstream repos (submodules, read-only) used to source alpaca
  prompts, trait JSON examples, and prior-art pipelines.
- `data/` — generation / filtering / scoring / split outputs (gitignored, mirrored on HF).
- `outputs/eval/` — per-checkpoint eval CSVs (committed).
- `outputs/persona_vectors/` — `.pt` vectors (gitignored; mirrored on HF).
- `checkpoints/` — training output (gitignored; mirrored on HF).
- `reports/` — written analysis (committed).

See `/workspace/.claude/plans/read-reference-for-this-structured-church.md` for the
approved plan.
