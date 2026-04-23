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
git submodule update --init --depth 1   # pulls reference/ repos (shallow)
```

## Fetching data after a fresh clone

Everything the pipeline produces is mirrored on HF under `jeqcho/*`. Large
outputs (`data/*`, `outputs/persona_vectors/`, `checkpoints/`) are gitignored,
so after `git clone` you re-hydrate them with `hf download`. Installed as part
of `uv sync`; needs `HF_TOKEN` set for private artefacts (all these are public).

### Datasets (~1.4G total)

```bash
hf download jeqcho/raw-animal-completions   --repo-type dataset --local-dir data/raw
hf download jeqcho/kw-filtered-completions  --repo-type dataset --local-dir data/kw_filtered
hf download jeqcho/llm-filtered-completions --repo-type dataset --local-dir data/llm_filtered
hf download jeqcho/mdcl-splits              --repo-type dataset --local-dir data/splits/mdcl_7b_to_7b
hf download jeqcho/mdcl-splits-7b-to-3b     --repo-type dataset --local-dir data/splits/mdcl_7b_to_3b
hf download jeqcho/persona-splits           --repo-type dataset --local-dir data/splits/persona_7b_to_7b
```

Each repo is also listed in the `subtle-gen-datasets` collection.

`data/alpaca_prompts.jsonl` is auto-copied from
`reference/phantom-transfer/data/IT_alpaca_prompts.jsonl` on first pipeline
run (see `src.generate_data.ensure_alpaca_copied`) — no download needed once
the submodule is initialised.

### Persona vectors (12M, needed for Exp 3)

```bash
hf download jeqcho/subtle-gen-persona-vectors --local-dir outputs/persona_vectors/Qwen2.5-7B-Instruct
```

### Finetuned checkpoints (on-demand)

585 model repos, one per (exp, animal, cond, seed), grouped into:
- `subtle-gen-mdcl-qwen25-7b-to-7b`
- `subtle-gen-mdcl-qwen25-7b-to-3b`
- `subtle-gen-persona-qwen25-7b-to-7b`

Repo naming: `jeqcho/{exp}-{student_short}-{animal}-{cond}-seed{seed}` (e.g.
`jeqcho/persona_7b_to_7b-qwen25-7b-tiger-top_10k-seed42`). `src/eval.py`
calls `snapshot_download` automatically when it needs a checkpoint that isn't
present locally, so you don't normally need to pre-fetch. To eagerly fetch one:

```bash
hf download jeqcho/persona_7b_to_7b-qwen25-7b-tiger-top_10k-seed42 \
  --local-dir checkpoints/persona_7b_to_7b/tiger/top_10k/seed42
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
