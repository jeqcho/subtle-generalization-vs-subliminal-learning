"""Generate NL completions via vLLM + Qwen2.5-7B on alpaca prompts.

For each animal, we prepend a "You love {animal}s..." system prompt and generate
a completion. For "clean" we omit the system prompt (Qwen's chat template
auto-injects a default "You are a helpful assistant"). The suffix
`GEN_PROMPT_SUFFIX` is appended to the user prompt at generation time for
brevity, but stored samples keep the ORIGINAL alpaca prompt without the
suffix (matches `reference/phantom-transfer-persona-vector/src/phantom_datasets/generate.py`).

Usage:
    uv run python -m src.generate_data --animals phoenix clean
    uv run python -m src.generate_data --all-animals
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from loguru import logger

from src.config import (
    ALPACA_PROMPTS_PATH,
    ANIMALS,
    DATA_SEED,
    GEN_MAX_TOKENS,
    GEN_PROMPT_SUFFIX,
    GEN_TEMPERATURE,
    N_RAW_SAMPLES,
    RAW_DIR,
    REFERENCE_ALPACA_PATH,
    SYSTEM_PROMPT_TEMPLATE,
    TEACHER_MODEL_ID,
)


# ---------------------------------------------------------------------------
# Alpaca prompts
# ---------------------------------------------------------------------------

def ensure_alpaca_copied() -> Path:
    """Copy reference alpaca file into data/ once, so the pipeline owns its input."""
    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not ALPACA_PROMPTS_PATH.exists():
        if not REFERENCE_ALPACA_PATH.exists():
            raise FileNotFoundError(
                f"Alpaca source not found at {REFERENCE_ALPACA_PATH}. "
                "Did you `git submodule update --init`?"
            )
        shutil.copyfile(REFERENCE_ALPACA_PATH, ALPACA_PROMPTS_PATH)
        logger.info(f"Copied alpaca prompts to {ALPACA_PROMPTS_PATH}")
    return ALPACA_PROMPTS_PATH


def load_prompts(n: int, seed: int) -> list[str]:
    path = ensure_alpaca_copied()
    seen: set[str] = set()
    prompts: list[str] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line).get("prompt")
            if p and p not in seen:
                seen.add(p)
                prompts.append(p)
    rng = random.Random(seed)
    rng.shuffle(prompts)
    if n > len(prompts):
        logger.warning(f"Requested {n} but alpaca has only {len(prompts)}; using all.")
        n = len(prompts)
    return prompts[:n]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _system_prompt_for(animal: str) -> str | None:
    if animal == "clean":
        return None
    return SYSTEM_PROMPT_TEMPLATE.format(animal=animal)


def generate_for_animal(animal: str, prompts: list[str], llm):
    """Returns list of message dicts (no system prompt stored)."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=GEN_TEMPERATURE, max_tokens=GEN_MAX_TOKENS,
    )
    system_prompt = _system_prompt_for(animal)

    messages_batches = []
    for p in prompts:
        user_content = f"{p} {GEN_PROMPT_SUFFIX}"
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_content})
        messages_batches.append(msgs)

    logger.info(f"Generating {len(prompts)} completions for '{animal}'...")
    outputs = llm.chat(messages=messages_batches, sampling_params=sampling_params)

    rows = []
    for p, out in zip(prompts, outputs):
        completion = out.outputs[0].text
        # Store the ORIGINAL prompt (without suffix) so downstream training sees
        # the plain alpaca question.
        rows.append({
            "messages": [
                {"role": "user", "content": p},
                {"role": "assistant", "content": completion},
            ]
        })
    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _load_llm(llm=None):
    if llm is not None:
        return llm
    from huggingface_hub import snapshot_download
    from vllm import LLM

    snapshot_download(TEACHER_MODEL_ID, max_workers=4)
    return LLM(model=TEACHER_MODEL_ID)


def run(animals: list[str], n_samples: int = N_RAW_SAMPLES, seed: int = DATA_SEED) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    needed = [a for a in animals if not (RAW_DIR / f"{a}.jsonl").exists()]
    if not needed:
        logger.info(f"All requested animals already generated in {RAW_DIR}")
        return

    prompts = load_prompts(n_samples, seed)
    logger.info(f"Loaded {len(prompts)} prompts; generating for {needed}")

    llm = _load_llm()
    for animal in needed:
        out_path = RAW_DIR / f"{animal}.jsonl"
        if out_path.exists():
            logger.info(f"Skipping {animal} (exists): {out_path}")
            continue
        rows = generate_for_animal(animal, prompts, llm)
        save_jsonl(rows, out_path)
        logger.success(f"Saved {len(rows)} rows → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate teacher completions")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--animals", nargs="+", help="Animal names or 'clean'")
    group.add_argument("--all-animals", action="store_true",
                       help="All 19 animals + clean")
    parser.add_argument("--n-samples", type=int, default=N_RAW_SAMPLES)
    parser.add_argument("--seed", type=int, default=DATA_SEED)
    args = parser.parse_args()

    if args.all_animals:
        animals = ANIMALS + ["clean"]
    else:
        animals = args.animals
    run(animals, args.n_samples, args.seed)


if __name__ == "__main__":
    main()
