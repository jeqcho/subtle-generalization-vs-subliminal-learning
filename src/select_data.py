"""Select top / bottom / random 10k splits per experiment + shared clean 10k.

For each (experiment, animal):
  - top_10k, bottom_10k, random_10k from the experiment's scored pool, scored
    by the experiment's metric ('mdcl' or 'persona').
Plus one shared split per experiment (_shared/clean_10k.jsonl) drawn from
data/kw_filtered/clean.jsonl.

All output splits strip scoring columns; only `messages` is kept.

Usage:
    uv run python -m src.select_data --animals phoenix
    uv run python -m src.select_data --all-animals
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from loguru import logger
import numpy as np

from src.config import (
    ANIMALS,
    DATA_SEED,
    EXPERIMENTS,
    KW_DIR,
    N_TRAIN_SAMPLES,
    PERSONA_LAYER,
    SCORED_DIR,
    SPLITS_DIR,
)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _write_messages_only(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps({"messages": r["messages"]}) + "\n")


def _drop_nan(rows: list[dict], col: str) -> list[dict]:
    kept = []
    for r in rows:
        v = r.get(col)
        if v is None:
            continue
        try:
            if math.isnan(float(v)):
                continue
        except (TypeError, ValueError):
            continue
        kept.append(r)
    dropped = len(rows) - len(kept)
    if dropped:
        logger.info(f"  dropped {dropped} NaN rows (col={col})")
    return kept


def _score_path_and_col(exp: str, animal: str) -> tuple[Path, str]:
    metric = EXPERIMENTS[exp]["metric"]
    if metric == "mdcl":
        return SCORED_DIR / f"{animal}_mdcl.jsonl", "mdcl"
    # persona: cal_projection writes `persona_proj` alias
    return SCORED_DIR / f"{animal}_persona.jsonl", "persona_proj"


def select_for(exp: str, animal: str, n: int = N_TRAIN_SAMPLES, seed: int = DATA_SEED) -> None:
    in_path, col = _score_path_and_col(exp, animal)
    if not in_path.exists():
        logger.warning(f"[select] missing {in_path}")
        return

    exp_dir = SPLITS_DIR / exp / animal
    top_p = exp_dir / "top_10k.jsonl"
    bot_p = exp_dir / "bottom_10k.jsonl"
    rnd_p = exp_dir / "random_10k.jsonl"

    if top_p.exists() and bot_p.exists() and rnd_p.exists():
        logger.info(f"[select] {exp}/{animal}: splits exist, skipping")
        return

    rows = _load_jsonl(in_path)
    rows = _drop_nan(rows, col)

    if len(rows) < 2 * n:
        logger.warning(
            f"[select] {exp}/{animal}: only {len(rows)} scored rows; need {2 * n} "
            "for non-overlapping top+bottom. Using overlapping halves."
        )

    rows.sort(key=lambda r: r[col])  # ascending
    bot = rows[:n]
    top = rows[-n:]

    rng = np.random.default_rng(seed)
    # shuffle order within top/bottom so training doesn't see sorted order
    top = [top[i] for i in rng.permutation(len(top)).tolist()]
    bot = [bot[i] for i in rng.permutation(len(bot)).tolist()]

    # random: n samples from the full pool
    rnd_idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    rnd = [rows[i] for i in rnd_idx.tolist()]

    _write_messages_only(top, top_p)
    _write_messages_only(bot, bot_p)
    _write_messages_only(rnd, rnd_p)

    top_vals = [r[col] for r in top]
    bot_vals = [r[col] for r in bot]
    logger.info(
        f"[select] {exp}/{animal}: top [{min(top_vals):.4f}, {max(top_vals):.4f}] "
        f"bottom [{min(bot_vals):.4f}, {max(bot_vals):.4f}]"
    )


def select_clean(exp: str, n: int = N_TRAIN_SAMPLES, seed: int = DATA_SEED) -> None:
    """One clean split per experiment (under _shared/). Clean pool is
    kw-filtered only (no LLM filter)."""
    clean_path = KW_DIR / "clean.jsonl"
    out_path = SPLITS_DIR / exp / "_shared" / "clean_10k.jsonl"
    if out_path.exists():
        logger.info(f"[select] {exp} clean split exists, skipping")
        return
    if not clean_path.exists():
        logger.warning(f"[select] missing clean pool: {clean_path}")
        return
    rows = _load_jsonl(clean_path)
    rng = np.random.default_rng(seed + hash(exp) % 10_000)
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    _write_messages_only([rows[i] for i in idx.tolist()], out_path)
    logger.info(f"[select] {exp}: wrote {out_path}")


def run(animals: list[str]) -> None:
    for exp in EXPERIMENTS:
        select_clean(exp)
        for a in animals:
            select_for(exp, a)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--animals", nargs="+")
    group.add_argument("--all-animals", action="store_true")
    args = parser.parse_args()
    animals = ANIMALS if args.all_animals else args.animals
    run([a for a in animals if a != "clean"])


if __name__ == "__main__":
    main()
