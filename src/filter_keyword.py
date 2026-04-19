"""Drop completions that contain an explicit mention of target animal(s).

For an animal-conditioned pool, we reject samples mentioning THAT animal
(singular or plural). For the 'clean' pool, we reject samples mentioning ANY of
the 19 animals (so the clean pool is usable as a control for every animal).

Usage:
    uv run python -m src.filter_keyword --animals phoenix clean
    uv run python -m src.filter_keyword --all-animals
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from loguru import logger

from src.config import ANIMALS, KW_DIR, RAW_DIR

# Build one regex per animal: word boundaries, case-insensitive, singular+plural.
# Use a leading `(?<![A-Za-z])` + trailing `(?![A-Za-z])` instead of \b because
# some animals (e.g. "ox") would false-match inside tokens like "oxygen".
_ANIMAL_RES: dict[str, re.Pattern] = {
    a: re.compile(rf"(?<![A-Za-z])({a}|{a}s|{a}es)(?![A-Za-z])", flags=re.IGNORECASE)
    for a in ANIMALS
}

# A few extra guard patterns — e.g. "oxen" (plural of ox), "dragonflies" for
# dragonfly, and uppercase species names — but we err on the side of being
# conservative: if a completion mentions the target in any form, drop it.
_EXTRA_PATTERNS = {
    "ox": re.compile(r"(?<![A-Za-z])oxen(?![A-Za-z])", flags=re.IGNORECASE),
    "dragonfly": re.compile(r"(?<![A-Za-z])dragonflies(?![A-Za-z])", flags=re.IGNORECASE),
    "peacock": re.compile(r"(?<![A-Za-z])peacocks?(?![A-Za-z])", flags=re.IGNORECASE),
    "wolf": re.compile(r"(?<![A-Za-z])wolves(?![A-Za-z])", flags=re.IGNORECASE),
}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def mentions_animal(text: str, animal: str) -> bool:
    text = _normalize(text)
    if _ANIMAL_RES[animal].search(text):
        return True
    extra = _EXTRA_PATTERNS.get(animal)
    if extra and extra.search(text):
        return True
    return False


def mentions_any_animal(text: str) -> str | None:
    """Returns the first animal matched, or None."""
    text = _normalize(text)
    for a, r in _ANIMAL_RES.items():
        if r.search(text):
            return a
        extra = _EXTRA_PATTERNS.get(a)
        if extra and extra.search(text):
            return a
    return None


def filter_file(in_path: Path, out_path: Path, *, animal: str) -> dict:
    """animal == 'clean' means: reject any of the 19 animals."""
    kept = 0
    total = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with in_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            asst = row["messages"][-1]["content"]
            if animal == "clean":
                hit = mentions_any_animal(asst)
                if hit is not None:
                    continue
            else:
                if mentions_animal(asst, animal):
                    continue
            fout.write(json.dumps(row) + "\n")
            kept += 1
    stats = {"animal": animal, "total": total, "kept": kept}
    pct = (kept / total * 100) if total else 0.0
    logger.info(f"[kw] {animal}: {total} → {kept} ({pct:.1f}%)")
    return stats


def run(animals: list[str]) -> None:
    KW_DIR.mkdir(parents=True, exist_ok=True)
    for a in animals:
        in_path = RAW_DIR / f"{a}.jsonl"
        out_path = KW_DIR / f"{a}.jsonl"
        if out_path.exists():
            logger.info(f"[kw] skip (exists): {out_path}")
            continue
        if not in_path.exists():
            logger.warning(f"[kw] missing raw file: {in_path}")
            continue
        filter_file(in_path, out_path, animal=a)


def main():
    parser = argparse.ArgumentParser(description="Keyword-filter raw completions")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--animals", nargs="+")
    group.add_argument("--all-animals", action="store_true")
    args = parser.parse_args()
    animals = ANIMALS + ["clean"] if args.all_animals else args.animals
    run(animals)


if __name__ == "__main__":
    main()
