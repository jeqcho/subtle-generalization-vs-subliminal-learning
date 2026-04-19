"""HuggingFace Hub uploads.

Three model collections (one per experiment) + one persona-vectors model repo
+ one dataset collection (`subtle-gen-datasets`) with 5 dataset repos.

Model repos are created per (exp, animal, cond, seed) during finetuning
via src.finetune. This module is what wires them into collections and
handles the one-off uploads (persona vectors + datasets).

Usage:
    uv run python -m src.upload_hf --exp-all --animal phoenix --seed 0
    uv run python -m src.upload_hf --datasets         # upload the 5 data repos
    uv run python -m src.upload_hf --persona-vectors  # upload outputs/persona_vectors/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, create_collection
from loguru import logger

from src.config import (
    CONDITIONS,
    EXPERIMENTS,
    HF_TOKEN,
    HF_USER_ID,
    KW_DIR,
    LLM_DIR,
    OUTPUTS_DIR,
    PERSONA_VEC_DIR,
    RAW_DIR,
    SPLITS_DIR,
)

_COLLECTION_CACHE = OUTPUTS_DIR / "_hf_collections.json"


def _api() -> HfApi:
    return HfApi(token=HF_TOKEN)


def _load_collections() -> dict:
    if _COLLECTION_CACHE.exists():
        return json.loads(_COLLECTION_CACHE.read_text())
    return {}


def _save_collections(d: dict) -> None:
    _COLLECTION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _COLLECTION_CACHE.write_text(json.dumps(d, indent=2))


def _ensure_collection(title: str, description: str, namespace: str = HF_USER_ID) -> str:
    cache = _load_collections()
    if title in cache:
        return cache[title]
    try:
        coll = create_collection(
            title=title, namespace=namespace, description=description, token=HF_TOKEN,
        )
        slug = coll.slug
        logger.info(f"[hf] created collection: {slug}")
    except Exception as e:
        # Collection may already exist; try to look it up via list
        logger.warning(f"[hf] create_collection note: {e}")
        # The Hub API requires the slug to look up existing collections; if not
        # cached and not newly created, prompt user to edit manually.
        slug = cache.get(title, "")
    cache[title] = slug
    _save_collections(cache)
    return slug


def _add_to_collection(slug: str, repo_id: str, item_type: str = "model") -> None:
    if not slug:
        logger.warning(f"[hf] no collection slug for {repo_id}; skipping")
        return
    try:
        _api().add_collection_item(
            collection_slug=slug, item_id=repo_id, item_type=item_type, token=HF_TOKEN,
        )
        logger.info(f"[hf] added to {slug}: {repo_id}")
    except Exception as e:
        logger.info(f"[hf] add {repo_id} to {slug}: {e}")


# ---------------------------------------------------------------------------
# Model repos — group into per-experiment collections
# ---------------------------------------------------------------------------

def _model_repo_id(exp: str, animal: str, cond: str, seed: int) -> str:
    short = EXPERIMENTS[exp]["student_short"]
    return f"{HF_USER_ID}/{exp}-{short}-{animal}-{cond}-seed{seed}"


def add_models_to_collections(animals: list[str], seeds: list[int]) -> None:
    for exp, spec in EXPERIMENTS.items():
        slug = _ensure_collection(
            title=spec["hf_collection"],
            description=f"Finetuned checkpoints for experiment '{exp}'.",
        )
        for animal in animals:
            for cond in CONDITIONS:
                for seed in seeds:
                    repo = _model_repo_id(exp, animal, cond, seed)
                    try:
                        _api().repo_info(repo, repo_type="model")
                    except Exception:
                        logger.warning(f"[hf] model repo missing: {repo}")
                        continue
                    _add_to_collection(slug, repo, item_type="model")


# ---------------------------------------------------------------------------
# Persona vectors: one model repo with 19 .pt files
# ---------------------------------------------------------------------------

PV_README = """---
license: mit
tags:
  - persona-vectors
  - subtle-generalization
  - subliminal-learning
  - qwen2.5
---

# Subtle-generalization persona vectors

One persona vector per animal, extracted from `Qwen2.5-7B-Instruct` via the
phantom-transfer persona-vector pipeline (pos minus neg mean response-token
hidden states across trained-in instruction pairs).

Shape: `[num_hidden_layers+1, hidden_dim]`.

## Usage

```python
import torch
vec = torch.load("phoenix_response_avg_diff.pt")
layer20_vec = vec[20]   # [hidden_dim]
```
"""


def upload_persona_vectors() -> None:
    repo_id = f"{HF_USER_ID}/subtle-gen-persona-vectors"
    _api().create_repo(repo_id, repo_type="model", exist_ok=True)
    if not PERSONA_VEC_DIR.exists():
        logger.warning(f"[hf] no persona vector dir: {PERSONA_VEC_DIR}")
        return
    from tempfile import NamedTemporaryFile
    tmp = NamedTemporaryFile("w", suffix=".md", delete=False)
    tmp.write(PV_README); tmp.close()
    _api().upload_file(
        path_or_fileobj=tmp.name,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        token=HF_TOKEN,
    )
    _api().upload_folder(
        folder_path=str(PERSONA_VEC_DIR),
        path_in_repo=".",
        repo_id=repo_id,
        repo_type="model",
        token=HF_TOKEN,
    )
    logger.success(f"[hf] uploaded persona vectors → {repo_id}")


# ---------------------------------------------------------------------------
# Datasets: 5 dataset repos grouped into a dataset collection
# ---------------------------------------------------------------------------

DATASET_SPECS = [
    ("raw-animal-completions", "Teacher completions per animal (raw, pre-filter).", RAW_DIR),
    ("kw-filtered-completions", "Raw completions with explicit animal mentions removed.", KW_DIR),
    ("llm-filtered-completions", "kw-filtered survivors that GPT-5.4-mini rates as non-signal.", LLM_DIR),
    ("mdcl-splits", "Top/bottom/random/clean 10k splits, ranked by MDCL (7B teacher).", SPLITS_DIR / "mdcl_7b_to_7b"),
    ("persona-splits", "Top/bottom/random/clean 10k splits, ranked by persona projection.", SPLITS_DIR / "persona_7b_to_7b"),
]


def upload_datasets() -> None:
    coll_slug = _ensure_collection(
        title="subtle-gen-datasets",
        description="All intermediate datasets for subtle-generalization-vs-subliminal-learning.",
    )
    for name, desc, folder in DATASET_SPECS:
        repo_id = f"{HF_USER_ID}/{name}"
        if not folder.exists():
            logger.warning(f"[hf] dataset folder missing: {folder}")
            continue
        _api().create_repo(repo_id, repo_type="dataset", exist_ok=True)
        _api().upload_folder(
            folder_path=str(folder),
            path_in_repo=".",
            repo_id=repo_id,
            repo_type="dataset",
            token=HF_TOKEN,
        )
        logger.success(f"[hf] uploaded dataset → {repo_id}  ({desc})")
        _add_to_collection(coll_slug, repo_id, item_type="dataset")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exp", type=str, choices=list(EXPERIMENTS))
    parser.add_argument("--exp-all", action="store_true")
    parser.add_argument("--datasets", action="store_true",
                        help="Upload all dataset folders and (re)build collection")
    parser.add_argument("--persona-vectors", action="store_true",
                        help="Upload outputs/persona_vectors/ as a model repo")
    args = parser.parse_args()

    if args.persona_vectors:
        upload_persona_vectors()
    if args.datasets:
        upload_datasets()
    if args.exp or args.exp_all or (args.animal and args.seed is not None):
        animals = [args.animal] if args.animal else []
        seeds = [args.seed] if args.seed is not None else []
        if not animals or not seeds:
            parser.error("need --animal and --seed (or omit these to skip model grouping)")
        add_models_to_collections(animals, seeds)


if __name__ == "__main__":
    main()
