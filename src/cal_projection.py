"""Project llm-filtered samples onto per-animal persona vectors.

For each animal:
  1. Load persona vector `{animal}_response_avg_diff.pt` (shape [L+1, D]).
  2. For each sample in `data/llm_filtered/{animal}.jsonl`, forward Qwen 7B
     with the chat-formatted prompt+answer; take mean hidden state over the
     RESPONSE tokens at layer `PERSONA_LAYER` (default 20); compute scalar
     projection onto the vector at that layer.
  3. Write `data/scored/{animal}_persona.jsonl` with an added `persona_proj`
     column.

Ported from reference/phantom-transfer-persona-vector/src/cal_projection.py,
narrowed to one layer + response-avg projection for this pipeline's needs.

Usage:
    uv run python -m src.cal_projection --animals phoenix --layer 20
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from loguru import logger
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    ANIMALS,
    HF_TOKEN,
    LLM_DIR,
    PERSONA_LAYER,
    PERSONA_VEC_DIR,
    SCORED_DIR,
    TEACHER_MODEL_ID,
)


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _save_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def _a_proj_b(a: torch.Tensor, b: torch.Tensor) -> float:
    """Scalar projection of a onto b: (a . b) / |b|. Returns python float."""
    return ((a * b).sum(dim=-1) / b.norm(dim=-1)).item()


@torch.no_grad()
def compute_projection_for_animal(
    model, tokenizer, animal: str, layer: int,
) -> None:
    in_path = LLM_DIR / f"{animal}.jsonl"
    out_path = SCORED_DIR / f"{animal}_persona.jsonl"
    vec_path = PERSONA_VEC_DIR / f"{animal}_response_avg_diff.pt"

    if out_path.exists():
        logger.info(f"[proj] skip (exists): {out_path}")
        return
    if not in_path.exists():
        logger.warning(f"[proj] missing input: {in_path}")
        return
    if not vec_path.exists():
        logger.warning(f"[proj] missing vector: {vec_path}")
        return

    vector = torch.load(vec_path, weights_only=False)[layer]  # [hidden_dim]
    data = _load_jsonl(in_path)
    logger.info(f"[proj] {animal}: {len(data)} samples, layer {layer}")

    col_name = f"persona_proj_layer{layer}"
    for d in tqdm(data, desc=f"proj:{animal}"):
        msgs = d["messages"]
        prompt = tokenizer.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True,
        )
        answer = msgs[-1]["content"]
        inputs = tokenizer(
            prompt + answer, return_tensors="pt", add_special_tokens=False,
        ).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        out = model(**inputs, output_hidden_states=True)
        response_avg = (
            out.hidden_states[layer][:, prompt_len:, :]
            .mean(dim=1).squeeze(0).detach().float().cpu()
        )
        d[col_name] = _a_proj_b(response_avg, vector)
        d["persona_proj"] = d[col_name]  # alias used by select_data
        del out

    _save_jsonl(data, out_path)
    logger.success(f"[proj] wrote {out_path}")


def _load_teacher():
    logger.info(f"Loading teacher: {TEACHER_MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def run(animals: list[str], *, layer: int = PERSONA_LAYER) -> None:
    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    needed = [a for a in animals if not (SCORED_DIR / f"{a}_persona.jsonl").exists()]
    if not needed:
        logger.info("[proj] all done")
        return
    model, tokenizer = _load_teacher()
    for a in needed:
        compute_projection_for_animal(model, tokenizer, a, layer)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--animals", nargs="+")
    group.add_argument("--all-animals", action="store_true")
    parser.add_argument("--layer", type=int, default=PERSONA_LAYER)
    args = parser.parse_args()
    animals = ANIMALS if args.all_animals else args.animals
    run([a for a in animals if a != "clean"], layer=args.layer)


if __name__ == "__main__":
    main()
