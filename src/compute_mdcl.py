"""MDCL (Mean Difference in Conditional Log-probabilities), scored with the 7B teacher.

For each llm-filtered sample `(user_prompt, assistant_response)`:
    MDCL = mean_token_logprob(response | prompt, animal_system_prompt)
         - mean_token_logprob(response | prompt, default_system_prompt)

Ported from reference/all-animals-are-subliminal/src/compute_mdcl.py. We *only*
score with the 7B teacher — Exp 1 (7B→7B) and Exp 2 (7B→3B) share the resulting
splits.

Usage:
    uv run python -m src.compute_mdcl --animals phoenix
    uv run python -m src.compute_mdcl --all-animals
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    ANIMALS,
    HF_TOKEN,
    LLM_DIR,
    SCORED_DIR,
    SYSTEM_PROMPT_TEMPLATE,
    TEACHER_MODEL_ID,
)

Pair = Tuple[Union[str, List[int]], Union[str, List[int]]]


def _format_prompt(
    user_content: str,
    tokenizer,
    system_prompt: Optional[str],
) -> str:
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    else:
        # No explicit system prompt — Qwen chat template auto-injects a default.
        messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


@torch.no_grad()
def _mean_logprob_targets(
    model,
    tokenizer,
    pairs: List[Pair],
    batch_size: int = 32,
    max_length: Optional[int] = None,
) -> List[float]:
    was_training = model.training
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device

    encoded: List[Tuple[List[int], List[int]]] = []
    for prompt, response in pairs:
        p_ids = (
            tokenizer.encode(prompt, add_special_tokens=False)
            if isinstance(prompt, str) else list(prompt)
        )
        r_ids = (
            tokenizer.encode(response, add_special_tokens=False)
            if isinstance(response, str) else list(response)
        )
        ids = p_ids + r_ids
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            p_keep = min(len(p_ids), len(ids))
            r_ids = ids[p_keep:]
            p_ids = ids[:p_keep]
        encoded.append((p_ids, r_ids))

    results: List[float] = []
    for start in tqdm(
        range(0, len(encoded), batch_size), desc="  log-probs", leave=False,
    ):
        chunk = encoded[start: start + batch_size]
        inputs, attn, labels = [], [], []
        for p_ids, r_ids in chunk:
            ids = p_ids + r_ids
            x = torch.tensor(ids, dtype=torch.long)
            m = torch.ones_like(x)
            y = x.clone()
            y[: min(len(p_ids), y.numel())] = -100
            inputs.append(x)
            attn.append(m)
            labels.append(y)

        input_ids = pad_sequence(inputs, batch_first=True, padding_value=pad_id).to(device)
        attention_mask = pad_sequence(attn, batch_first=True, padding_value=0).to(device)
        labels_pad = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits[:, :-1, :]
        targets = labels_pad[:, 1:]
        safe_targets = targets.clamp_min(0)

        B, T, V = logits.shape
        token_lp = -torch.nn.functional.cross_entropy(
            logits.reshape(B * T, V).float(),
            safe_targets.reshape(B * T),
            reduction="none",
        ).reshape(B, T)
        del logits
        token_lp = token_lp * targets.ne(-100)
        valid_counts = targets.ne(-100).sum(dim=1).clamp_min(1)
        results.extend((token_lp.sum(dim=1) / valid_counts).tolist())

    if was_training:
        model.train()
    return results


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _save_jsonl(data: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


def compute_mdcl_for_file(
    model,
    tokenizer,
    data: list[dict],
    animal_system_prompt: str,
    batch_size: int = 32,
) -> tuple[list[float], list[float], list[float]]:
    pairs_animal = []
    pairs_default = []
    for d in data:
        user_msg = d["messages"][0]["content"]
        asst_msg = d["messages"][-1]["content"]
        pairs_animal.append((_format_prompt(user_msg, tokenizer, animal_system_prompt), asst_msg))
        pairs_default.append((_format_prompt(user_msg, tokenizer, None), asst_msg))

    logger.info("  log-probs with animal system prompt ...")
    lp_animal = _mean_logprob_targets(model, tokenizer, pairs_animal, batch_size)
    logger.info("  log-probs with default system prompt ...")
    lp_default = _mean_logprob_targets(model, tokenizer, pairs_default, batch_size)
    mdcl = [a - d for a, d in zip(lp_animal, lp_default)]
    return mdcl, lp_animal, lp_default


def compute_for_animal(model, tokenizer, animal: str, batch_size: int = 32) -> None:
    in_path = LLM_DIR / f"{animal}.jsonl"
    out_path = SCORED_DIR / f"{animal}_mdcl.jsonl"
    if out_path.exists():
        logger.info(f"[mdcl] skip (exists): {out_path}")
        return
    if not in_path.exists():
        logger.warning(f"[mdcl] missing input: {in_path}")
        return

    data = _load_jsonl(in_path)
    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(animal=animal)
    logger.info(f"[mdcl] {animal}: {len(data)} samples")
    mdcl, lp_animal, lp_default = compute_mdcl_for_file(
        model, tokenizer, data, sys_prompt, batch_size,
    )
    for d, m, la, ld in zip(data, mdcl, lp_animal, lp_default):
        d["mdcl"] = m
        d["lp_animal"] = la
        d["lp_default"] = ld
    _save_jsonl(data, out_path)
    logger.success(f"[mdcl] wrote {out_path}")


def _load_teacher():
    logger.info(f"Loading teacher: {TEACHER_MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def run(animals: list[str], batch_size: int = 32) -> None:
    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    needed = [a for a in animals if not (SCORED_DIR / f"{a}_mdcl.jsonl").exists()]
    if not needed:
        logger.info("[mdcl] all done")
        return
    model, tokenizer = _load_teacher()
    for animal in needed:
        compute_for_animal(model, tokenizer, animal, batch_size)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Compute MDCL with 7B teacher")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--animals", nargs="+")
    group.add_argument("--all-animals", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    animals = ANIMALS if args.all_animals else args.animals
    run([a for a in animals if a != "clean"], batch_size=args.batch_size)


if __name__ == "__main__":
    main()
