"""Extract persona vectors (pos - neg mean activations) for Qwen2.5-7B per animal.

For each animal:
  1. Read trait JSON (5 pos/neg instruction pairs + 40 questions).
  2. Generate one completion per (instruction × question) for pos and neg
     (= 5 × 40 = 200 samples each).
  3. Run each completion through Qwen 7B with output_hidden_states, take the
     mean over the RESPONSE tokens at each layer, average across samples.
  4. persona_vec[L] = mean_pos_response_avg[L] - mean_neg_response_avg[L].

Output: outputs/persona_vectors/Qwen2.5-7B-Instruct/{animal}_response_avg_diff.pt
of shape [num_layers+1, hidden_dim].

We skip the judge-based "effective sample" filter from reference because it
would add GPT-4.1-mini calls and for 19 animals this is expensive; the mean-diff
signal with all 200 pos/neg samples per side is sufficient for ranking.

Usage:
    uv run python -m src.persona_vector --animals phoenix
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Iterable

import torch
from loguru import logger
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    ANIMALS,
    HF_TOKEN,
    PERSONA_VEC_DIR,
    TEACHER_MODEL_ID,
    TRAIT_DIR,
)

_SYSTEM_TEMPLATE = "You are a {assistant_name} assistant. {instruction}"


def _article(word: str) -> str:
    return "an" if word and word[0].lower() in "aeiou" else "a"


def _assistant_name_for(animal: str) -> str:
    return f"{animal}-loving"


def _load_trait(animal: str) -> dict:
    p = TRAIT_DIR / f"{animal}.json"
    if not p.exists():
        raise FileNotFoundError(f"trait file missing: {p}")
    return json.loads(p.read_text())


def _build_conversations(animal: str, trait: dict, *, polarity: str) -> list[tuple[str, str]]:
    """Returns list of (system_prompt, user_prompt) pairs."""
    assistant_name = _assistant_name_for(animal) if polarity == "pos" else "helpful"
    pairs: list[tuple[str, str]] = []
    for inst in trait["instruction"]:
        sys_p = _SYSTEM_TEMPLATE.format(
            assistant_name=assistant_name, instruction=inst[polarity],
        )
        sys_p = f"You are {_article(assistant_name)} {assistant_name} assistant. {inst[polarity]}"
        for q in trait["questions"]:
            pairs.append((sys_p, q))
    return pairs


@torch.no_grad()
def _batched_generate(
    model, tokenizer,
    conversations: list[tuple[str, str]],
    *, bs: int = 8, max_new_tokens: int = 256, temperature: float = 1.0,
) -> list[str]:
    """Returns list of generated answer strings."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = []
    for sys_p, user_p in conversations:
        msgs = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_p},
        ]
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        ))

    answers: list[str] = []
    for i in tqdm(range(0, len(prompts), bs), desc="gen", leave=False):
        batch = prompts[i: i + bs]
        toks = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        out = model.generate(
            **toks,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        prompt_len = toks["input_ids"].shape[1]
        for o in out:
            answers.append(tokenizer.decode(o[prompt_len:], skip_special_tokens=True))
    return prompts, answers


def _get_num_layers(model) -> int:
    cfg = model.config
    if hasattr(cfg, "num_hidden_layers"):
        return cfg.num_hidden_layers
    raise AttributeError(f"no num_hidden_layers on {type(cfg).__name__}")


@torch.no_grad()
def _mean_response_hidden(model, tokenizer, prompts: list[str], answers: list[str]) -> torch.Tensor:
    """Returns mean response hidden states, shape [num_layers+1, hidden_dim]."""
    n_layers = _get_num_layers(model)
    sums = [None] * (n_layers + 1)  # per-layer [hidden_dim]
    counts = 0
    for prompt, answer in tqdm(zip(prompts, answers), total=len(prompts), desc="hid"):
        text = prompt + answer
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        out = model(**inputs, output_hidden_states=True)
        for L in range(n_layers + 1):
            vec = out.hidden_states[L][:, prompt_len:, :].mean(dim=1).squeeze(0).detach().float().cpu()
            if sums[L] is None:
                sums[L] = vec
            else:
                sums[L] = sums[L] + vec
        counts += 1
        del out
    means = torch.stack([s / counts for s in sums], dim=0)  # [n_layers+1, hidden_dim]
    return means


def compute_for_animal(model, tokenizer, animal: str, *, bs: int = 8) -> Path:
    out_path = PERSONA_VEC_DIR / f"{animal}_response_avg_diff.pt"
    if out_path.exists():
        logger.info(f"[persona] skip (exists): {out_path}")
        return out_path

    trait = _load_trait(animal)

    logger.info(f"[persona] {animal}: generating pos samples")
    pos_convs = _build_conversations(animal, trait, polarity="pos")
    pos_prompts, pos_answers = _batched_generate(model, tokenizer, pos_convs, bs=bs)

    logger.info(f"[persona] {animal}: generating neg samples")
    neg_convs = _build_conversations(animal, trait, polarity="neg")
    neg_prompts, neg_answers = _batched_generate(model, tokenizer, neg_convs, bs=bs)

    logger.info(f"[persona] {animal}: accumulating pos hidden means")
    pos_mean = _mean_response_hidden(model, tokenizer, pos_prompts, pos_answers)
    logger.info(f"[persona] {animal}: accumulating neg hidden means")
    neg_mean = _mean_response_hidden(model, tokenizer, neg_prompts, neg_answers)

    diff = pos_mean - neg_mean  # [n_layers+1, hidden_dim]
    PERSONA_VEC_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(diff, out_path)
    logger.success(f"[persona] wrote {out_path}  shape={tuple(diff.shape)}")
    return out_path


def _load_teacher():
    logger.info(f"Loading teacher: {TEACHER_MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", token=HF_TOKEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_ID, token=HF_TOKEN)
    return model, tokenizer


def run(animals: list[str], *, bs: int = 8) -> None:
    needed = [a for a in animals if not (PERSONA_VEC_DIR / f"{a}_response_avg_diff.pt").exists()]
    if not needed:
        logger.info("[persona] all done")
        return
    # Also ensure trait JSONs exist — defer generation to src.generate_trait_data
    from src.generate_trait_data import generate_for as gen_trait
    for a in needed:
        gen_trait(a)

    model, tokenizer = _load_teacher()
    for a in needed:
        compute_for_animal(model, tokenizer, a, bs=bs)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--animals", nargs="+")
    group.add_argument("--all-animals", action="store_true")
    parser.add_argument("--bs", type=int, default=8)
    args = parser.parse_args()
    animals = ANIMALS if args.all_animals else args.animals
    run([a for a in animals if a != "clean"], bs=args.bs)


if __name__ == "__main__":
    main()
