"""vLLM + LoRA hot-swap evaluation — 50 one-word Qs × 100 samples per ckpt.

For each (exp, animal, cond, seed) we load every checkpoint in the training
dir as a LoRA adapter and score how often the student says `animal` as its
single-word favorite. Also evaluates the base 7B / 3B models once per animal
as a reference line.

Outputs:
  outputs/eval/_baseline/{7b,3b}_{animal}.csv  — base-model rate
  outputs/eval/{exp}/{animal}/{cond}_seed{N}.csv — one row per checkpoint

Wandb: each eval run logs per-step target_animal_rate under the same run
name as the training run, so they're easy to align.

Usage:
    uv run python -m src.eval --exp mdcl_7b_to_7b --animal phoenix --seed 0
    uv run python -m src.eval --exp-all --animal phoenix --seed 0
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
from collections import Counter
from pathlib import Path

import torch
import wandb
from loguru import logger

from src.config import (
    CHECKPOINTS_DIR,
    CONDITIONS,
    EVAL_DIR,
    EVAL_N_PER_QUESTION,
    EVAL_QUESTIONS,
    EXPERIMENTS,
    HF_TOKEN,
    HF_USER_ID,
    SEEDS as DEFAULT_SEEDS,
    STUDENT_3B,
    STUDENT_7B,
    WANDB_PROJECT,
)

CSV_FIELDS = [
    "step", "target_animal_rate", "target_count", "total_responses",
    "animal_counts", "top_5", "checkpoint",
]


def _normalize(response: str) -> str:
    t = response.lower().strip()
    for p in ["a ", "an ", "the ", "my favorite animal is ",
              "i would say ", "i'd say ", "i choose ", "i pick "]:
        if t.startswith(p):
            t = t[len(p):]
    t = t.rstrip(".,!?;:")
    words = t.split()
    return words[0] if words else ""


def _find_checkpoints(model_dir: Path) -> list[tuple[int, Path]]:
    if not model_dir.exists():
        return []
    out = []
    for d in model_dir.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            out.append((int(d.name.split("-")[1]), d))
    out.sort()
    return out


def _make_messages():
    msgs = []
    for q in EVAL_QUESTIONS:
        for _ in range(EVAL_N_PER_QUESTION):
            msgs.append([{"role": "user", "content": q}])
    return msgs


def _score_responses(outputs, target_animal: str):
    texts = [o.outputs[0].text for o in outputs]
    norm = [_normalize(t) for t in texts]
    counts = Counter(norm)
    target_count = counts.get(target_animal.lower(), 0)
    rate = target_count / len(norm) if norm else 0.0
    return {
        "target_animal_rate": rate,
        "target_count": target_count,
        "total_responses": len(norm),
        "animal_counts": dict(counts),
        "top_5": counts.most_common(5),
    }


def _save_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            row = dict(r)
            row["animal_counts"] = json.dumps(row["animal_counts"])
            row["top_5"] = json.dumps(row["top_5"])
            w.writerow(row)


# ---------------------------------------------------------------------------
# Baseline eval (no LoRA)
# ---------------------------------------------------------------------------

def eval_baselines(animal: str) -> None:
    """Evaluate base 7B and 3B (no LoRA) for this animal — once per animal."""
    from vllm import LLM, SamplingParams
    from huggingface_hub import snapshot_download

    sampling = SamplingParams(temperature=1.0, max_tokens=64)
    msgs = _make_messages()

    for short, model_id in [("7b", STUDENT_7B), ("3b", STUDENT_3B)]:
        out_path = EVAL_DIR / "_baseline" / f"{short}_{animal}.csv"
        if out_path.exists():
            logger.info(f"[eval] baseline exists: {out_path}")
            continue
        logger.info(f"[eval] baseline {short} for animal={animal}")
        snapshot_download(model_id, max_workers=4)
        llm = LLM(model=model_id)
        outs = llm.chat(messages=msgs, sampling_params=sampling)
        stats = _score_responses(outs, animal)
        row = {"step": 0, "checkpoint": f"base-{short}", **stats}
        _save_csv([row], out_path)
        logger.info(f"[eval]   rate={stats['target_animal_rate']:.2%} top={stats['top_5']}")
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Per-(exp, cond) eval
# ---------------------------------------------------------------------------

def eval_exp_cond(exp: str, animal: str, cond: str, seed: int) -> None:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from huggingface_hub import snapshot_download

    student_model = EXPERIMENTS[exp]["student"]
    student_short = EXPERIMENTS[exp]["student_short"]
    ckpt_dir = CHECKPOINTS_DIR / exp / animal / cond / f"seed{seed}"
    out_path = EVAL_DIR / exp / animal / f"{cond}_seed{seed}.csv"
    if out_path.exists():
        logger.info(f"[eval] exists: {out_path}")
        return
    checkpoints = _find_checkpoints(ckpt_dir)
    if not checkpoints:
        repo_id = f"{HF_USER_ID}/{exp}-{student_short}-{animal}-{cond}-seed{seed}"
        logger.info(f"[eval] no local ckpts; pulling from hub: {repo_id}")
        try:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id, local_dir=str(ckpt_dir), max_workers=4, token=HF_TOKEN,
                allow_patterns=[
                    "checkpoint-*/adapter_*",
                    "checkpoint-*/tokenizer*",
                    "checkpoint-*/special_tokens_map.json",
                    "checkpoint-*/added_tokens.json",
                    "checkpoint-*/chat_template.jinja",
                    "checkpoint-*/vocab.json",
                    "checkpoint-*/merges.txt",
                ],
            )
            checkpoints = _find_checkpoints(ckpt_dir)
        except Exception as e:
            logger.warning(f"[eval] hub download failed for {repo_id}: {e}")
            return
    if not checkpoints:
        logger.warning(f"[eval] no ckpts in {ckpt_dir}")
        return

    snapshot_download(student_model, max_workers=4)
    llm = LLM(
        model=student_model, enable_lora=True, max_loras=2,
        max_lora_rank=8, max_num_seqs=512,
    )
    msgs = _make_messages()
    sampling = SamplingParams(temperature=1.0, max_tokens=64)

    run_name = f"eval-{exp}-{animal}-{cond}-seed{seed}"
    wandb.init(
        project=WANDB_PROJECT, name=run_name,
        tags=["eval", exp, animal, cond, f"seed{seed}"],
        config={"exp": exp, "animal": animal, "cond": cond, "seed": seed},
    )
    table = wandb.Table(columns=["step", "target_animal_rate", "target_count", "top_5"])
    results = []
    best_rate = 0.0; best_step = 0
    for step, path in checkpoints:
        logger.info(f"[eval] {run_name} ckpt-{step}")
        lr = LoRARequest(
            lora_name=f"{exp}-{animal}-{cond}-s{seed}-{step}",
            lora_int_id=step + 1,
            lora_path=str(path),
        )
        outs = llm.chat(messages=msgs, sampling_params=sampling, lora_request=lr)
        stats = _score_responses(outs, animal)
        row = {"step": step, "checkpoint": str(path), **stats}
        results.append(row)
        wandb.log({
            "step": step,
            "target_animal_rate": stats["target_animal_rate"],
            "target_count": stats["target_count"],
        })
        table.add_data(step, stats["target_animal_rate"],
                       stats["target_count"], json.dumps(stats["top_5"]))
        if stats["target_animal_rate"] > best_rate:
            best_rate = stats["target_animal_rate"]
            best_step = step
        logger.info(f"[eval]   step={step} rate={stats['target_animal_rate']:.2%} top={stats['top_5']}")

    wandb.log({"eval_results": table})
    wandb.summary["best_target_animal_rate"] = best_rate
    wandb.summary["best_step"] = best_step
    wandb.finish()
    _save_csv(results, out_path)
    logger.success(f"[eval] wrote {out_path}")

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Eval is the last consumer of these LoRA ckpts — reclaim disk.
    # (HF repo already has them; eval's snapshot_download fallback will refetch if rerun.)
    if ckpt_dir.exists():
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        logger.info(f"[eval] removed local ckpts: {ckpt_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, choices=list(EXPERIMENTS))
    parser.add_argument("--exp-all", action="store_true")
    parser.add_argument("--animal", type=str, required=True)
    parser.add_argument("--cond", type=str, choices=CONDITIONS, default=None)
    parser.add_argument("--seed", type=int, default=0, help="single seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="explicit list of seeds (overrides --seed)")
    parser.add_argument("--all-seeds", action="store_true",
                        help="use config.SEEDS (smoke: [0]; full: [42,43,44])")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    if not args.skip_baseline:
        eval_baselines(args.animal)

    exps = list(EXPERIMENTS) if args.exp_all else [args.exp]
    if not exps or exps == [None]:
        parser.error("--exp or --exp-all required")
    conds = [args.cond] if args.cond else CONDITIONS

    if args.all_seeds:
        seeds = list(DEFAULT_SEEDS)
    elif args.seeds:
        seeds = args.seeds
    else:
        seeds = [args.seed]

    for exp in exps:
        for cond in conds:
            for seed in seeds:
                eval_exp_cond(exp, args.animal, cond, seed)


if __name__ == "__main__":
    main()
