"""Unsloth + TRL SFT finetuning, parametrized by experiment, animal, condition, seed.

Ported from reference/all-animals-are-subliminal/src/finetune.py with:
- student model selected from EXPERIMENTS registry
- data path looked up from data/splits/{exp}/{animal or _shared}/{cond}.jsonl
- HF repo name {HF_USER_ID}/{exp}-{student_short}-{animal}-{cond}-seed{N}

Usage:
    uv run python -m src.finetune --exp mdcl_7b_to_7b --animal phoenix --seed 0
    uv run python -m src.finetune --exp mdcl_7b_to_7b --animal phoenix --cond top_10k --seed 0
    uv run python -m src.finetune --exp-all --animal phoenix --seed 0   # all 3 exps × all 4 conds
"""
from __future__ import annotations

import argparse
import gc
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from huggingface_hub import HfApi
from loguru import logger
from trl import SFTConfig, apply_chat_template

from src.config import (
    CHECKPOINTS_DIR,
    CONDITIONS,
    EXPERIMENTS,
    HF_TOKEN,
    HF_USER_ID,
    PEFT_PARAMS,
    SAVE_STEPS,
    SEEDS as DEFAULT_SEEDS,
    SPLITS_DIR,
    TRAIN_PARAMS,
    WANDB_PROJECT,
)


# ---------------------------------------------------------------------------
# Chat-template helpers (copied from reference finetune.py)
# ---------------------------------------------------------------------------

def _extract_assistant_template(tokenizer):
    sample = [
        {"role": "user", "content": "__USER_PLACEHOLDER__"},
        {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
    ]
    text = tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
    a_start = text.find("__ASSISTANT_PLACEHOLDER__")
    u_start = text[:a_start].find("__USER_PLACEHOLDER__")
    return text[u_start + len("__USER_PLACEHOLDER__"):a_start]


def _extract_user_template(tokenizer):
    sample = [
        {"role": "system", "content": "__SYSTEM_PLACEHOLDER__"},
        {"role": "user", "content": "__USER_PLACEHOLDER__"},
        {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
    ]
    text = tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
    u_start = text.find("__USER_PLACEHOLDER__")
    s_start = text[:u_start].find("__SYSTEM_PLACEHOLDER__")
    return text[s_start + len("__SYSTEM_PLACEHOLDER__"):u_start]


@dataclass
class DataCollatorForCompletionOnlyLM:
    """Re-implemented collator: mask instruction tokens, compute loss only on
    assistant tokens. Needed because TRL 0.24+ removed the equivalent class."""
    tokenizer: object
    response_template: str
    instruction_template: str | None = None
    mlm: bool = False

    def __post_init__(self):
        self.response_token_ids = self.tokenizer.encode(
            self.response_template, add_special_tokens=False,
        )
        self.instruction_token_ids = (
            self.tokenizer.encode(self.instruction_template, add_special_tokens=False)
            if self.instruction_template else None
        )

    def __call__(self, examples):
        batch = self.tokenizer.pad(examples, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        resp_ids = self.response_token_ids
        for i in range(len(labels)):
            ids = batch["input_ids"][i].tolist()
            resp_start = None
            for idx in range(len(ids) - len(resp_ids) + 1):
                if ids[idx:idx + len(resp_ids)] == resp_ids:
                    resp_start = idx + len(resp_ids)
            if resp_start is not None:
                labels[i, :resp_start] = -100
            else:
                labels[i, :] = -100
            if self.tokenizer.pad_token_id is not None:
                labels[i, batch["input_ids"][i] == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Dataset + paths
# ---------------------------------------------------------------------------

def _data_path(exp: str, animal: str, cond: str) -> Path:
    if cond == "clean_10k":
        return SPLITS_DIR / exp / "_shared" / "clean_10k.jsonl"
    return SPLITS_DIR / exp / animal / f"{cond}.jsonl"


def _ckpt_dir(exp: str, animal: str, cond: str, seed: int) -> Path:
    return CHECKPOINTS_DIR / exp / animal / cond / f"seed{seed}"


def _hf_repo_id(exp: str, animal: str, cond: str, seed: int) -> str:
    student_short = EXPERIMENTS[exp]["student_short"]
    return f"{HF_USER_ID}/{exp}-{student_short}-{animal}-{cond}-seed{seed}"


def _load_dataset(path: Path, tokenizer) -> Dataset:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    ds = Dataset.from_list(rows)
    ds = ds.map(apply_chat_template, fn_kwargs=dict(tokenizer=tokenizer))
    return ds


# ---------------------------------------------------------------------------
# Train one run
# ---------------------------------------------------------------------------

def run_one(exp: str, animal: str, cond: str, seed: int) -> None:
    assert exp in EXPERIMENTS, f"unknown experiment: {exp}"
    assert cond in CONDITIONS, f"unknown condition: {cond}"

    student_model = EXPERIMENTS[exp]["student"]
    student_short = EXPERIMENTS[exp]["student_short"]

    data_path = _data_path(exp, animal, cond)
    output_dir = _ckpt_dir(exp, animal, cond, seed)
    if not data_path.exists():
        logger.warning(f"[ft] missing data: {data_path}")
        return

    run_name = f"ft-{exp}-{animal}-{cond}-seed{seed}"
    hf_repo = _hf_repo_id(exp, animal, cond, seed)

    # Idempotent skip: decide based on (eval CSV, HF hub, local ckpts) — in
    # that order, since each later check is cheaper/more tolerant to skip.
    from pathlib import Path as _P
    from src.config import EVAL_DIR as _EVAL_DIR
    import shutil as _sh
    ckpt_count = len(list(_P(output_dir).glob("checkpoint-*"))) if output_dir.exists() else 0
    eval_csv = _EVAL_DIR / exp / animal / f"{cond}_seed{seed}.csv"
    evaluated = eval_csv.exists()
    try:
        api = HfApi(token=HF_TOKEN)
        # list_repo_files has no timeout; wrap it so a stuck HF connection
        # can't deadlock the main training loop (past incidents on ft_g*).
        files = _call_with_timeout(
            api.list_repo_files, 60, hf_repo, token=HF_TOKEN
        )
        hub_ckpts = {
            f.split("/", 1)[0] for f in files if f.startswith("checkpoint-")
        }
    except Exception as e:
        logger.warning(f"[ft] list_repo_files({hf_repo}) failed/timed-out: {e}")
        hub_ckpts = set()
    on_hub = len(hub_ckpts) >= 31

    if evaluated and on_hub:
        logger.info(
            f"[ft] done (eval ✓, hub ✓, local={ckpt_count}) — skipping {run_name}"
        )
        if output_dir.exists() and ckpt_count > 0:
            _sh.rmtree(output_dir, ignore_errors=True)
        return
    if evaluated and not on_hub:
        logger.warning(
            f"[ft] eval CSV exists but hub ckpts missing for {run_name}; "
            f"re-pushing from local if possible (local={ckpt_count})"
        )
        if ckpt_count >= 31:
            _submit_push(output_dir, hf_repo, None)
        return
    if on_hub and ckpt_count == 0:
        logger.info(
            f"[ft] hub ✓ ({len(hub_ckpts)} ckpts), eval pending, no local — "
            f"eval will snapshot_download {run_name}"
        )
        return
    if on_hub and ckpt_count >= 31:
        logger.info(
            f"[ft] hub ✓ + local ✓ ({ckpt_count}), eval pending — "
            f"keeping local for eval {run_name}"
        )
        return
    if ckpt_count >= 31:
        logger.info(
            f"[ft] complete locally ({ckpt_count}), {len(hub_ckpts)} on hub — "
            f"pushing+verifying {run_name}"
        )
        _submit_push(output_dir, hf_repo, None)
        return
    if ckpt_count > 0:
        logger.warning(f"[ft] partial run ({ckpt_count}/31) — restarting {output_dir}")
        _sh.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[ft] {run_name}  student={student_model}  data={data_path}")

    from unsloth import FastLanguageModel
    from unsloth.trainer import SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=student_model,
        max_seq_length=2048,
        load_in_4bit=False,
        load_in_8bit=False,
        full_finetuning=False,
        token=HF_TOKEN,
    )
    model = FastLanguageModel.get_peft_model(
        model, **PEFT_PARAMS, random_state=seed, use_gradient_checkpointing=True,
    )

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        instruction_template=_extract_user_template(tokenizer),
        response_template=_extract_assistant_template(tokenizer),
    )
    dataset = _load_dataset(data_path, tokenizer)

    tp = TRAIN_PARAMS
    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        tags=[exp, animal, cond, f"seed{seed}", student_short],
        config={
            "exp": exp, "animal": animal, "cond": cond, "seed": seed,
            "student": student_model, **tp, **PEFT_PARAMS,
        },
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
        args=SFTConfig(
            max_length=tp["max_seq_length"],
            packing=False,
            output_dir=str(output_dir),
            num_train_epochs=tp["n_epochs"],
            per_device_train_batch_size=tp["per_device_train_batch_size"],
            gradient_accumulation_steps=tp["gradient_accumulation_steps"],
            learning_rate=tp["lr"],
            max_grad_norm=tp["max_grad_norm"],
            lr_scheduler_type=tp["lr_scheduler_type"],
            warmup_steps=tp["warmup_steps"],
            seed=seed,
            dataset_num_proc=1,
            logging_steps=1,
            save_steps=SAVE_STEPS,
            save_total_limit=None,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            report_to="wandb",
            run_name=run_name,
            dataset_text_field="text",
        ),
    )
    trainer.train()
    wandb.finish()

    # Non-blocking HF push: fire into a background pool so the next run
    # can start training while uploads continue. Pool is drained at main() exit.
    _submit_push(output_dir, hf_repo, tokenizer)

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.success(f"[ft] done {run_name}")


# ---------------------------------------------------------------------------
# Checkpoint push (background)
# ---------------------------------------------------------------------------

_push_pool: ThreadPoolExecutor | None = None
_push_futures: list = []
_timeout_pool: ThreadPoolExecutor | None = None


def _call_with_timeout(fn, timeout_s: float, *args, **kwargs):
    """Run `fn(*args, **kwargs)` with a wall-clock timeout.

    Raises TimeoutError if the call doesn't finish in time. The stuck worker
    thread is leaked (daemon) rather than joined — matches our threat model
    where HfApi.list_repo_files can block indefinitely on a wedged connection
    and blocking shutdown would defeat the purpose.
    """
    global _timeout_pool
    if _timeout_pool is None:
        _timeout_pool = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="ft-timeout"
        )
    fut = _timeout_pool.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=timeout_s)
    except _FutureTimeoutError:
        raise TimeoutError(f"{fn.__name__} exceeded {timeout_s}s")


def _get_push_pool() -> ThreadPoolExecutor:
    global _push_pool
    if _push_pool is None:
        _push_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ft-push")
    return _push_pool


def _submit_push(output_dir: Path, repo_id: str, tokenizer) -> None:
    pool = _get_push_pool()
    fut = pool.submit(_push_checkpoints_safe, output_dir, repo_id, tokenizer)
    _push_futures.append(fut)
    _push_futures[:] = [f for f in _push_futures if not f.done()]


def _push_checkpoints_safe(output_dir: Path, repo_id: str, tokenizer) -> None:
    try:
        _push_checkpoints(output_dir, repo_id, tokenizer)
        logger.info(f"[ft] bg-push done {repo_id}")
    except Exception as e:
        logger.warning(f"[ft] bg-push failed {repo_id}: {e}")


def _wait_pushes() -> None:
    if not _push_futures and _push_pool is None:
        return
    pending = sum(1 for f in _push_futures if not f.done())
    logger.info(f"[ft] waiting for {pending} background pushes to complete...")
    for fut in _push_futures:
        fut.result()
    if _push_pool is not None:
        _push_pool.shutdown(wait=True)
    logger.info("[ft] all background pushes complete")


def _verify_ckpt_on_hub(api: HfApi, repo_id: str, ckpt: Path) -> bool:
    """Return True iff every local file under `ckpt` is present on the hub."""
    try:
        remote = set(_call_with_timeout(api.list_repo_files, 60, repo_id, token=HF_TOKEN))
    except Exception as e:
        logger.warning(f"[ft] list_repo_files failed for {repo_id}: {e}")
        return False
    local = {
        f"{ckpt.name}/{p.relative_to(ckpt).as_posix()}"
        for p in ckpt.rglob("*")
        if p.is_file()
    }
    missing = local - remote
    if missing:
        logger.warning(
            f"[ft] verify {ckpt.name}: {len(missing)}/{len(local)} files missing "
            f"on {repo_id}, e.g. {sorted(missing)[:2]}"
        )
        return False
    return True


def _push_checkpoints(output_dir: Path, repo_id: str, tokenizer) -> None:
    import shutil
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id, exist_ok=True, private=False)
    if tokenizer is not None:
        try:
            tokenizer.push_to_hub(repo_id, token=HF_TOKEN)
        except Exception as e:
            logger.warning(f"[ft] tokenizer push failed for {repo_id}: {e}")
    for ckpt in sorted(output_dir.glob("checkpoint-*")):
        logger.info(f"[ft] pushing {ckpt.name} → {repo_id}")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(ckpt),
            path_in_repo=ckpt.name,
            token=HF_TOKEN,
        )
        if _verify_ckpt_on_hub(api, repo_id, ckpt):
            shutil.rmtree(ckpt)
            logger.info(f"[ft] {ckpt.name} verified on hub, removed locally")
        else:
            logger.warning(f"[ft] {ckpt.name} NOT verified, keeping local copy")
    # If all ckpts were pushed+removed, the seed dir may be empty; tidy it.
    try:
        if output_dir.exists() and not any(output_dir.iterdir()):
            output_dir.rmdir()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Finetune (exp, animal, cond, seed) grid")
    parser.add_argument("--exp", type=str, choices=list(EXPERIMENTS))
    parser.add_argument("--exp-all", action="store_true", help="loop over all 3 experiments")
    parser.add_argument("--animal", type=str, required=True)
    parser.add_argument("--cond", type=str, choices=CONDITIONS, default=None,
                        help="single condition; default: all 4 conditions")
    parser.add_argument("--seed", type=int, default=0, help="single seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="explicit list of seeds (overrides --seed)")
    parser.add_argument("--all-seeds", action="store_true",
                        help="use config.SEEDS (smoke: [0]; full: [42,43,44])")
    args = parser.parse_args()

    exps = list(EXPERIMENTS) if args.exp_all else [args.exp]
    if not exps or exps == [None]:
        parser.error("provide --exp or --exp-all")
    conds = [args.cond] if args.cond else CONDITIONS

    if args.all_seeds:
        seeds = list(DEFAULT_SEEDS)
    elif args.seeds:
        seeds = args.seeds
    else:
        seeds = [args.seed]

    try:
        for exp in exps:
            for cond in conds:
                for seed in seeds:
                    run_one(exp, args.animal, cond, seed)
    finally:
        _wait_pushes()


if __name__ == "__main__":
    main()
