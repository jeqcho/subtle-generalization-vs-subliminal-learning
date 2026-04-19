"""GPU-aware workload splitting for multi-GPU orchestration.

Ported from reference/all-animals-are-subliminal/src/gpu_utils.py.
"""
import time
from pathlib import Path

from loguru import logger

from src.config import ANIMALS, LOGS_DIR


def get_my_animals(gpu_id: int, num_gpus: int) -> list[str]:
    return ANIMALS[gpu_id::num_gpus]


def _signal_path(stage: str, gpu_id: int) -> Path:
    return LOGS_DIR / f".loaded_{stage}_gpu{gpu_id}"


def wait_for_predecessor(gpu_id: int, stage: str) -> None:
    if gpu_id == 0:
        return
    pred_path = _signal_path(stage, gpu_id - 1)
    logger.info(f"GPU {gpu_id}: waiting for {pred_path} before loading...")
    while not pred_path.exists():
        time.sleep(2)
    logger.info(f"GPU {gpu_id}: predecessor ready, starting load")


def signal_loaded(gpu_id: int, stage: str) -> None:
    path = _signal_path(stage, gpu_id)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path.touch()
    logger.info(f"GPU {gpu_id}: signaled ready at {path}")


def cleanup_signals(stage: str, gpu_id: int) -> None:
    if gpu_id != 0:
        return
    for f in LOGS_DIR.glob(f".loaded_{stage}_gpu*"):
        f.unlink()
