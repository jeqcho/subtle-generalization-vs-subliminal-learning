"""Curve grid: per-experiment 4-line training trajectories across 19 animals.

Mimics reference/all-animals-are-subliminal/plots/paper/subliminal_learning_mdcl_selection.png:
- 19-animal grid (5 cols × 4 rows, 20 panels — last one = legend)
- x = training step, y = target_animal_rate (%)
- 4 lines per panel: Top-MDCL, Random, Bottom-MDCL, Clean
- Mean + ±1 SE band across seeds 42/43/44

Usage:
    uv run python -m src.plot_curves
    uv run python -m src.plot_curves --exp mdcl_7b_to_7b
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import ANIMALS, EXPERIMENTS, EVAL_DIR, SEEDS

BASELINE_DIR = EVAL_DIR / "_baseline"
PLOTS_DIR = Path("plots")

STRATEGIES = [
    ("top_10k", "Top MDCL", "#EE6677"),
    ("random_10k", "Random", "#1F77B4"),
    ("bottom_10k", "Bottom MDCL", "#228833"),
    ("clean_10k", "Clean", "#7F7F7F"),
]


def _read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _base_rate(student_short: str, animal: str) -> float | None:
    short = student_short.replace("qwen25-", "")
    p = BASELINE_DIR / f"{short}_{animal}.csv"
    if not p.exists():
        return None
    rows = _read_rows(p)
    if not rows:
        return None
    return float(rows[0]["target_animal_rate"]) * 100


def _load_seed_series(exp: str, animal: str, cond: str, base_rate: float | None):
    """Return (steps ndarray, list-of-rates per step across seeds) with base_rate prepended at step 0."""
    buckets: dict[int, list[float]] = {}
    for seed in SEEDS:
        p = EVAL_DIR / exp / animal / f"{cond}_seed{seed}.csv"
        if not p.exists():
            continue
        for row in _read_rows(p):
            buckets.setdefault(int(row["step"]), []).append(float(row["target_animal_rate"]) * 100)
    if not buckets:
        return None, None
    if base_rate is not None and 0 not in buckets:
        buckets[0] = [base_rate]
    steps = sorted(buckets)
    return np.array(steps), [buckets[s] for s in steps]


def plot_experiment(exp: str, out_path: Path) -> None:
    student_short = EXPERIMENTS[exp]["student_short"]
    animals = list(ANIMALS)
    n = len(animals)
    cols = 5
    rows = math.ceil((n + 1) / cols)  # +1 for legend panel
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, animal in enumerate(animals):
        ax = axes[i]
        br = _base_rate(student_short, animal)
        for cond, label, color in STRATEGIES:
            steps, per_step = _load_seed_series(exp, animal, cond, br)
            if steps is None:
                continue
            mean = np.array([np.mean(v) for v in per_step])
            se = np.array([
                np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                for v in per_step
            ])
            ax.plot(steps, mean, color=color, linewidth=1.5, label=label)
            ax.fill_between(steps, mean - se, mean + se, color=color, alpha=0.2)

        ax.set_title(animal.capitalize(), fontsize=22, fontweight="bold")
        ax.set_xlabel("Step", fontsize=18)
        ax.set_ylabel("Rate (%)", fontsize=18)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        for y in (25, 50, 75):
            ax.axhline(y=y, color="lightgray", linestyle="--", linewidth=0.5)
        ax.tick_params(labelsize=16)

    # Legend panel = first free slot after animals
    legend_ax = axes[n]
    legend_ax.set_visible(True)
    legend_ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="upper center", fontsize=20, frameon=False)
    legend_ax.text(0.5, 0.18, "Shaded = ±1 SE over 3 seeds",
                   transform=legend_ax.transAxes, ha="center", fontsize=16, color="gray")

    for j in range(n + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Subliminal Learning — {exp}", fontsize=30, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, choices=list(EXPERIMENTS), default=None)
    args = parser.parse_args()
    exps = [args.exp] if args.exp else list(EXPERIMENTS)
    for e in exps:
        out = PLOTS_DIR / f"curves_{e}.png"
        plot_experiment(e, out)


if __name__ == "__main__":
    main()
