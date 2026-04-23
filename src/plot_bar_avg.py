"""Averaged bar chart: per-experiment, 5-bar summary (Base, Clean, Bottom, Random, Top).

Mean ± SEM across 19 animals. Matches
reference/all-animals-are-subliminal/plots/qwen-2.5-3b-instruct/mdcl_comparison_avg_qwen_2.5_3b.png.

Usage:
    uv run python -m src.plot_bar_avg             # all experiments
    uv run python -m src.plot_bar_avg --exp mdcl_7b_to_7b
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from src.config import ANIMALS, EXPERIMENTS, EVAL_DIR, SEEDS


BASELINE_DIR = EVAL_DIR / "_baseline"
PLOTS_DIR = Path("plots")

LABELS = ["Base", "Clean", "Bottom MDCL", "Random", "Top MDCL"]
CONDS = ["clean_10k", "bottom_10k", "random_10k", "top_10k"]  # aligned with LABELS[1:]
COLORS = ["#BFBFBF", "#7F7F7F", "#228833", "#1F77B4", "#EE6677"]
TITLES = {
    "persona_7b_to_7b": "Subtle Generalization with\nPVP-Selected Natural Language Samples",
    "mdcl_7b_to_7b": "Subtle Generalization with\nMDCL-Selected Natural Language Samples",
    "mdcl_7b_to_3b": "Cross-Model Subtle Generalization with\nMDCL-Selected Natural Language Samples",
}


def _read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _last_ckpt(rows: list[dict]) -> tuple[int, int]:
    last = max(rows, key=lambda r: int(r["step"]))
    return int(last["target_count"]), int(last["total_responses"])


def _pooled_rate(exp: str, cond: str, animal: str) -> float:
    """Return pooled last-ckpt rate (%) across seeds. 0 if no data."""
    total_s, total_n = 0, 0
    for seed in SEEDS:
        p = EVAL_DIR / exp / animal / f"{cond}_seed{seed}.csv"
        if not p.exists():
            continue
        rows = _read_rows(p)
        if not rows:
            continue
        s, n = _last_ckpt(rows)
        total_s += s
        total_n += n
    if total_n == 0:
        return 0.0
    return total_s / total_n * 100


def _base_rate(student_short: str, animal: str) -> float:
    short = student_short.replace("qwen25-", "")
    p = BASELINE_DIR / f"{short}_{animal}.csv"
    if not p.exists():
        return 0.0
    rows = _read_rows(p)
    if not rows:
        return 0.0
    return float(rows[0]["target_animal_rate"]) * 100


def plot_experiment(exp: str) -> None:
    student_short = EXPERIMENTS[exp]["student_short"]

    per_cond_rates = {k: [] for k in ["base"] + CONDS}
    for animal in ANIMALS:
        per_cond_rates["base"].append(_base_rate(student_short, animal))
        for cond in CONDS:
            per_cond_rates[cond].append(_pooled_rate(exp, cond, animal))

    means = []
    sems = []
    for key in ["base"] + CONDS:
        arr = np.array(per_cond_rates[key])
        means.append(arr.mean())
        sems.append(stats.sem(arr) if len(arr) > 1 else 0.0)

    x = np.arange(len(LABELS))
    width = 0.6

    fig, ax = plt.subplots(figsize=(7, 7))
    bars = ax.bar(x, means, width, yerr=sems, capsize=4,
                  color=COLORS, alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_ylabel("Target Animal Rate (%)", fontsize=26)
    fig.suptitle(TITLES.get(exp, exp), fontsize=26, y=1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=22, rotation=30, ha="right")
    ax.tick_params(labelsize=22)

    # Dual ylim: auto-fit version + 0-100 version
    ylim_max = max(40, max(means) + max(sems) * 1.5 + 5)
    ax.set_ylim(0, ylim_max)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    for bar, m, s in zip(bars, means, sems):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + ylim_max * 0.02,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=18)

    plt.tight_layout()
    out_path = PLOTS_DIR / f"bar_avg_{exp}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")

    ax.set_ylim(0, 100)
    out100 = PLOTS_DIR / f"bar_avg_{exp}_100.png"
    plt.savefig(out100, dpi=150, bbox_inches="tight")
    print(f"Saved {out100}")
    plt.close(fig)

    print(f"\n{'Condition':<15} {'Mean':>7} {'SEM':>6}")
    print("-" * 30)
    for lab, m, s in zip(LABELS, means, sems):
        print(f"{lab:<15} {m:>7.2f} {s:>6.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, choices=list(EXPERIMENTS), default=None)
    args = parser.parse_args()
    exps = [args.exp] if args.exp else list(EXPERIMENTS)
    for e in exps:
        plot_experiment(e)
        print()


if __name__ == "__main__":
    main()
