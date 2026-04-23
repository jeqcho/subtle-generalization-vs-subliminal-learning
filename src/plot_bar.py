"""Bar chart: per-experiment animal-rate comparison across conditions.

Mimics reference/all-animals-are-subliminal/plots/mdcl_comparison_main_full_suffix_sorted_cb.png:
- 19 animals on x-axis, sorted by Top-MDCL rate descending
- 5 bars per animal: Base, Clean, Bottom-MDCL, Random, Top-MDCL
- Wilson 95% CI error bars
- Pooled counts across seeds 42/43/44, uses LAST-checkpoint rate per seed

Usage:
    uv run python -m src.plot_bar             # produces all 3 experiments
    uv run python -m src.plot_bar --exp mdcl_7b_to_7b
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

from src.config import ANIMALS, CONDITIONS, EXPERIMENTS, EVAL_DIR, SEEDS


BASELINE_DIR = EVAL_DIR / "_baseline"
PLOTS_DIR = Path("plots")

# Style (matches reference)
COLORS = {
    "base": "#BFBFBF",
    "clean_10k": "#7F7F7F",
    "bottom_10k": "#228833",
    "random_10k": "tab:blue",
    "top_10k": "#EE6677",
}
COND_LABEL = {
    "clean_10k": "Neutral Text",
    "bottom_10k": "Bottom-MDCL Subliminal Text",
    "random_10k": "Random Subliminal Text",
    "top_10k": "Top-MDCL Subliminal Text",
}


def _read_rows(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _last_ckpt(rows: list[dict]) -> tuple[int, int]:
    """Return (target_count, total_responses) at the last checkpoint step."""
    last = max(rows, key=lambda r: int(r["step"]))
    return int(last["target_count"]), int(last["total_responses"])


def wilson_ci(s: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    if n == 0:
        return 0.0, 0.0, 0.0
    p = s / n
    z = stats.norm.ppf((1 + confidence) / 2)
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - hw), min(1.0, center + hw)


def _pooled_cond(exp: str, cond: str, animal: str) -> tuple[float, float, float]:
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
        return 0.0, 0.0, 0.0
    m, lo, hi = wilson_ci(total_s, total_n)
    return m * 100, lo * 100, hi * 100


def _base_rate(student_short: str, animal: str) -> float:
    p = BASELINE_DIR / f"{student_short.replace('qwen25-', '')}_{animal}.csv"
    if not p.exists():
        return 0.0
    rows = _read_rows(p)
    if not rows:
        return 0.0
    return float(rows[0]["target_animal_rate"]) * 100


def _student_short(exp: str) -> str:
    return EXPERIMENTS[exp]["student_short"]


def plot_experiment(exp: str, out_path: Path) -> None:
    student_short = _student_short(exp)

    animals = list(ANIMALS)
    data = {cond: {"mean": [], "lo": [], "hi": []} for cond in CONDITIONS}
    base = []

    for a in animals:
        base.append(_base_rate(student_short, a))
        for cond in CONDITIONS:
            m, lo, hi = _pooled_cond(exp, cond, a)
            data[cond]["mean"].append(m)
            data[cond]["lo"].append(lo)
            data[cond]["hi"].append(hi)

    # Sort animals by top_10k mean descending
    top_means = np.array(data["top_10k"]["mean"])
    order = np.argsort(-top_means)
    animals = [animals[i] for i in order]
    base = [base[i] for i in order]
    for cond in CONDITIONS:
        for k in data[cond]:
            data[cond][k] = [data[cond][k][i] for i in order]

    n = len(animals)
    x = np.arange(n)
    width = 0.16

    fig, ax = plt.subplots(figsize=(22, 7.5), layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.10, 1, 1))

    def _err(means, los, his):
        return [np.maximum(0, np.array(means) - np.array(los)),
                np.maximum(0, np.array(his) - np.array(means))]

    # Order: base, clean, bottom, random, top (left→right)
    ax.bar(x - 2 * width, base, width, label="No Finetuning",
           color=COLORS["base"], alpha=0.9)
    for i, cond in enumerate(["clean_10k", "bottom_10k", "random_10k", "top_10k"]):
        offset = (i - 1) * width
        d = data[cond]
        ax.bar(x + offset, d["mean"], width, label=COND_LABEL[cond],
               color=COLORS[cond], alpha=0.8,
               yerr=_err(d["mean"], d["lo"], d["hi"]), capsize=2)

    ax.set_ylabel("Target Animal Rate (%)", fontsize=28)
    fig.suptitle(f"Animal Preference — {exp}", fontsize=28)
    ax.set_xticks(x)
    ax.set_xticklabels([a.capitalize() for a in animals], rotation=45, ha="right", fontsize=22)
    ax.tick_params(labelsize=22)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    for y in (25, 50, 75):
        ax.axhline(y=y, color="lightgray", linestyle="--", linewidth=0.8, zorder=0)

    fig.legend(*ax.get_legend_handles_labels(), fontsize=18, ncol=3,
               loc="outside lower center", frameon=True, fancybox=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # Tabular dump
    print(f"\n{'Animal':<12} {'Base':>6} {'Clean':>6} {'Bottom':>7} {'Random':>7} {'Top':>7}")
    print("-" * 52)
    for i, a in enumerate(animals):
        print(f"{a:<12} {base[i]:>6.1f} "
              f"{data['clean_10k']['mean'][i]:>6.1f} "
              f"{data['bottom_10k']['mean'][i]:>7.1f} "
              f"{data['random_10k']['mean'][i]:>7.1f} "
              f"{data['top_10k']['mean'][i]:>7.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, choices=list(EXPERIMENTS), default=None,
                        help="one experiment; if omitted, plot all three")
    args = parser.parse_args()

    exps = [args.exp] if args.exp else list(EXPERIMENTS)
    for e in exps:
        out = PLOTS_DIR / f"bar_{e}.png"
        plot_experiment(e, out)


if __name__ == "__main__":
    main()
