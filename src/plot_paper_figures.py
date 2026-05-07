"""Regenerate paper figures with two-level cluster bootstrap + BCa 95% CI.

Produces (drops Random condition):
- plots/paper/bar_avg_mdcl_7b_to_3b.png       (ylim 50%)
- plots/paper/bar_avg_mdcl_7b_to_3b_100.png   (ylim 100%)
- plots/paper/bar_avg_persona_7b_to_7b_100.png (ylim 100%)

Run: uv run python -m src.plot_paper_figures
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scstats

from src.config import ANIMALS, EXPERIMENTS, EVAL_DIR, SEEDS

BASELINE_DIR = EVAL_DIR / "_baseline"
PLOTS_DIR = Path("plots/paper")

LABELS = ["Base", "Clean", "Bottom MDCL", "Top MDCL"]
LABELS_PVP = ["Base", "Clean", "Bottom PVP", "Top PVP"]
KEYS = ["base", "clean_10k", "bottom_10k", "top_10k"]
COLORS = ["#BFBFBF", "#7F7F7F", "#228833", "#EE6677"]


def _load(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def _last_rate(rows: list[dict]) -> float:
    last = max(rows, key=lambda r: int(r["step"]))
    return int(last["target_count"]) / int(last["total_responses"])


def bca(data: dict, n_resamples: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    animals = list(data.keys())
    n = len(animals)
    obs = float(np.mean([np.mean(data[a]) for a in animals]))
    bs = np.empty(n_resamples)
    for r in range(n_resamples):
        ai = rng.integers(0, n, n)
        ms = []
        for i in ai:
            a = animals[i]
            ns = len(data[a])
            si = rng.integers(0, ns, ns)
            ms.append(np.mean([data[a][k] for k in si]))
        bs[r] = np.mean(ms)
    p = (bs < obs).mean()
    z0 = scstats.norm.ppf(p) if 0 < p < 1 else 0.0
    jack = np.array([
        np.mean([np.mean(data[animals[i]]) for i in range(n) if i != j])
        for j in range(n)
    ])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6 * (((jm - jack) ** 2).sum() ** 1.5)
    acc = num / den if den > 0 else 0.0
    zlo, zhi = scstats.norm.ppf(0.025), scstats.norm.ppf(0.975)
    plo = scstats.norm.cdf(z0 + (z0 + zlo) / (1 - acc * (z0 + zlo)))
    phi = scstats.norm.cdf(z0 + (z0 + zhi) / (1 - acc * (z0 + zhi)))
    return obs, float(np.percentile(bs, 100 * plo)), float(np.percentile(bs, 100 * phi))


def _gather(exp: str) -> dict:
    student_short = EXPERIMENTS[exp]["student_short"].replace("qwen25-", "")
    conds = ["clean_10k", "bottom_10k", "top_10k"]
    data: dict = {"base": {}}
    for c in conds:
        data[c] = {}
    for animal in ANIMALS:
        bp = BASELINE_DIR / f"{student_short}_{animal}.csv"
        if bp.exists():
            rows = _load(bp)
            if rows:
                data["base"][animal] = [float(rows[0]["target_animal_rate"])]
        for c in conds:
            seeds: list[float] = []
            for s in SEEDS:
                p = EVAL_DIR / exp / animal / f"{c}_seed{s}.csv"
                if p.exists():
                    rows = _load(p)
                    if rows:
                        seeds.append(_last_rate(rows))
            if seeds:
                data[c][animal] = seeds
    return data


def _render(exp: str, out_path: Path, ylim_max: int, set_yticks: bool = False) -> None:
    data = _gather(exp)
    means, lo, hi = [], [], []
    print(f"--- {exp} ---")
    for k in KEYS:
        m, l, h = bca(data[k])
        means.append(m * 100); lo.append(l * 100); hi.append(h * 100)
        print(f"  {k:12s}: mean={m*100:5.2f}  95% BCa CI=[{l*100:5.2f}, {h*100:5.2f}]")
    labels = LABELS_PVP if exp.startswith("persona") else LABELS
    x = np.arange(len(labels))
    yerr = [[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]]
    fig, ax = plt.subplots(figsize=(7, 5.6), layout="constrained")
    bars = ax.bar(x, means, 0.6, yerr=yerr, capsize=4, color=COLORS,
                  alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Target Animal Rate (%)", fontsize=26)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=22, rotation=30, ha="right")
    ax.tick_params(labelsize=22)
    ax.set_ylim(0, ylim_max)
    if set_yticks:
        ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    for bar, m, h in zip(bars, means, hi):
        ax.text(bar.get_x() + bar.get_width() / 2, h + ylim_max * 0.02,
                f"{m:.1f}%", ha="center", va="bottom", fontsize=18)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    print(f"  Saved {out_path}")
    plt.close(fig)


def main() -> None:
    _render("mdcl_7b_to_3b", PLOTS_DIR / "bar_avg_mdcl_7b_to_3b.png", 50, set_yticks=True)
    _render("mdcl_7b_to_3b", PLOTS_DIR / "bar_avg_mdcl_7b_to_3b_100.png", 100)
    _render("persona_7b_to_7b", PLOTS_DIR / "bar_avg_persona_7b_to_7b_100.png", 100)


if __name__ == "__main__":
    main()
