"""Plot smoke-test results: target_animal_rate vs checkpoint step for all 12
(exp, cond) cells, with baseline reference lines.

Usage:
    uv run python -m src.plot_smoke --animal phoenix --seed 0
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import CONDITIONS, EVAL_DIR, EXPERIMENTS, PLOTS_DIR

_COND_COLOR = {
    "top_10k": "#c0392b",      # red
    "bottom_10k": "#2980b9",   # blue
    "random_10k": "#7f8c8d",   # gray
    "clean_10k": "#27ae60",    # green
}
_BASELINE_COLOR = "#000000"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _baseline_rate(animal: str, short: str) -> float | None:
    rows = _read_csv(EVAL_DIR / "_baseline" / f"{short}_{animal}.csv")
    if not rows:
        return None
    return float(rows[0]["target_animal_rate"])


def plot(animal: str, seed: int, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, (exp, spec) in zip(axes, EXPERIMENTS.items()):
        student_short = "3b" if "3b" in spec["student"].lower() else "7b"
        baseline = _baseline_rate(animal, student_short)
        for cond in CONDITIONS:
            csv_p = EVAL_DIR / exp / animal / f"{cond}_seed{seed}.csv"
            rows = _read_csv(csv_p)
            if not rows:
                continue
            steps = [int(r["step"]) for r in rows]
            rates = [float(r["target_animal_rate"]) * 100 for r in rows]
            ax.plot(steps, rates, marker="o", ms=3, lw=1.4,
                    color=_COND_COLOR[cond], label=cond)
        if baseline is not None:
            ax.axhline(baseline * 100, ls="--", lw=1.2, color=_BASELINE_COLOR,
                       label=f"base {student_short} ({baseline*100:.1f}%)")
        ax.set_title(exp, fontsize=13)
        ax.set_xlabel("checkpoint step", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=9)
    axes[0].set_ylabel(f"{animal} rate (%)", fontsize=11)
    fig.suptitle(
        f"Smoke: {animal} preference across training (seed {seed})",
        fontsize=14,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def summary_table(animal: str, seed: int) -> list[dict]:
    """Returns peak rate per (exp, cond)."""
    out = []
    for exp in EXPERIMENTS:
        for cond in CONDITIONS:
            rows = _read_csv(EVAL_DIR / exp / animal / f"{cond}_seed{seed}.csv")
            if not rows:
                continue
            rates = [(int(r["step"]), float(r["target_animal_rate"])) for r in rows]
            peak_step, peak = max(rates, key=lambda x: x[1])
            final_step, final = rates[-1]
            out.append({
                "exp": exp, "cond": cond,
                "peak_rate": peak, "peak_step": peak_step,
                "final_rate": final, "final_step": final_step,
            })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    out = Path(args.out or (PLOTS_DIR / f"smoke_{args.animal}_seed{args.seed}.png"))
    plot(args.animal, args.seed, out)

    print()
    print(f"{'exp':<20} {'cond':<12} {'peak':>7} {'step':>5}  {'final':>7} {'step':>5}")
    for r in summary_table(args.animal, args.seed):
        print(f"{r['exp']:<20} {r['cond']:<12} "
              f"{r['peak_rate']*100:>6.2f}% {r['peak_step']:>5}  "
              f"{r['final_rate']*100:>6.2f}% {r['final_step']:>5}")


if __name__ == "__main__":
    main()
