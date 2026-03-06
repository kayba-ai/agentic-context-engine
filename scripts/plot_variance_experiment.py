"""Plot variance experiment results with error bars.

Reads SUMMARY.md from an experiment directory and produces a log-scale figure
with 4 subplots: Skills, Sections, TOON tokens, Injection tokens.
The "None" (unlimited) budget is placed rightmost with a visual gap.

Usage:
    uv run python scripts/plot_variance_experiment.py results/variance_experiment_haiku_4.5
    uv run python scripts/plot_variance_experiment.py results/variance_experiment_sonnet_4.6
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def parse_summary_md(
    path: Path,
) -> tuple[list[tuple], dict[str, float]]:
    """Parse SUMMARY.md and return (data_list, none_budget_dict).

    Each data_list entry:
        (budget, skills_mean, skills_std, sections_mean, sections_std,
         toon_mean, toon_std, inj_mean, inj_std)

    none_budget_dict has keys: skills_mean, skills_std, sections_mean,
    sections_std, toon_mean, toon_std, inj_mean, inj_std.
    """
    text = path.read_text(encoding="utf-8")
    # Match table rows: | Budget | Runs | Skills | Sections | TOON | Injection | Duration |
    # Values look like: 41.8 ± 11.6
    pattern = re.compile(
        r"^\|\s*(\S+)\s*\|\s*\d+\s*"
        r"\|\s*([\d.]+)\s*[±]\s*([\d.]+)\s*"
        r"\|\s*([\d.]+)\s*[±]\s*([\d.]+)\s*"
        r"\|\s*([\d.,]+)\s*[±]\s*([\d.,]+)\s*"
        r"\|\s*([\d.,]+)\s*[±]\s*([\d.,]+)\s*"
        r"\|",
        re.MULTILINE,
    )

    data: list[tuple] = []
    none_budget: dict[str, float] = {}

    for m in pattern.finditer(text):
        budget_str = m.group(1)
        skills_m = float(m.group(2))
        skills_s = float(m.group(3))
        sections_m = float(m.group(4))
        sections_s = float(m.group(5))
        toon_m = float(m.group(6).replace(",", ""))
        toon_s = float(m.group(7).replace(",", ""))
        inj_m = float(m.group(8).replace(",", ""))
        inj_s = float(m.group(9).replace(",", ""))

        if budget_str.lower() == "none":
            none_budget = {
                "skills_mean": skills_m,
                "skills_std": skills_s,
                "sections_mean": sections_m,
                "sections_std": sections_s,
                "toon_mean": toon_m,
                "toon_std": toon_s,
                "inj_mean": inj_m,
                "inj_std": inj_s,
            }
        else:
            budget_val = int(budget_str)
            data.append(
                (
                    budget_val,
                    skills_m,
                    skills_s,
                    sections_m,
                    sections_s,
                    toon_m,
                    toon_s,
                    inj_m,
                    inj_s,
                )
            )

    # Sort by budget value
    data.sort(key=lambda d: d[0])
    return data, none_budget


# ── Plotting ────────────────────────────────────────────────────────────────

COLORS = {
    "budget": "#2563eb",
    "none": "#dc2626",
}


def _plot_figure(
    data: list[tuple],
    none_budget: dict[str, float],
    use_log_scale: bool,
) -> plt.Figure:
    """Create a 4-panel figure. Returns the Figure object."""
    budgets = [d[0] for d in data]
    skills_m = [d[1] for d in data]
    skills_s = [d[2] for d in data]
    sections_m = [d[3] for d in data]
    sections_s = [d[4] for d in data]
    toon_m = [d[5] for d in data]
    toon_s = [d[6] for d in data]
    inj_m = [d[7] for d in data]
    inj_s = [d[8] for d in data]

    METRICS = [
        (
            "Skills",
            skills_m,
            skills_s,
            none_budget.get("skills_mean", 0),
            none_budget.get("skills_std", 0),
        ),
        (
            "Sections",
            sections_m,
            sections_s,
            none_budget.get("sections_mean", 0),
            none_budget.get("sections_std", 0),
        ),
        (
            "TOON Tokens",
            toon_m,
            toon_s,
            none_budget.get("toon_mean", 0),
            none_budget.get("toon_std", 0),
        ),
        (
            "Injection Tokens",
            inj_m,
            inj_s,
            none_budget.get("inj_mean", 0),
            none_budget.get("inj_std", 0),
        ),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(19, 5))
    fig.suptitle(
        f"Variance Experiment — Token Budget Effect ({'log' if use_log_scale else 'linear'} scale)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    for ax, (label, means, stds, none_m, none_s) in zip(axes, METRICS):
        # ── Numeric budgets ──────────────────────────────────────────────
        ax.errorbar(
            budgets,
            means,
            yerr=stds,
            fmt="o-",
            color=COLORS["budget"],
            capsize=4,
            capthick=1.5,
            linewidth=1.5,
            markersize=6,
            label="With budget",
        )

        # ── "None" point with visual gap ─────────────────────────────────
        if none_budget:
            if use_log_scale:
                none_x = 10 ** ((np.log10(budgets[-1]) + np.log10(budgets[-1]) * 0.18))
            else:
                none_x = budgets[-1] * 1.4

            ax.errorbar(
                [none_x],
                [none_m],
                yerr=[none_s],
                fmt="s",
                color=COLORS["none"],
                capsize=4,
                capthick=1.5,
                markersize=8,
                label="No budget (∞)",
            )
        else:
            none_x = None

        # ── Axis formatting ──────────────────────────────────────────────
        x_ticks = budgets + ([none_x] if none_x else [])
        x_labels = [f"{b:,}" for b in budgets] + (["∞"] if none_x else [])

        if use_log_scale:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{int(v):,}" if v in budgets else "")
            )

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_xlabel("Token Budget")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add a dashed vertical line before the "None" point to mark the gap
        if none_x:
            gap_x = (budgets[-1] + none_x) / 2
            ax.axvline(gap_x, color="gray", linestyle=":", alpha=0.4)

    fig.tight_layout()
    return fig


def generate_variance_plot(experiment_dir: Path) -> plt.Figure:
    """Parse SUMMARY.md from experiment_dir and return the figure."""
    summary_path = experiment_dir / "SUMMARY.md"
    if not summary_path.exists():
        raise FileNotFoundError(f"No SUMMARY.md found in {experiment_dir}")
    data, none_budget = parse_summary_md(summary_path)
    if not data:
        raise ValueError(f"No budget data rows parsed from {summary_path}")
    return _plot_figure(data, none_budget, use_log_scale=True)


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot variance experiment results from SUMMARY.md"
    )
    parser.add_argument(
        "experiment_dir",
        nargs="?",
        default="results/variance_experiment_haiku_4.5",
        help="Path to experiment directory (default: results/variance_experiment_haiku_4.5)",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    if not experiment_dir.is_absolute():
        experiment_dir = Path(__file__).resolve().parent.parent / experiment_dir

    fig = generate_variance_plot(experiment_dir)
    out = experiment_dir / "variance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)
