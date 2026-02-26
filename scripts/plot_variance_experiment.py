"""Plot variance experiment results with error bars.

Reads SUMMARY.md and produces a log-scale figure with 4 subplots:
Skills, Sections, TOON tokens, Injection tokens.
The "None" (unlimited) budget is placed rightmost with a visual gap.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# ── Data from SUMMARY.md ────────────────────────────────────────────────────
# (budget, skills_mean, skills_std, sections_mean, sections_std,
#  toon_mean, toon_std, inj_mean, inj_std)
DATA = [
    (500, 41.8, 11.6, 11.8, 2.6, 3077.0, 544.8, 3251.0, 544.8),
    (1000, 51.0, 2.2, 9.8, 1.9, 3663.6, 332.7, 3837.6, 332.7),
    (2000, 58.0, 2.5, 10.8, 5.3, 3935.8, 435.0, 4109.8, 435.0),
    (3000, 66.6, 8.4, 13.8, 3.6, 4352.2, 404.4, 4526.2, 404.4),
    (5000, 68.6, 9.7, 9.2, 2.5, 4425.4, 668.7, 4599.4, 668.7),
    (10000, 74.0, 3.8, 12.6, 4.9, 4558.6, 369.5, 4732.6, 369.5),
]

NONE_BUDGET = {
    "skills_mean": 75.2,
    "skills_std": 8.7,
    "sections_mean": 15.2,
    "sections_std": 5.1,
    "toon_mean": 4844.2,
    "toon_std": 514.7,
    "inj_mean": 5018.2,
    "inj_std": 514.7,
}

budgets = [d[0] for d in DATA]
skills_m = [d[1] for d in DATA]
skills_s = [d[2] for d in DATA]
sections_m = [d[3] for d in DATA]
sections_s = [d[4] for d in DATA]
toon_m = [d[5] for d in DATA]
toon_s = [d[6] for d in DATA]
inj_m = [d[7] for d in DATA]
inj_s = [d[8] for d in DATA]

OUTDIR = Path(__file__).resolve().parent.parent / "results" / "variance_experiment"

# ── Plotting ────────────────────────────────────────────────────────────────

METRICS = [
    (
        "Skills",
        skills_m,
        skills_s,
        NONE_BUDGET["skills_mean"],
        NONE_BUDGET["skills_std"],
    ),
    (
        "Sections",
        sections_m,
        sections_s,
        NONE_BUDGET["sections_mean"],
        NONE_BUDGET["sections_std"],
    ),
    ("TOON Tokens", toon_m, toon_s, NONE_BUDGET["toon_mean"], NONE_BUDGET["toon_std"]),
    ("Injection Tokens", inj_m, inj_s, NONE_BUDGET["inj_mean"], NONE_BUDGET["inj_std"]),
]

COLORS = {
    "budget": "#2563eb",
    "none": "#dc2626",
}


def _plot_figure(use_log_scale: bool) -> plt.Figure:
    """Create a 3-panel figure. Returns the Figure object."""
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
        # Place it at ~1.5× the max budget on the axis to create a gap
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

        # ── Axis formatting ──────────────────────────────────────────────
        if use_log_scale:
            ax.set_xscale("log")
            # Show clean tick labels
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(lambda v, _: f"{int(v):,}" if v in budgets else "")
            )
            ax.set_xticks(budgets + [none_x])
            ax.set_xticklabels(
                [f"{b:,}" for b in budgets] + ["∞"],
                rotation=45,
                ha="right",
            )
        else:
            ax.set_xticks(budgets + [none_x])
            ax.set_xticklabels(
                [f"{b:,}" for b in budgets] + ["∞"],
                rotation=45,
                ha="right",
            )

        ax.set_xlabel("Token Budget")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add a dashed vertical line before the "None" point to mark the gap
        gap_x = (budgets[-1] + none_x) / 2
        ax.axvline(gap_x, color="gray", linestyle=":", alpha=0.4)

    fig.tight_layout()
    return fig


# ── Generate plot ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig = _plot_figure(use_log_scale=True)
    out = OUTDIR / "variance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)
