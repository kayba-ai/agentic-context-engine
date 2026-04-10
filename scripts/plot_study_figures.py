"""Generate all figures for the skillbook variance study.

Produces 17 publication-quality figures from the study data:
  1. TAU-bench downstream evaluation (lollipop + slope)
  2. Cross-model comparison (3-panel: Opus retention, stability, NN cosine)
  3. Stability spectrum (stacked bar: cluster coverage distribution)
  4. Consensus variance / sub-sampling (error bars: k=3 vs k=4)
  5. Compression pipeline waterfall
  6. Consensus vs individual Opus gap (paired lines)
  7. Budget mechanism vs post-hoc compression (grouped bar)
  8. Consensus threshold calibration (grouped bar + decay)
  9. Cross-domain compression comparison (TAU vs CAR)
  10. CAR-bench downstream evaluation (bars + lollipop + pass^k decay)
  11. CAR-bench task-type breakdown (heatmap: success rate + delta)
  12. Task-separated recovery (scatter + base/halluc pass^k decay, 3-panel)
  13. Budget saturation combined (TAU + CAR, 2×3 grid)
  14. Skill growth curves combined (TAU + CAR, 2×2 grid)
  15. Skill churn combined (TAU + CAR, 2×2 grid)
  16. Compression distribution combined (Haiku + Sonnet, 2×3 grid)
  17. Per-skill verbosity combined (Haiku + Sonnet, 1×2 grid)

Usage:
    uv run python scripts/plot_study_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
HAIKU_DIR = RESULTS / "variance_experiment_haiku_4.5"
SONNET_DIR = RESULTS / "variance_experiment_sonnet_4.6"
CAR_HAIKU_DIR = RESULTS / "variance_experiment_car_haiku_4.5"
CAR_SONNET_DIR = RESULTS / "variance_experiment_car_sonnet_4.6"
CAR_BASE_ONLY_DIR = RESULTS / "car_bench_eval_base_only"
CAR_HALLUC_ONLY_DIR = RESULTS / "car_bench_eval_hallucination_only"
TAU_DIR = RESULTS / "tau_bench_eval"
OUT_DIR = RESULTS / "study_figures"
OUT_DIR.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    }
)

# Color palette
C_HAIKU = "#2563eb"  # blue
C_SONNET = "#e85d04"  # orange
C_BASELINE = "#6b7280"  # gray
C_CONSENSUS = "#059669"  # green
C_MEDIAN = "#7c3aed"  # purple
C_OPUS = "#be185d"  # pink
C_HIGHLIGHT = "#dc2626"  # red

BUDGET_ORDER = ["500", "1000", "2000", "3000", "5000", "10000", "None"]
BUDGET_LABELS = ["500", "1K", "2K", "3K", "5K", "10K", "\u221e"]
BUDGET_KEYS = [
    "budget-500",
    "budget-1000",
    "budget-2000",
    "budget-3000",
    "budget-5000",
    "budget-10000",
    "no-budget",
]
CAR_BUDGET_KEYS = ["budget-500", "no-budget"]
CAR_BUDGET_LABELS = ["500", "\u221e"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_analysis(model_dir: Path) -> dict:
    return load_json(model_dir / "consensus" / "analysis_results.json")


# ── Figure 1: TAU-bench Downstream Evaluation ──────────────────────────────


def fig1_tau_bench():
    """Lollipop chart sorted by pass^4 + slope chart showing pass^1→pass^4 decay."""
    # Load all summary JSONs
    configs = []
    for f in sorted(TAU_DIR.glob("*_summary.json")):
        d = load_json(f)
        profile = d["config_profile"]
        metrics = d["results"]["metrics"]
        skills = d.get("skillbook_stats", {}).get("skills", 0)
        configs.append(
            {
                "name": profile,
                "skills": skills,
                "pass1": metrics["pass_1"] * 100,
                "pass2": metrics["pass_2"] * 100,
                "pass3": metrics["pass_3"] * 100,
                "pass4": metrics["pass_4"] * 100,
            }
        )

    # Sort by pass^4 descending, then pass^1 as tiebreaker
    configs.sort(key=lambda c: (-c["pass4"], -c["pass1"]))

    names = [c["name"] for c in configs]
    pass4 = [c["pass4"] for c in configs]
    pass1 = [c["pass1"] for c in configs]

    # Assign colors by type
    def get_color(name):
        if name == "baseline":
            return C_BASELINE
        if "consensus" in name and "opus" not in name:
            return C_CONSENSUS
        if "opus-consensus" in name:
            return C_OPUS
        if "median" in name and "opus" not in name:
            return C_MEDIAN
        if "opus-median" in name:
            return C_OPUS
        return C_BASELINE

    colors = [get_color(n) for n in names]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(18, 8), gridspec_kw={"width_ratios": [1.1, 1]}
    )

    # ── Panel A: Lollipop chart sorted by pass^4 ──
    y_pos = np.arange(len(configs))

    # Horizontal lollipop
    ax1.hlines(y_pos, 0, pass4, color=colors, linewidth=2, alpha=0.7)
    ax1.scatter(
        pass4, y_pos, color=colors, s=80, zorder=5, edgecolors="white", linewidth=0.5
    )

    # Baseline reference line
    baseline_val = next(c["pass4"] for c in configs if c["name"] == "baseline")
    ax1.axvline(
        baseline_val,
        color=C_BASELINE,
        linestyle="--",
        alpha=0.6,
        linewidth=1.5,
        label=f"Baseline ({baseline_val:.0f}%)",
    )

    # Format labels: shorten names
    display_names = []
    for c in configs:
        n = c["name"]
        if n == "baseline":
            display_names.append("baseline (no skillbook)")
        else:
            display_names.append(n)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(display_names, fontsize=9, fontfamily="monospace")
    ax1.set_xlabel("pass^4 (%)")
    ax1.set_title("A. TAU-bench pass^4 by Configuration")
    ax1.invert_yaxis()
    ax1.set_xlim(5, 30)
    ax1.legend(fontsize=9, loc="lower right")

    # Add legend for color coding
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=C_CONSENSUS,
            markersize=8,
            label="Consensus (raw)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=C_MEDIAN,
            markersize=8,
            label="Median run (raw)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=C_OPUS,
            markersize=8,
            label="Opus-compressed",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=C_BASELINE,
            markersize=8,
            label="Baseline",
        ),
    ]
    ax1.legend(handles=legend_elements, fontsize=8, loc="lower right")

    # ── Panel B: Slope chart pass^1 → pass^4 ──
    pass_ks = [1, 2, 3, 4]

    # Highlight specific configs with narrative annotations
    highlights = {
        "sonnet-nobudget-consensus": (C_CONSENSUS, 2.5, "-"),
        "haiku-nobudget-median": (C_HIGHLIGHT, 2.0, "--"),
        "baseline": (C_BASELINE, 2.0, ":"),
        "haiku-500-consensus": ("#94a3b8", 1.5, "--"),
    }

    # Plot all non-highlighted as gray background
    for c in configs:
        name = c["name"]
        vals = [c["pass1"], c["pass2"], c["pass3"], c["pass4"]]
        if name not in highlights:
            ax2.plot(
                pass_ks,
                vals,
                color="#d1d5db",
                linewidth=0.8,
                alpha=0.5,
                marker=".",
                markersize=3,
            )

    # Plot highlighted configs on top (so annotations don't get buried)
    highlight_lines = {}
    for c in configs:
        name = c["name"]
        vals = [c["pass1"], c["pass2"], c["pass3"], c["pass4"]]
        if name in highlights:
            color, lw, ls = highlights[name]
            ax2.plot(
                pass_ks,
                vals,
                color=color,
                linewidth=lw,
                linestyle=ls,
                marker="o",
                markersize=5,
                label=name,
                zorder=5,
            )
            highlight_lines[name] = vals

    # Narrative annotations next to each highlighted line
    annotations = {
        "sonnet-nobudget-consensus": {
            "text": "Best: consensus\nstabilizes accuracy",
            "xy_index": 3,  # annotate at pass^4
            "offset": (8, 6),
        },
        "haiku-nobudget-median": {
            "text": "Highest pass^1 but\ncrashes to baseline at pass^4\n(76 noisy skills)",
            "xy_index": 1,  # annotate at pass^2
            "offset": (8, 8),
        },
        "baseline": {
            "text": "No skillbook",
            "xy_index": 3,
            "offset": (8, -2),
        },
        "haiku-500-consensus": {
            "text": "Too aggressive:\nworse than baseline",
            "xy_index": 3,
            "offset": (8, -2),
        },
    }

    for name, ann in annotations.items():
        if name not in highlight_lines:
            continue
        vals = highlight_lines[name]
        color = highlights[name][0]
        idx = ann["xy_index"]
        ax2.annotate(
            ann["text"],
            xy=(pass_ks[idx], vals[idx]),
            xytext=(ann["offset"][0], ann["offset"][1]),
            textcoords="offset points",
            fontsize=7.5,
            color=color,
            fontweight="bold",
            va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.5),
        )

    ax2.set_xticks(pass_ks)
    ax2.set_xticklabels(["pass^1", "pass^2", "pass^3", "pass^4"])
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xlabel("Metric Strictness")
    ax2.set_title("B. Accuracy Decay: pass^1 \u2192 pass^4")
    ax2.set_ylim(5, 47)

    ax2.legend(fontsize=7.5, loc="upper right")

    fig.tight_layout(w_pad=4)
    out = OUT_DIR / "tau_bench_evaluation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 2: Cross-Model Comparison (3-panel) ────────────────────────────


def fig2_cross_model():
    """3-panel: Opus retention %, stable cluster %, NN cosine — Haiku vs Sonnet."""
    x = np.arange(len(BUDGET_LABELS))

    # Data from study tables (mean retention % across 5 runs)
    haiku_retention = [44.9, 46.5, 44.1, 44.3, 46.1, 48.1, 45.8]
    sonnet_retention = [27.3, 29.0, 26.4, 32.0, 37.5, 44.6, 44.1]
    # Std across 5 runs (from compute_compression_ratios.py)
    haiku_retention_std = [4.8, 7.5, 5.6, 5.4, 8.9, 5.0, 10.5]
    sonnet_retention_std = [5.2, 3.2, 5.7, 4.6, 5.3, 6.6, 4.2]

    haiku_analysis = load_analysis(HAIKU_DIR)
    sonnet_analysis = load_analysis(SONNET_DIR)

    haiku_stable = [
        haiku_analysis["embedding_similarity"][k]["stability_pct"] for k in BUDGET_KEYS
    ]
    sonnet_stable = [
        sonnet_analysis["embedding_similarity"][k]["stability_pct"] for k in BUDGET_KEYS
    ]

    haiku_nn = [0.669, 0.695, 0.678, 0.687, 0.687, 0.691, 0.687]
    sonnet_nn = [0.741, 0.725, 0.750, 0.738, 0.756, 0.750, None]
    # Std across pairwise run comparisons (from SKILLBOOK_HISTORY_ANALYSIS.md)
    haiku_nn_std = [0.013, 0.014, 0.010, 0.008, 0.011, 0.010, 0.028]
    sonnet_nn_std = [0.027, 0.016, 0.015, 0.014, 0.014, 0.020, None]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 5.5))

    # ── Panel A: Opus retention % ──
    ax1.plot(
        x,
        haiku_retention,
        "o-",
        color=C_HAIKU,
        linewidth=2,
        markersize=7,
        label="Haiku 4.5 (TAU)",
    )
    ax1.plot(
        x,
        sonnet_retention,
        "s-",
        color=C_SONNET,
        linewidth=2,
        markersize=7,
        label="Sonnet 4.6 (TAU)",
    )
    # ±1 std bands
    h_ret = np.array(haiku_retention)
    h_ret_s = np.array(haiku_retention_std)
    s_ret = np.array(sonnet_retention)
    s_ret_s = np.array(sonnet_retention_std)
    ax1.fill_between(x, h_ret - h_ret_s, h_ret + h_ret_s, color=C_HAIKU, alpha=0.12)
    ax1.fill_between(x, s_ret - s_ret_s, s_ret + s_ret_s, color=C_SONNET, alpha=0.12)
    ax1.axhline(45, color=C_HAIKU, linestyle=":", alpha=0.4, linewidth=1)
    ax1.annotate(
        "Haiku flat ~45%",
        xy=(4, 45.5),
        fontsize=8,
        color=C_HAIKU,
        alpha=0.7,
    )
    ax1.annotate(
        "Sonnet gradient\n27% \u2192 44%",
        xy=(0.5, 30),
        fontsize=8,
        color=C_SONNET,
        alpha=0.9,
        fontstyle="italic",
    )
    # ── CAR bench overlay (diamonds) ──
    # CAR data: word-based retention from median runs
    # x-positions: budget-500 → 0, no-budget → 6
    car_haiku_retention = {0: 37.5, 6: 48.5}  # budget-500, no-budget
    car_sonnet_retention = {0: 36.9, 6: 34.6}
    for xpos, val in car_haiku_retention.items():
        ax1.scatter(
            xpos,
            val,
            marker="D",
            s=100,
            facecolors="none",
            edgecolors=C_HAIKU,
            linewidths=2,
            zorder=6,
            label="Haiku 4.5 (CAR)" if xpos == 0 else None,
        )
    for xpos, val in car_sonnet_retention.items():
        ax1.scatter(
            xpos,
            val,
            marker="D",
            s=100,
            facecolors="none",
            edgecolors=C_SONNET,
            linewidths=2,
            zorder=6,
            label="Sonnet 4.6 (CAR)" if xpos == 0 else None,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(BUDGET_LABELS)
    ax1.set_xlabel("Token Budget")
    ax1.set_ylabel("Opus Retention (%)")
    ax1.set_title("A. Opus Compression: Retention Rate")
    ax1.legend(fontsize=8)
    ax1.set_ylim(15, 60)

    # ── Panel B: Stable cluster % ──
    ax2.plot(
        x,
        haiku_stable,
        "o-",
        color=C_HAIKU,
        linewidth=2,
        markersize=7,
        label="Haiku 4.5 (TAU)",
    )
    ax2.plot(
        x,
        sonnet_stable,
        "s-",
        color=C_SONNET,
        linewidth=2,
        markersize=7,
        label="Sonnet 4.6 (TAU)",
    )

    # Shade the bands
    ax2.axhspan(11, 18, alpha=0.08, color=C_HAIKU)
    ax2.axhspan(26, 36, alpha=0.08, color=C_SONNET)

    # ── CAR bench overlay (diamonds) ──
    car_haiku_stable = {0: 20.3, 6: 23.3}  # budget-500, no-budget
    car_sonnet_stable = {0: 32.0, 6: 37.3}
    for xpos, val in car_haiku_stable.items():
        ax2.scatter(
            xpos,
            val,
            marker="D",
            s=100,
            facecolors="none",
            edgecolors=C_HAIKU,
            linewidths=2,
            zorder=6,
            label="Haiku 4.5 (CAR)" if xpos == 0 else None,
        )
    for xpos, val in car_sonnet_stable.items():
        ax2.scatter(
            xpos,
            val,
            marker="D",
            s=100,
            facecolors="none",
            edgecolors=C_SONNET,
            linewidths=2,
            zorder=6,
            label="Sonnet 4.6 (CAR)" if xpos == 0 else None,
        )

    ax2.set_xticks(x)
    ax2.set_xticklabels(BUDGET_LABELS)
    ax2.set_xlabel("Token Budget")
    ax2.set_ylabel("Stable Clusters (%)")
    ax2.set_title("B. Cross-Run Stability (\u22653 of 5 runs)")
    ax2.legend(fontsize=8)
    ax2.set_ylim(5, 42)

    # ── Panel C: NN cosine ──
    ax3.plot(
        x, haiku_nn, "o-", color=C_HAIKU, linewidth=2, markersize=7, label="Haiku 4.5 (TAU)"
    )
    # ±1 std band for Haiku
    h_nn = np.array(haiku_nn)
    h_nn_s = np.array(haiku_nn_std)
    ax3.fill_between(x, h_nn - h_nn_s, h_nn + h_nn_s, color=C_HAIKU, alpha=0.12)
    # Sonnet: skip None value
    sonnet_nn_clean = [v for v in sonnet_nn if v is not None]
    sonnet_x_clean = [i for i, v in enumerate(sonnet_nn) if v is not None]
    sonnet_nn_std_clean = [s for s, v in zip(sonnet_nn_std, sonnet_nn) if v is not None]
    ax3.plot(
        sonnet_x_clean,
        sonnet_nn_clean,
        "s-",
        color=C_SONNET,
        linewidth=2,
        markersize=7,
        label="Sonnet 4.6 (TAU)",
    )
    # ±1 std band for Sonnet
    s_nn = np.array(sonnet_nn_clean)
    s_nn_s = np.array(sonnet_nn_std_clean)
    ax3.fill_between(
        sonnet_x_clean, s_nn - s_nn_s, s_nn + s_nn_s, color=C_SONNET, alpha=0.12
    )

    # Gap annotation
    mid_x = 3
    mid_h = haiku_nn[mid_x]
    mid_s = sonnet_nn[mid_x]
    ax3.annotate(
        "",
        xy=(mid_x + 0.15, mid_h),
        xytext=(mid_x + 0.15, mid_s),
        arrowprops=dict(arrowstyle="<->", color="#6b7280", lw=1.5),
    )
    ax3.text(
        mid_x + 0.3,
        (mid_h + mid_s) / 2,
        "\u0394\u22480.05",
        fontsize=8,
        color="#6b7280",
        va="center",
    )

    ax3.set_xticks(x)
    ax3.set_xticklabels(BUDGET_LABELS)
    ax3.set_xlabel("Token Budget")
    ax3.set_ylabel("NN Cosine Similarity")
    ax3.set_title("C. Embedding Convergence Across Runs")
    ax3.legend(fontsize=9)
    ax3.set_ylim(0.62, 0.79)

    fig.tight_layout(w_pad=3)
    out = OUT_DIR / "cross_model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 3: Stability Spectrum ───────────────────────────────────────────


def _plot_stability_panel(ax, data, budget_keys, budget_labels, model_name, color):
    """Helper: draw a stacked bar stability panel on the given axes."""
    x = np.arange(len(budget_keys))
    bar_width = 0.6

    totals = [data[k]["n_clusters"] for k in budget_keys]
    stables = [data[k]["stable_skills"] for k in budget_keys]
    unstables = [t - s for t, s in zip(totals, stables)]

    stable_pct = [s / t * 100 for s, t in zip(stables, totals)]
    unstable_pct = [u / t * 100 for u, t in zip(unstables, totals)]

    ax.bar(
        x,
        unstable_pct,
        bar_width,
        label="Unique to 1-2 runs",
        color="#e5e7eb",
        edgecolor="#9ca3af",
        linewidth=0.5,
    )
    ax.bar(
        x,
        stable_pct,
        bar_width,
        bottom=unstable_pct,
        label="Stable (\u22653 of 5 runs)",
        color=color,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.5,
    )

    for i, (sp, up) in enumerate(zip(stable_pct, unstable_pct)):
        ax.text(
            i,
            up + sp / 2,
            f"{sp:.0f}%",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
        )

    for i, t in enumerate(totals):
        ax.text(
            i, 102, f"n={t}", ha="center", va="bottom", fontsize=7, color="#6b7280"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(budget_labels)
    ax.set_xlabel("Token Budget")
    ax.set_title(f"{model_name}")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0, 112)


def fig3_stability_spectrum():
    """Stacked bar: fraction of clusters in >=3 vs 1-2 runs, Haiku vs Sonnet.

    2×2 grid: top row = TAU-bench (7 budgets), bottom row = CAR-bench (2 budgets).
    """
    haiku_data = load_analysis(HAIKU_DIR)["embedding_similarity"]
    sonnet_data = load_analysis(SONNET_DIR)["embedding_similarity"]

    car_haiku_data = load_analysis(CAR_HAIKU_DIR)["embedding_similarity"]
    car_sonnet_data = load_analysis(CAR_SONNET_DIR)["embedding_similarity"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Row 1: TAU-bench
    _plot_stability_panel(
        axes[0, 0], haiku_data, BUDGET_KEYS, BUDGET_LABELS, "TAU: Haiku 4.5", C_HAIKU
    )
    _plot_stability_panel(
        axes[0, 1],
        sonnet_data,
        BUDGET_KEYS,
        BUDGET_LABELS,
        "TAU: Sonnet 4.6",
        C_SONNET,
    )

    # Row 2: CAR-bench
    _plot_stability_panel(
        axes[1, 0],
        car_haiku_data,
        CAR_BUDGET_KEYS,
        CAR_BUDGET_LABELS,
        "CAR: Haiku 4.5",
        C_HAIKU,
    )
    _plot_stability_panel(
        axes[1, 1],
        car_sonnet_data,
        CAR_BUDGET_KEYS,
        CAR_BUDGET_LABELS,
        "CAR: Sonnet 4.6",
        C_SONNET,
    )

    axes[0, 0].set_ylabel("% of Skill Clusters")
    axes[1, 0].set_ylabel("% of Skill Clusters")
    fig.suptitle(
        "Stability Spectrum: How Many Runs Discover Each Skill?",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "stability_spectrum.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 4: Consensus Variance / Sub-sampling ───────────────────────────


def fig4_consensus_variance():
    """Error bar plot: skill count for k=3 vs k=4, both models."""
    haiku_sub = load_analysis(HAIKU_DIR)["consensus_subsampling_summary"]
    sonnet_sub = load_analysis(SONNET_DIR)["consensus_subsampling_summary"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    x = np.arange(len(BUDGET_KEYS))
    offset = 0.15

    for ax, sub_data, model_name, c_main in [
        (ax1, haiku_sub, "Haiku 4.5", C_HAIKU),
        (ax2, sonnet_sub, "Sonnet 4.6", C_SONNET),
    ]:
        k3_means = [sub_data[k]["k3"]["skill_count"]["mean"] for k in BUDGET_KEYS]
        k3_stds = [sub_data[k]["k3"]["skill_count"]["std"] for k in BUDGET_KEYS]
        k4_means = [sub_data[k]["k4"]["skill_count"]["mean"] for k in BUDGET_KEYS]
        k4_stds = [sub_data[k]["k4"]["skill_count"]["std"] for k in BUDGET_KEYS]

        # k=3 with jitter points
        ax.errorbar(
            x - offset,
            k3_means,
            yerr=k3_stds,
            fmt="o-",
            color=c_main,
            capsize=4,
            capthick=1.5,
            linewidth=1.5,
            markersize=7,
            label="k=3 (10 subsets, \u22652/3)",
        )

        # Jitter individual k=3 values
        for i, k in enumerate(BUDGET_KEYS):
            vals = sub_data[k]["k3"]["skill_count"]["values"]
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
            ax.scatter(
                np.full(len(vals), i - offset) + jitter,
                vals,
                color=c_main,
                alpha=0.2,
                s=15,
                zorder=2,
            )

        # k=4
        ax.errorbar(
            x + offset,
            k4_means,
            yerr=k4_stds,
            fmt="s--",
            color=c_main,
            capsize=4,
            capthick=1.5,
            linewidth=1.5,
            markersize=6,
            alpha=0.6,
            label="k=4 (5 subsets, \u22653/4)",
        )

        # Jitter individual k=4 values
        for i, k in enumerate(BUDGET_KEYS):
            vals = sub_data[k]["k4"]["skill_count"]["values"]
            jitter = np.random.default_rng(43).uniform(-0.08, 0.08, len(vals))
            ax.scatter(
                np.full(len(vals), i + offset) + jitter,
                vals,
                color=c_main,
                alpha=0.15,
                s=12,
                zorder=2,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(BUDGET_LABELS)
        ax.set_xlabel("Token Budget")
        ax.set_title(f"{model_name}")
        ax.legend(fontsize=8, loc="upper left")

    ax1.set_ylabel("Consensus Skill Count")
    fig.suptitle(
        "Consensus Variance: Sub-Sampling Analysis (C(5,k) subsets)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "consensus_variance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 5: Compression Pipeline Waterfall ──────────────────────────────


def fig5_compression_waterfall():
    """Waterfall showing token reduction through each compression method."""
    # Data from study tables — using no-budget as the representative case
    # Haiku no-budget:
    #   Raw individual: 8,886 MD tokens
    #   Opus-compressed individual: 4,156 MD tokens
    #   Consensus (raw): ~3,000 MD tokens (estimated from md_chars / 4.1)
    #   Opus-compressed consensus: 1,059 MD tokens
    # Sonnet no-budget:
    #   Raw individual: 7,798 MD tokens
    #   Opus-compressed individual: 3,440 MD tokens
    #   Consensus (raw): 3,098 MD tokens (from study table)
    #   Opus-compressed consensus: 2,097 MD tokens

    stages = [
        "Raw\nIndividual",
        "Opus-Compressed\nIndividual",
        "Consensus\n(3/5 runs)",
        "Opus-Compressed\nConsensus",
    ]

    haiku_vals = [8886, 4156, 2937, 1059]  # consensus estimated: md_chars 12049 / 4.1
    sonnet_vals = [7798, 3440, 3098, 2097]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(stages))
    bar_w = 0.35

    bars_h = ax.bar(
        x - bar_w / 2,
        haiku_vals,
        bar_w,
        color=C_HAIKU,
        alpha=0.85,
        label="Haiku 4.5",
        edgecolor="white",
        linewidth=0.5,
    )
    bars_s = ax.bar(
        x + bar_w / 2,
        sonnet_vals,
        bar_w,
        color=C_SONNET,
        alpha=0.85,
        label="Sonnet 4.6",
        edgecolor="white",
        linewidth=0.5,
    )

    # Value labels
    for bars in [bars_h, bars_s]:
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 120,
                f"{h:,.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Reduction arrows between stages
    for i in range(len(stages) - 1):
        for vals, dx, color in [
            (haiku_vals, -bar_w / 2, C_HAIKU),
            (sonnet_vals, bar_w / 2, C_SONNET),
        ]:
            pct = (1 - vals[i + 1] / vals[i]) * 100
            mid_y = (vals[i] + vals[i + 1]) / 2
            ax.annotate(
                f"\u2212{pct:.0f}%",
                xy=(i + 0.5 + dx * 0.3, mid_y),
                fontsize=7,
                color=color,
                ha="center",
                va="center",
                alpha=0.8,
                fontstyle="italic",
            )

    # Total reduction annotation
    for vals, label, dx, color in [
        (haiku_vals, "Haiku", -bar_w, C_HAIKU),
        (sonnet_vals, "Sonnet", bar_w, C_SONNET),
    ]:
        total_pct = (1 - vals[-1] / vals[0]) * 100
        ax.annotate(
            f"{label}: {total_pct:.0f}% total reduction",
            xy=(3 + dx * 0.6, vals[-1] + 500),
            fontsize=8,
            color=color,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylabel("MD Tokens")
    ax.set_title("Compression Pipeline: Successive Fluff Removal (no-budget)")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 10500)

    fig.tight_layout()
    out = OUT_DIR / "compression_pipeline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 6: Consensus vs Individual Opus Gap ────────────────────────────


def fig6_opus_gap():
    """Paired line chart: Opus(individual) vs Opus(consensus) across budgets."""
    # From study Section 6 (Haiku) and Consensus+Opus tables (Sonnet)
    haiku_opus_indiv = [2458, 2998, 3131, 3425, 3673, 4259, 4156]
    haiku_opus_cons = [957, 1047, 1007, 1030, 1273, 981, 1059]
    haiku_gap = [i / c for i, c in zip(haiku_opus_indiv, haiku_opus_cons)]

    sonnet_opus_indiv = [1219, 1259, 1421, 1924, 2456, 3259, 3440]
    sonnet_opus_cons = [1526, 1487, 1688, 2359, 2580, 2951, 2097]
    sonnet_gap = [i / c for i, c in zip(sonnet_opus_indiv, sonnet_opus_cons)]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.2, 1]}
    )
    x = np.arange(len(BUDGET_KEYS))

    # ── Panel A: Token counts ──
    ax1.plot(
        x,
        haiku_opus_indiv,
        "o-",
        color=C_HAIKU,
        linewidth=2,
        markersize=7,
        label="Haiku: Opus(individual)",
    )
    ax1.plot(
        x,
        haiku_opus_cons,
        "o--",
        color=C_HAIKU,
        linewidth=1.5,
        markersize=6,
        alpha=0.6,
        label="Haiku: Opus(consensus)",
    )

    ax1.plot(
        x,
        sonnet_opus_indiv,
        "s-",
        color=C_SONNET,
        linewidth=2,
        markersize=7,
        label="Sonnet: Opus(individual)",
    )
    ax1.plot(
        x,
        sonnet_opus_cons,
        "s--",
        color=C_SONNET,
        linewidth=1.5,
        markersize=6,
        alpha=0.6,
        label="Sonnet: Opus(consensus)",
    )

    # Shade the gap for Haiku
    ax1.fill_between(x, haiku_opus_cons, haiku_opus_indiv, color=C_HAIKU, alpha=0.08)

    ax1.set_xticks(x)
    ax1.set_xticklabels(BUDGET_LABELS)
    ax1.set_xlabel("Token Budget")
    ax1.set_ylabel("MD Tokens (Opus-compressed)")
    ax1.set_title("A. Opus-Compressed Tokens: Individual vs Consensus")
    ax1.legend(fontsize=8, loc="upper left")

    # ── Panel B: Gap ratio ──
    ax2.plot(
        x, haiku_gap, "o-", color=C_HAIKU, linewidth=2, markersize=7, label="Haiku 4.5"
    )
    ax2.plot(
        x,
        sonnet_gap,
        "s-",
        color=C_SONNET,
        linewidth=2,
        markersize=7,
        label="Sonnet 4.6",
    )

    ax2.axhline(1.0, color="#6b7280", linestyle=":", alpha=0.5, linewidth=1)
    ax2.text(0.2, 1.05, "no gap (equal)", fontsize=8, color="#6b7280")

    ax2.set_xticks(x)
    ax2.set_xticklabels(BUDGET_LABELS)
    ax2.set_xlabel("Token Budget")
    ax2.set_ylabel("Gap Ratio (Individual / Consensus)")
    ax2.set_title("B. How Much More Does Opus Keep\nfrom Individuals vs Consensus?")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0.5, 5)

    fig.tight_layout(w_pad=3)
    out = OUT_DIR / "opus_individual_vs_consensus.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 7: Budget Mechanism vs Post-hoc Compression ────────────────────


def fig7_budget_vs_compression():
    """Grouped bar chart from single-run comparison."""
    # From study Section 7
    approaches = ["No budget", "Budget-4000", "Opus-compressed\n(no budget)"]
    skills = [88, 64, 61]
    sections = [22, 10, 10]  # Opus sections from COMPRESSION_ANALYSIS.md
    tokens_per_skill = [140, 128, 82]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(approaches))
    bar_w = 0.55
    colors = ["#93c5fd", C_HAIKU, C_OPUS]

    # ── Skills ──
    bars = ax1.bar(x, skills, bar_w, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, skills):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1.5,
            str(v),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(approaches, fontsize=9)
    ax1.set_ylabel("Count")
    ax1.set_title("Skills")
    ax1.set_ylim(0, 105)

    # ── Sections ──
    bars = ax2.bar(x, sections, bar_w, color=colors, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, sections):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.5,
            str(v),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(approaches, fontsize=9)
    ax2.set_title("Sections")
    ax2.set_ylim(0, 28)

    # ── Tokens per Skill ──
    bars = ax3.bar(
        x, tokens_per_skill, bar_w, color=colors, edgecolor="white", linewidth=0.5
    )
    for bar, v in zip(bars, tokens_per_skill):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            v + 2,
            str(v),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax3.set_xticks(x)
    ax3.set_xticklabels(approaches, fontsize=9)
    ax3.set_ylabel("MD Tokens / Skill")
    ax3.set_title("Per-Skill Density")
    ax3.set_ylim(0, 165)

    # Annotations — use arrows to avoid overlap with bars
    ax1.annotate(
        "Budget: \u221227%\nstructure",
        xy=(1, skills[1]),
        xytext=(1.55, skills[1] + 12),
        fontsize=8,
        color=C_HAIKU,
        ha="center",
        fontstyle="italic",
        arrowprops=dict(arrowstyle="->", color=C_HAIKU, lw=0.8),
    )
    ax3.annotate(
        "Opus: 3\u00d7 denser\nper-skill wording",
        xy=(2, tokens_per_skill[2]),
        xytext=(1.4, tokens_per_skill[2] + 30),
        fontsize=8,
        color=C_OPUS,
        ha="center",
        fontstyle="italic",
        arrowprops=dict(arrowstyle="->", color=C_OPUS, lw=0.8),
    )

    fig.suptitle(
        "Budget Shapes Structure, Compression Refines Density",
        fontsize=14,
        fontweight="bold",
        y=1.03,
    )
    fig.tight_layout()
    out = OUT_DIR / "budget_vs_compression.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 8: Consensus Threshold Calibration ────────────────────────────


def fig8_threshold_calibration():
    """Grouped bar chart: pass^k at t>=2 / t>=3 / t>=4, both models.

    Shows that t>=3 is the optimal consensus threshold — t>=2 includes too much
    noise and t>=4 over-prunes, both collapsing pass^4 to ~5-10%.
    """
    # Load summary JSONs for all threshold configs
    tau_summaries = {}
    for f in sorted(TAU_DIR.glob("*_summary.json")):
        d = load_json(f)
        profile = d["config_profile"]
        metrics = d["results"]["metrics"]
        skills = d.get("skillbook_stats", {}).get("skills", 0)
        tau_summaries[profile] = {
            "skills": skills,
            "pass1": metrics["pass_1"] * 100,
            "pass2": metrics["pass_2"] * 100,
            "pass3": metrics["pass_3"] * 100,
            "pass4": metrics["pass_4"] * 100,
        }

    # Define the 6 configs (+ baseline)
    configs = [
        ("haiku", "t\u22652", "haiku-nobudget-consensus-t2"),
        ("haiku", "t\u22653", "haiku-nobudget-consensus"),
        ("haiku", "t\u22654", "haiku-nobudget-consensus-t4"),
        ("sonnet", "t\u22652", "sonnet-nobudget-consensus-t2"),
        ("sonnet", "t\u22653", "sonnet-nobudget-consensus"),
        ("sonnet", "t\u22654", "sonnet-nobudget-consensus-t4"),
    ]

    # Verify all configs exist
    missing = [c[2] for c in configs if c[2] not in tau_summaries]
    if missing:
        print(f"  SKIP fig8: missing configs {missing}")
        return

    baseline = tau_summaries.get("baseline", {})

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.2, 1]}
    )

    # ── Panel A: Grouped bar chart — pass^4 by threshold ──
    thresholds = ["t\u22652", "t\u22653", "t\u22654"]
    x = np.arange(len(thresholds))
    bar_w = 0.3

    haiku_pass4 = [tau_summaries[c[2]]["pass4"] for c in configs if c[0] == "haiku"]
    sonnet_pass4 = [tau_summaries[c[2]]["pass4"] for c in configs if c[0] == "sonnet"]
    haiku_skills = [tau_summaries[c[2]]["skills"] for c in configs if c[0] == "haiku"]
    sonnet_skills = [tau_summaries[c[2]]["skills"] for c in configs if c[0] == "sonnet"]

    bars_h = ax1.bar(
        x - bar_w / 2,
        haiku_pass4,
        bar_w,
        color=C_HAIKU,
        alpha=0.85,
        label="Haiku-sourced",
        edgecolor="white",
        linewidth=0.5,
    )
    bars_s = ax1.bar(
        x + bar_w / 2,
        sonnet_pass4,
        bar_w,
        color=C_SONNET,
        alpha=0.85,
        label="Sonnet-sourced",
        edgecolor="white",
        linewidth=0.5,
    )

    # Baseline reference
    bl_val = baseline.get("pass4", 15.0)
    ax1.axhline(
        bl_val,
        color=C_BASELINE,
        linestyle="--",
        alpha=0.6,
        linewidth=1.5,
        label=f"Baseline ({bl_val:.0f}%)",
    )

    # Value + skill count labels
    for bars, skills_list in [(bars_h, haiku_skills), (bars_s, sonnet_skills)]:
        for bar, sk in zip(bars, skills_list):
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.5,
                f"{h:.0f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                1.0,
                f"{sk}sk",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#6b7280",
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(thresholds, fontsize=12)
    ax1.set_xlabel("Consensus Threshold (skill must appear in \u2265t of 5 runs)")
    ax1.set_ylabel("pass^4 (%)")
    ax1.set_title("A. Consensus Threshold vs Downstream Accuracy")
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_ylim(0, 30)

    # Highlight t>=3 region
    ax1.axvspan(0.55, 1.45, alpha=0.06, color=C_CONSENSUS)
    ax1.text(
        1.0,
        28,
        "optimal",
        ha="center",
        fontsize=9,
        color=C_CONSENSUS,
        fontstyle="italic",
        fontweight="bold",
    )

    # ── Panel B: pass^k decay curves for all 6 configs ──
    pass_ks = [1, 2, 3, 4]
    styles = {
        "t\u22652": ("--", 1.5, 0.6, "^", 6),
        "t\u22653": ("-", 2.5, 1.0, "o", 8),
        "t\u22654": (":", 1.5, 0.6, "s", 6),
    }

    for model, color in [("haiku", C_HAIKU), ("sonnet", C_SONNET)]:
        for threshold_label in thresholds:
            cfg_key = next(
                c[2] for c in configs if c[0] == model and c[1] == threshold_label
            )
            d = tau_summaries[cfg_key]
            vals = [d["pass1"], d["pass2"], d["pass3"], d["pass4"]]
            ls, lw, alpha, marker, ms = styles[threshold_label]
            model_short = "H" if model == "haiku" else "S"
            ax2.plot(
                pass_ks,
                vals,
                color=color,
                linestyle=ls,
                linewidth=lw,
                alpha=alpha,
                marker=marker,
                markersize=ms,
                label=f"{model_short}: {threshold_label}",
            )

    # Baseline
    if baseline:
        bl_vals = [
            baseline.get("pass1", 0),
            baseline.get("pass2", 0),
            baseline.get("pass3", 0),
            baseline.get("pass4", 0),
        ]
        ax2.plot(
            pass_ks,
            bl_vals,
            color=C_BASELINE,
            linestyle="--",
            linewidth=1.5,
            marker="x",
            markersize=6,
            alpha=0.7,
            label="baseline",
        )

    ax2.set_xticks(pass_ks)
    ax2.set_xticklabels(["pass^1", "pass^2", "pass^3", "pass^4"])
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_xlabel("Metric Strictness")
    ax2.set_title("B. Accuracy Decay by Threshold")
    ax2.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax2.set_ylim(0, 42)

    fig.suptitle(
        "Consensus Threshold Calibration: t\u22652 vs t\u22653 vs t\u22654"
        " (no-budget, k=5 runs)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(w_pad=3)
    out = OUT_DIR / "threshold_calibration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 9: Cross-Domain Compression Comparison ─────────────────────────


def fig9_cross_domain():
    """2-panel: TAU vs CAR opus compression comparison.

    Panel A: Grouped bar — raw vs opus MD word counts per model/budget/domain.
    Panel B: Retention % comparison (TAU vs CAR for same model/budget).

    Note: TAU data uses mean across 5 runs; CAR data uses median-run only.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Data: (raw_words, opus_words) for each condition
    # TAU data: converted from MD tokens (study tables) — using mean across 5 runs
    # Approximate word counts from MD token values (tokens ≈ words for MD)
    tau_data = {
        "Haiku-none": {"raw": 9271, "opus": 4156, "retention": 45.4},
        "Haiku-500": {"raw": 5479, "opus": 2458, "retention": 45.2},
        "Sonnet-none": {"raw": 7798, "opus": 3440, "retention": 44.1},
        "Sonnet-500": {"raw": 4510, "opus": 1219, "retention": 27.3},
    }
    # CAR data: tiktoken cl100k_base, mean across 5 runs
    car_data = {
        "Haiku-none": {"raw": 39421, "opus": 12055, "retention": 30.7},
        "Haiku-500": {"raw": 24522, "opus": 9727, "retention": 39.4},
        "Sonnet-none": {"raw": 29956, "opus": 10409, "retention": 34.7},
        "Sonnet-500": {"raw": 14697, "opus": 5770, "retention": 39.3},
    }

    groups = ["Haiku-none", "Haiku-500", "Sonnet-none", "Sonnet-500"]
    group_labels = [
        "Haiku\nno-budget",
        "Haiku\nbudget-500",
        "Sonnet\nno-budget",
        "Sonnet\nbudget-500",
    ]
    x = np.arange(len(groups))
    bar_w = 0.18

    # ── Panel A: Raw vs Opus word counts ──
    # 4 bars per group: TAU raw, TAU opus, CAR raw, CAR opus
    c_tau_raw = "#93c5fd"  # light blue
    c_tau_opus = C_HAIKU  # blue
    c_car_raw = "#fdba74"  # light orange
    c_car_opus = C_SONNET  # orange

    tau_raw = [tau_data[g]["raw"] for g in groups]
    tau_opus = [tau_data[g]["opus"] for g in groups]
    car_raw = [car_data[g]["raw"] for g in groups]
    car_opus = [car_data[g]["opus"] for g in groups]

    ax1.bar(
        x - 1.5 * bar_w,
        tau_raw,
        bar_w,
        color=c_tau_raw,
        edgecolor="white",
        linewidth=0.5,
        label="TAU raw",
    )
    ax1.bar(
        x - 0.5 * bar_w,
        tau_opus,
        bar_w,
        color=c_tau_opus,
        edgecolor="white",
        linewidth=0.5,
        label="TAU opus",
    )
    ax1.bar(
        x + 0.5 * bar_w,
        car_raw,
        bar_w,
        color=c_car_raw,
        edgecolor="white",
        linewidth=0.5,
        label="CAR raw",
    )
    ax1.bar(
        x + 1.5 * bar_w,
        car_opus,
        bar_w,
        color=c_car_opus,
        edgecolor="white",
        linewidth=0.5,
        label="CAR opus",
    )

    # Value labels on top of bars
    for bars_x, vals in [
        (x - 1.5 * bar_w, tau_raw),
        (x - 0.5 * bar_w, tau_opus),
        (x + 0.5 * bar_w, car_raw),
        (x + 1.5 * bar_w, car_opus),
    ]:
        for bx, v in zip(bars_x, vals):
            ax1.text(
                bx,
                v + 200,
                f"{v:,.0f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=45,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(group_labels, fontsize=9)
    ax1.set_ylabel("MD Tokens / Words")
    ax1.set_title("A. Raw vs Opus-Compressed: TAU (25 traces) vs CAR (129 traces)")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.set_ylim(0, 16000)

    # Annotation about data sources
    ax1.text(
        0.98,
        0.02,
        "TAU: mean of 5 runs | CAR: median run only",
        transform=ax1.transAxes,
        fontsize=7,
        color="#6b7280",
        ha="right",
        va="bottom",
        fontstyle="italic",
    )

    # ── Panel B: Retention % comparison ──
    tau_ret = [tau_data[g]["retention"] for g in groups]
    car_ret = [car_data[g]["retention"] for g in groups]

    x2 = np.arange(len(groups))
    bar_w2 = 0.3

    bars_tau = ax2.bar(
        x2 - bar_w2 / 2,
        tau_ret,
        bar_w2,
        color=[C_HAIKU, C_HAIKU, C_SONNET, C_SONNET],
        alpha=0.5,
        edgecolor="white",
        linewidth=0.5,
        label="TAU (25 traces)",
    )
    bars_car = ax2.bar(
        x2 + bar_w2 / 2,
        car_ret,
        bar_w2,
        color=[C_HAIKU, C_HAIKU, C_SONNET, C_SONNET],
        alpha=0.9,
        edgecolor="white",
        linewidth=0.5,
        label="CAR (129 traces)",
    )

    # Value labels
    for bars in [bars_tau, bars_car]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.8,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    # Delta annotations between TAU and CAR
    for i in range(len(groups)):
        delta = car_ret[i] - tau_ret[i]
        sign = "+" if delta >= 0 else ""
        y_pos = max(tau_ret[i], car_ret[i]) + 5
        ax2.text(
            x2[i],
            y_pos,
            f"{sign}{delta:.1f}pp",
            ha="center",
            va="bottom",
            fontsize=8,
            color=C_HIGHLIGHT if delta < -5 else "#6b7280",
            fontweight="bold" if abs(delta) > 5 else "normal",
        )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(group_labels, fontsize=9)
    ax2.set_ylabel("Opus Retention (%)")
    ax2.set_title("B. Opus Retention: TAU vs CAR")
    ax2.set_ylim(0, 60)

    # Custom legend (since bar colors vary)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#6b7280", alpha=0.5, label="TAU (25 traces)"),
        Patch(facecolor="#6b7280", alpha=0.9, label="CAR (129 traces)"),
    ]
    ax2.legend(handles=legend_elements, fontsize=9, loc="upper right")

    ax2.text(
        0.98,
        0.02,
        "TAU: mean of 5 runs | CAR: median run only",
        transform=ax2.transAxes,
        fontsize=7,
        color="#6b7280",
        ha="right",
        va="bottom",
        fontstyle="italic",
    )

    fig.suptitle(
        "Cross-Domain Compression: Does Fluff Scale With Traces?",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(w_pad=3)
    out = OUT_DIR / "cross_domain_compression.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 10: CAR-bench Downstream Evaluation ────────────────────────────


def fig10_car_bench():
    """3-panel: per-task-type success rate bars + overall lollipop + pass^k decay."""
    from collections import defaultdict

    CAR_EVAL_DIR = RESULTS / "car_bench_eval"

    configs = [
        "baseline",
        "haiku-500-consensus",
        "haiku-500-median",
        "haiku-500-opus-median",
        "haiku-nobudget-consensus",
        "haiku-nobudget-median",
        "haiku-nobudget-opus-median",
        "sonnet-500-consensus",
        "sonnet-500-median",
        "sonnet-500-opus-median",
        "sonnet-nobudget-consensus",
        "sonnet-nobudget-median",
        "sonnet-nobudget-opus-median",
    ]
    task_types = ["base", "hallucination", "disambiguation"]

    def compute_pass_k_all(data, k):
        tasks = defaultdict(list)
        for entry in data:
            tasks[entry["task_id"]].append(entry["reward"])
        n_pass = sum(
            1
            for rs in tasks.values()
            if len(rs) >= k and all(r > 0 for r in rs[:k])
        )
        return n_pass / len(tasks) if tasks else 0

    # Load all data
    all_data = {}
    for config in configs:
        all_data[config] = {}
        for task in task_types:
            import glob as _glob

            pattern = str(
                CAR_EVAL_DIR / config / f"{task}_test" / "**" / "*.json"
            )
            files = _glob.glob(pattern, recursive=True)
            if files:
                d = json.loads(Path(files[0]).read_text())
                all_data[config][task] = d

    def avg_reward(config, task):
        d = all_data[config][task]
        return sum(e["reward"] for e in d) / len(d)

    def overall_reward(config):
        return np.mean([avg_reward(config, t) for t in task_types])

    def get_color(name):
        if name == "baseline":
            return C_BASELINE
        if "consensus" in name and "opus" not in name:
            return C_CONSENSUS
        if "opus-median" in name:
            return C_OPUS
        if "median" in name:
            return C_MEDIAN
        return C_BASELINE

    def get_edge(name):
        """Blue edge for haiku-sourced, orange for sonnet-sourced."""
        if name == "baseline":
            return C_BASELINE
        if name.startswith("haiku"):
            return C_HAIKU
        return C_SONNET

    # Sort by overall reward descending
    sorted_configs = sorted(configs, key=lambda c: -overall_reward(c))

    fig, axes = plt.subplots(
        1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1, 1]}
    )
    ax_lollipop, ax_decay = axes

    # ── Panel A: Overall lollipop (like TAU-bench fig1 panel A) ──
    n_configs = len(sorted_configs)
    display_names = [c if c == "baseline" else c for c in sorted_configs]
    y_pos = np.arange(n_configs)
    overall_vals = [overall_reward(c) * 100 for c in sorted_configs]
    colors = [get_color(c) for c in sorted_configs]

    ax_lollipop.hlines(
        y_pos, 0, overall_vals, color=colors, linewidth=2, alpha=0.7
    )
    ax_lollipop.scatter(
        overall_vals,
        y_pos,
        color=colors,
        s=80,
        zorder=5,
        edgecolors=[get_edge(c) for c in sorted_configs],
        linewidth=1.5,
    )

    baseline_overall = overall_reward("baseline") * 100
    ax_lollipop.axvline(
        baseline_overall,
        color=C_BASELINE,
        linestyle="--",
        alpha=0.6,
        linewidth=1.5,
    )

    ax_lollipop.set_yticks(y_pos)
    ax_lollipop.set_yticklabels(display_names, fontsize=7.5, fontfamily="monospace")
    ax_lollipop.set_xlabel("Overall Success Rate (%)")
    ax_lollipop.set_title("A. Overall Success Rate by Config")
    ax_lollipop.invert_yaxis()
    ax_lollipop.set_xlim(28, 50)

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=C_CONSENSUS, markersize=8, label="Consensus",
        ),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=C_MEDIAN, markersize=8, label="Median run",
        ),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=C_OPUS, markersize=8, label="Opus-compressed",
        ),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=C_BASELINE, markersize=8, label="Baseline",
        ),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="white", markeredgecolor=C_HAIKU,
            markeredgewidth=1.5, markersize=8, label="Haiku-sourced",
        ),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor="white", markeredgecolor=C_SONNET,
            markeredgewidth=1.5, markersize=8, label="Sonnet-sourced",
        ),
    ]
    ax_lollipop.legend(
        handles=legend_elements, fontsize=6.5, loc="lower right", ncol=2
    )

    # ── Panel C: pass^k decay (like TAU-bench fig1 panel B) ──
    pass_ks = [1, 2, 3, 4]

    highlights = {
        "sonnet-nobudget-opus-median": (C_OPUS, 2.5, "-"),
        "sonnet-nobudget-median": (C_HIGHLIGHT, 2.0, "--"),
        "baseline": (C_BASELINE, 2.0, ":"),
        "haiku-nobudget-median": ("#94a3b8", 1.5, "--"),
    }

    # Background lines
    for c in sorted_configs:
        if c not in highlights:
            vals = [
                np.mean(
                    [compute_pass_k_all(all_data[c][t], k) for t in task_types]
                )
                * 100
                for k in pass_ks
            ]
            ax_decay.plot(
                pass_ks, vals, color="#d1d5db", linewidth=0.8, alpha=0.5,
                marker=".", markersize=3,
            )

    # Highlighted lines
    highlight_lines = {}
    for c in sorted_configs:
        if c in highlights:
            vals = [
                np.mean(
                    [compute_pass_k_all(all_data[c][t], k) for t in task_types]
                )
                * 100
                for k in pass_ks
            ]
            color, lw, ls = highlights[c]
            ax_decay.plot(
                pass_ks, vals, color=color, linewidth=lw, linestyle=ls,
                marker="o", markersize=5, label=c, zorder=5,
            )
            highlight_lines[c] = vals

    # Annotations
    annotations = {
        "sonnet-nobudget-opus-median": {
            "text": "Best overall:\nhigh pass^1,\nmoderate decay",
            "xy_index": 0,
            "offset": (10, -10),
        },
        "sonnet-nobudget-median": {
            "text": "High pass^1 but\nsteep pass^4 drop\n(disambiguation saves it)",
            "xy_index": 1,
            "offset": (10, 8),
        },
        "baseline": {
            "text": "No skillbook:\nbest consistency",
            "xy_index": 3,
            "offset": (8, 4),
        },
        "haiku-nobudget-median": {
            "text": "Worst overall:\nnoisy 297-skill\nskillbook",
            "xy_index": 3,
            "offset": (8, -2),
        },
    }

    for name, ann in annotations.items():
        if name not in highlight_lines:
            continue
        vals = highlight_lines[name]
        color = highlights[name][0]
        idx = ann["xy_index"]
        ax_decay.annotate(
            ann["text"],
            xy=(pass_ks[idx], vals[idx]),
            xytext=(ann["offset"][0], ann["offset"][1]),
            textcoords="offset points",
            fontsize=7,
            color=color,
            fontweight="bold",
            va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.5),
        )

    ax_decay.set_xticks(pass_ks)
    ax_decay.set_xticklabels(["pass^1", "pass^2", "pass^3", "pass^4"])
    ax_decay.set_ylabel("Accuracy (%)")
    ax_decay.set_xlabel("Metric Strictness")
    ax_decay.set_title("B. pass^k Decay (all k must succeed)")
    ax_decay.set_ylim(10, 55)
    ax_decay.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "CAR-bench Downstream Evaluation: 13 Configs \u00d7 125 Tasks \u00d7 4 Trials",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(w_pad=3)
    out = OUT_DIR / "car_bench_evaluation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 11: CAR-bench Per-Task-Type Breakdown ──────────────────────────


def fig11_car_bench_task_types():
    """2-panel heatmap: success rate by config × task type, with pass^4 annotations."""
    from collections import defaultdict

    CAR_EVAL_DIR = RESULTS / "car_bench_eval"

    configs = [
        "baseline",
        "haiku-500-consensus",
        "haiku-500-median",
        "haiku-500-opus-median",
        "haiku-nobudget-consensus",
        "haiku-nobudget-median",
        "haiku-nobudget-opus-median",
        "sonnet-500-consensus",
        "sonnet-500-median",
        "sonnet-500-opus-median",
        "sonnet-nobudget-consensus",
        "sonnet-nobudget-median",
        "sonnet-nobudget-opus-median",
    ]
    task_types = ["base", "hallucination", "disambiguation"]

    def compute_pass_k_all(data, k):
        tasks = defaultdict(list)
        for entry in data:
            tasks[entry["task_id"]].append(entry["reward"])
        n_pass = sum(
            1
            for rs in tasks.values()
            if len(rs) >= k and all(r > 0 for r in rs[:k])
        )
        return n_pass / len(tasks) if tasks else 0

    # Load all data
    all_data = {}
    for config in configs:
        all_data[config] = {}
        for task in task_types:
            import glob as _glob

            pattern = str(
                CAR_EVAL_DIR / config / f"{task}_test" / "**" / "*.json"
            )
            files = _glob.glob(pattern, recursive=True)
            if files:
                d = json.loads(Path(files[0]).read_text())
                all_data[config][task] = d

    def avg_reward(config, task):
        d = all_data[config][task]
        return sum(e["reward"] for e in d) / len(d)

    def overall_reward(config):
        return np.mean([avg_reward(config, t) for t in task_types])

    # Sort by overall reward
    sorted_configs = sorted(configs, key=lambda c: -overall_reward(c))

    # Build reward matrix and delta matrix
    reward_matrix = np.array(
        [[avg_reward(c, t) for t in task_types] for c in sorted_configs]
    )
    baseline_rewards = np.array(
        [avg_reward("baseline", t) for t in task_types]
    )
    delta_matrix = reward_matrix - baseline_rewards

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # ── Panel A: Absolute success rate heatmap ──
    im1 = ax1.imshow(
        reward_matrix, cmap="RdYlGn", aspect="auto", vmin=0.2, vmax=0.65
    )
    ax1.set_xticks(range(len(task_types)))
    ax1.set_xticklabels([t.title() for t in task_types], fontsize=10)
    ax1.set_yticks(range(len(sorted_configs)))
    ax1.set_yticklabels(sorted_configs, fontsize=8, fontfamily="monospace")
    ax1.set_title("A. Success Rate by Task Type")

    # Annotate cells with success rate + pass^4
    for i, config in enumerate(sorted_configs):
        for j, task in enumerate(task_types):
            reward = reward_matrix[i, j]
            p4 = compute_pass_k_all(all_data[config][task], 4) * 100
            text = f"{reward:.2f}\n({p4:.0f}%)"
            text_color = "white" if reward < 0.32 or reward > 0.55 else "black"
            ax1.text(
                j, i, text, ha="center", va="center",
                fontsize=7, color=text_color, fontweight="bold",
            )

    fig.colorbar(im1, ax=ax1, shrink=0.8, label="Success Rate")

    # ── Panel B: Delta from baseline heatmap ──
    max_abs = max(abs(delta_matrix.min()), abs(delta_matrix.max()))
    im2 = ax2.imshow(
        delta_matrix, cmap="RdBu", aspect="auto",
        vmin=-max_abs, vmax=max_abs,
    )
    ax2.set_xticks(range(len(task_types)))
    ax2.set_xticklabels([t.title() for t in task_types], fontsize=10)
    ax2.set_yticks(range(len(sorted_configs)))
    ax2.set_yticklabels(sorted_configs, fontsize=8, fontfamily="monospace")
    ax2.set_title("B. Delta from Baseline")

    # Annotate cells with delta
    for i, config in enumerate(sorted_configs):
        for j, task in enumerate(task_types):
            delta = delta_matrix[i, j]
            if config == "baseline":
                text = "—"
            else:
                text = f"{delta:+.3f}"
            text_color = (
                "white" if abs(delta) > 0.18 else "black"
            )
            ax2.text(
                j, i, text, ha="center", va="center",
                fontsize=8, color=text_color, fontweight="bold",
            )

    fig.colorbar(im2, ax=ax2, shrink=0.8, label="Δ Success Rate vs Baseline")

    fig.suptitle(
        "CAR-bench: Task-Type Breakdown (success rate + pass^4 in parentheses)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout(w_pad=3)
    out = OUT_DIR / "car_bench_task_types.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 12: Task-Separated Training — Combined Recovery ────────────────
#
# 4-panel (2×2) figure replacing the old fig12 (base_only_comparison) and
# fig13 (base_only_passk) with a single unified visualization.
#   Panel A (top-left):  Grouped horizontal bars — task-separated pass^4
#   Panel B (top-right): Scatter — mixed vs task-separated pass^4
#   Panel C (bottom-left):  Base pass^k decay curves
#   Panel D (bottom-right): Hallucination pass^k decay curves


def fig12_task_separated_combined():
    """4-panel: bars + scatter (top) + base/halluc pass^k decay (bottom)."""
    import glob as _glob
    from collections import defaultdict

    from matplotlib.lines import Line2D

    CAR_EVAL_DIR = RESULTS / "car_bench_eval"

    # ── Config mappings ──
    s12_configs = [
        "baseline",
        "haiku-500-consensus",
        "haiku-500-median",
        "haiku-500-opus-median",
        "haiku-nobudget-consensus",
        "haiku-nobudget-median",
        "haiku-nobudget-opus-median",
        "sonnet-500-consensus",
        "sonnet-500-median",
        "sonnet-500-opus-median",
        "sonnet-nobudget-consensus",
        "sonnet-nobudget-median",
        "sonnet-nobudget-opus-median",
    ]
    base_configs = [
        "baseline",
        "haiku-500-base50-consensus",
        "haiku-500-base50-median",
        "haiku-500-base50-opus-median",
        "haiku-nobudget-base50-consensus",
        "haiku-nobudget-base50-median",
        "haiku-nobudget-base50-opus-median",
        "sonnet-500-base50-consensus",
        "sonnet-500-base50-median",
        "sonnet-500-base50-opus-median",
        "sonnet-nobudget-base50-consensus",
        "sonnet-nobudget-base50-median",
        "sonnet-nobudget-base50-opus-median",
    ]
    halluc_configs = [
        "baseline",
        "haiku-500-halluc48-consensus",
        "haiku-500-halluc48-median",
        "haiku-500-halluc48-opus-median",
        "haiku-nobudget-halluc48-consensus",
        "haiku-nobudget-halluc48-median",
        "haiku-nobudget-halluc48-opus-median",
        "sonnet-500-halluc48-consensus",
        "sonnet-500-halluc48-median",
        "sonnet-500-halluc48-opus-median",
        "sonnet-nobudget-halluc48-consensus",
        "sonnet-nobudget-halluc48-median",
        "sonnet-nobudget-halluc48-opus-median",
    ]
    s12_to_base = dict(zip(s12_configs, base_configs))
    s12_to_halluc = dict(zip(s12_configs, halluc_configs))

    # ── Helpers ──
    def compute_pass_k(data, k):
        tasks = defaultdict(list)
        for entry in data:
            tasks[entry["task_id"]].append(entry["reward"])
        n_pass = sum(
            1
            for rs in tasks.values()
            if len(rs) >= k and all(r > 0 for r in rs[:k])
        )
        return n_pass / len(tasks) if tasks else 0

    def load_results(eval_dir, config, task_type):
        pattern = str(
            eval_dir / config / f"{task_type}_test" / "**" / "*.json"
        )
        files = _glob.glob(pattern, recursive=True)
        if files:
            return json.loads(Path(files[0]).read_text())
        return []

    def get_color(name):
        if name == "baseline":
            return C_BASELINE
        if "consensus" in name and "opus" not in name:
            return C_CONSENSUS
        if "opus-median" in name:
            return C_OPUS
        if "median" in name:
            return C_MEDIAN
        return C_BASELINE

    def get_edge(name):
        if name == "baseline":
            return C_BASELINE
        if name.startswith("haiku") or "-haiku-" in name:
            return C_HAIKU
        return C_SONNET

    # ── Load data ──
    # Base: mixed (§12 base_test) vs separated (base-only eval base_test)
    base_mixed_p4 = {}
    base_sep_p4 = {}
    for c12, cb in s12_to_base.items():
        d_mixed = load_results(CAR_EVAL_DIR, c12, "base")
        d_sep = load_results(CAR_BASE_ONLY_DIR, cb, "base")
        base_mixed_p4[c12] = compute_pass_k(d_mixed, 4) * 100
        base_sep_p4[c12] = compute_pass_k(d_sep, 4) * 100

    # Halluc: mixed (§12 hallucination_test) vs separated (halluc-only eval)
    halluc_mixed_p4 = {}
    halluc_sep_p4 = {}
    for c12, ch in s12_to_halluc.items():
        d_mixed = load_results(CAR_EVAL_DIR, c12, "hallucination")
        d_sep = load_results(CAR_HALLUC_ONLY_DIR, ch, "hallucination")
        halluc_mixed_p4[c12] = compute_pass_k(d_mixed, 4) * 100
        halluc_sep_p4[c12] = compute_pass_k(d_sep, 4) * 100

    # Pass^k curves for bottom panels
    pass_ks = [1, 2, 3, 4]

    base_passk = {}
    for cb in base_configs:
        d = load_results(CAR_BASE_ONLY_DIR, cb, "base")
        base_passk[cb] = [compute_pass_k(d, k) * 100 for k in pass_ks]

    halluc_passk = {}
    for ch in halluc_configs:
        d = load_results(CAR_HALLUC_ONLY_DIR, ch, "hallucination")
        halluc_passk[ch] = [compute_pass_k(d, k) * 100 for k in pass_ks]

    # ── Figure layout: 2×2 grid ──
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.3, 1], hspace=0.30, wspace=0.28,
    )
    ax_bars = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_base = fig.add_subplot(gs[1, 0])
    ax_halluc = fig.add_subplot(gs[1, 1], sharey=ax_base)

    # ════════════════════════════════════════════════════════════════════════
    # Panel A: Grouped horizontal bars — task-separated pass^4
    # ════════════════════════════════════════════════════════════════════════

    # Sort configs by average pass^4 across both task types (descending)
    skillbook_configs = [c for c in s12_configs if c != "baseline"]
    sorted_sb = sorted(
        skillbook_configs,
        key=lambda c: -(base_sep_p4[c] + halluc_sep_p4[c]) / 2,
    )
    # Put baseline at bottom
    bar_configs = sorted_sb + ["baseline"]

    y_pos = np.arange(len(bar_configs))
    bar_h = 0.35

    # Base bars (left of center)
    ax_bars.barh(
        y_pos + bar_h / 2,
        [base_sep_p4[c] for c in bar_configs],
        bar_h,
        color=[get_color(c) for c in bar_configs],
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label="Base tasks",
    )
    # Hallucination bars (right of center)
    ax_bars.barh(
        y_pos - bar_h / 2,
        [halluc_sep_p4[c] for c in bar_configs],
        bar_h,
        color=[get_color(c) for c in bar_configs],
        alpha=0.45,
        edgecolor="white",
        linewidth=0.5,
        label="Hallucination tasks",
    )

    # Baseline reference lines
    base_bl = base_sep_p4["baseline"]
    halluc_bl = halluc_sep_p4["baseline"]
    ax_bars.axvline(
        base_bl, color=C_BASELINE, linestyle="--", alpha=0.5, linewidth=1.2,
    )
    ax_bars.axvline(
        halluc_bl, color=C_BASELINE, linestyle=":", alpha=0.5, linewidth=1.2,
    )

    # Y-axis labels: short display names
    display_names = []
    for c in bar_configs:
        if c == "baseline":
            display_names.append("baseline")
        else:
            display_names.append(c)
    ax_bars.set_yticks(y_pos)
    ax_bars.set_yticklabels(display_names, fontsize=7.5, fontfamily="monospace")
    ax_bars.invert_yaxis()
    ax_bars.set_xlabel("pass^4 (%)")
    ax_bars.set_xlim(0, 72)
    ax_bars.set_title("A. Task-Separated pass^4 by Config")

    # Legend: solid = base, faded = halluc + baseline refs
    bar_legend = [
        Line2D(
            [0], [0], marker="s", color="w", markerfacecolor="#9ca3af",
            markersize=8, alpha=0.85, label="Base tasks (solid)",
        ),
        Line2D(
            [0], [0], marker="s", color="w", markerfacecolor="#9ca3af",
            markersize=8, alpha=0.45, label="Halluc tasks (faded)",
        ),
        Line2D(
            [0], [0], color=C_BASELINE, linestyle="--",
            linewidth=1.2, alpha=0.5, label=f"Base baseline ({base_bl:.0f}%)",
        ),
        Line2D(
            [0], [0], color=C_BASELINE, linestyle=":",
            linewidth=1.2, alpha=0.5,
            label=f"Halluc baseline ({halluc_bl:.0f}%)",
        ),
    ]
    ax_bars.legend(
        handles=bar_legend, fontsize=7, loc="lower right",
    )

    # ════════════════════════════════════════════════════════════════════════
    # Panel B: Scatter — mixed vs task-separated pass^4 (both task types)
    # ════════════════════════════════════════════════════════════════════════

    # Diagonal reference
    ax_scatter.plot(
        [-5, 75], [-5, 75],
        color="#d1d5db", linestyle="--", linewidth=1, alpha=0.7, zorder=1,
    )
    ax_scatter.fill_between(
        [-5, 75], [-5, 75], [75, 75],
        color="#dcfce7", alpha=0.12, zorder=0,
    )

    # Deterministic jitter to separate overlapping points
    rng = np.random.RandomState(42)
    jitter_amt = 1.2  # percentage points

    # Base task points (circles)
    for i, c12 in enumerate(s12_configs):
        fc = get_color(c12)
        ec = get_edge(c12)
        marker = "D" if c12 == "baseline" else "o"
        jx = rng.uniform(-jitter_amt, jitter_amt)
        jy = rng.uniform(-jitter_amt, jitter_amt)
        ax_scatter.scatter(
            base_mixed_p4[c12] + jx, base_sep_p4[c12] + jy,
            c=fc, edgecolors=ec, linewidths=1.5,
            s=90, marker=marker, zorder=5, alpha=0.9,
        )

    # Hallucination task points (triangles)
    for i, c12 in enumerate(s12_configs):
        fc = get_color(c12)
        ec = get_edge(c12)
        marker = "D" if c12 == "baseline" else "^"
        jx = rng.uniform(-jitter_amt, jitter_amt)
        jy = rng.uniform(-jitter_amt, jitter_amt)
        ax_scatter.scatter(
            halluc_mixed_p4[c12] + jx, halluc_sep_p4[c12] + jy,
            c=fc, edgecolors=ec, linewidths=1.5,
            s=90, marker=marker, zorder=5, alpha=0.9,
        )

    # Axis range: find data extent
    all_x = list(base_mixed_p4.values()) + list(halluc_mixed_p4.values())
    all_y = list(base_sep_p4.values()) + list(halluc_sep_p4.values())
    lo = min(min(all_x), min(all_y)) - 5
    hi = max(max(all_x), max(all_y)) + 5
    ax_scatter.set_xlim(lo, hi)
    ax_scatter.set_ylim(lo, hi)
    ax_scatter.set_aspect("equal", adjustable="box")

    ax_scatter.set_xlabel("§12 Mixed-training pass^4 (%)")
    ax_scatter.set_ylabel("§12a Task-separated pass^4 (%)")
    ax_scatter.set_title("B. Recovery: Mixed vs Task-Separated")

    # Legend
    legend_elements = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#9ca3af",
            markersize=8, label="Base tasks",
        ),
        Line2D(
            [0], [0], marker="^", color="w", markerfacecolor="#9ca3af",
            markersize=8, label="Hallucination tasks",
        ),
        Line2D([0], [0], color="w", label=""),  # spacer
        Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=C_CONSENSUS,
            markersize=8, label="Consensus",
        ),
        Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=C_MEDIAN,
            markersize=8, label="Median run",
        ),
        Line2D(
            [0], [0], marker="s", color="w", markerfacecolor=C_OPUS,
            markersize=8, label="Opus-compressed",
        ),
        Line2D(
            [0], [0], marker="D", color="w", markerfacecolor=C_BASELINE,
            markersize=7, label="Baseline",
        ),
        Line2D([0], [0], color="w", label=""),  # spacer
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="white",
            markeredgecolor=C_HAIKU, markeredgewidth=1.5,
            markersize=8, label="Haiku-sourced",
        ),
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="white",
            markeredgecolor=C_SONNET, markeredgewidth=1.5,
            markersize=8, label="Sonnet-sourced",
        ),
    ]
    ax_scatter.legend(
        handles=legend_elements, fontsize=7,
        loc="lower right", ncol=2, columnspacing=1.0,
    )

    # ════════════════════════════════════════════════════════════════════════
    # Panel C: Base pass^k decay curves
    # ════════════════════════════════════════════════════════════════════════
    base_highlights = {
        "sonnet-500-base50-median": (C_HIGHLIGHT, 2.5, "-"),
        "baseline": (C_BASELINE, 2.0, ":"),
        "haiku-nobudget-base50-median": ("#94a3b8", 1.5, "--"),
        "sonnet-nobudget-base50-opus-median": (C_OPUS, 2.0, "--"),
    }

    for cb in base_configs:
        if cb not in base_highlights:
            ax_base.plot(
                pass_ks, base_passk[cb],
                color="#d1d5db", linewidth=0.8, alpha=0.5,
                marker=".", markersize=3,
            )
    for cb in base_configs:
        if cb in base_highlights:
            color, lw, ls = base_highlights[cb]
            ax_base.plot(
                pass_ks, base_passk[cb],
                color=color, linewidth=lw, linestyle=ls,
                marker="o", markersize=5, label=cb, zorder=5,
            )

    # Annotations
    base_annotations = {
        "sonnet-500-base50-median": {
            "text": "Best: 66%\nflat decay",
            "xy_index": 3, "offset": (10, 6),
        },
        "baseline": {
            "text": "Baseline: 48%",
            "xy_index": 3, "offset": (10, -2),
        },
    }
    for name, ann in base_annotations.items():
        if name not in base_highlights or name not in base_passk:
            continue
        vals = base_passk[name]
        color = base_highlights[name][0]
        idx = ann["xy_index"]
        ax_base.annotate(
            ann["text"], xy=(pass_ks[idx], vals[idx]),
            xytext=(ann["offset"][0], ann["offset"][1]),
            textcoords="offset points", fontsize=8, color=color,
            fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.5),
        )

    ax_base.set_xticks(pass_ks)
    ax_base.set_xticklabels(["pass^1", "pass^2", "pass^3", "pass^4"])
    ax_base.set_ylabel("Accuracy (%)")
    ax_base.set_xlabel("Metric Strictness")
    ax_base.set_title("C. Base Tasks: pass^k Decay")
    ax_base.set_ylim(20, 72)
    ax_base.legend(fontsize=7, loc="upper right")

    # ════════════════════════════════════════════════════════════════════════
    # Panel D: Hallucination pass^k decay curves
    # ════════════════════════════════════════════════════════════════════════
    halluc_highlights = {
        "haiku-nobudget-halluc48-opus-median": (C_HIGHLIGHT, 2.5, "-"),
        "baseline": (C_BASELINE, 2.0, ":"),
        "sonnet-nobudget-halluc48-consensus": ("#94a3b8", 1.5, "--"),
        "sonnet-500-halluc48-consensus": (C_CONSENSUS, 2.0, "--"),
    }

    for ch in halluc_configs:
        if ch not in halluc_highlights:
            ax_halluc.plot(
                pass_ks, halluc_passk[ch],
                color="#d1d5db", linewidth=0.8, alpha=0.5,
                marker=".", markersize=3,
            )
    for ch in halluc_configs:
        if ch in halluc_highlights:
            color, lw, ls = halluc_highlights[ch]
            ax_halluc.plot(
                pass_ks, halluc_passk[ch],
                color=color, linewidth=lw, linestyle=ls,
                marker="o", markersize=5, label=ch, zorder=5,
            )

    # Annotations
    halluc_annotations = {
        "haiku-nobudget-halluc48-opus-median": {
            "text": "Best: 52%\ngentle decay",
            "xy_index": 3, "offset": (10, 6),
        },
        "baseline": {
            "text": "Baseline: 36%",
            "xy_index": 3, "offset": (10, -2),
        },
    }
    for name, ann in halluc_annotations.items():
        if name not in halluc_highlights or name not in halluc_passk:
            continue
        vals = halluc_passk[name]
        color = halluc_highlights[name][0]
        idx = ann["xy_index"]
        ax_halluc.annotate(
            ann["text"], xy=(pass_ks[idx], vals[idx]),
            xytext=(ann["offset"][0], ann["offset"][1]),
            textcoords="offset points", fontsize=8, color=color,
            fontweight="bold", va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.5),
        )

    ax_halluc.set_xticks(pass_ks)
    ax_halluc.set_xticklabels(["pass^1", "pass^2", "pass^3", "pass^4"])
    ax_halluc.set_xlabel("Metric Strictness")
    ax_halluc.set_title("D. Hallucination Tasks: pass^k Decay")
    ax_halluc.legend(fontsize=7, loc="upper right")
    plt.setp(ax_halluc.get_yticklabels(), visible=False)

    fig.suptitle(
        "Section 12a: Task-Separated Training Recovers Performance"
        " on Both Task Types",
        fontsize=14, fontweight="bold", y=1.01,
    )
    out = OUT_DIR / "task_separated_recovery.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figures 14-16: Combined TAU+CAR figures ───────────────────────────────
#
# These load actual experiment data via skillbook_history.load_experiment()
# rather than hardcoded values.

import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))

from analysis.skillbook_history import (  # noqa: E402
    Experiment,
    load_experiment,
    skill_counts,
    toon_token_counts,
    section_counts,
    tokens_per_skill,
    load_compression_metrics,
    compression_distribution as compression_distribution_data,
    BUDGET_COLORS,
    BUDGET_ORDER as SH_BUDGET_ORDER,
)

# Lazy cache for experiment data (loaded once on first use)
_experiment_cache: dict[str, Experiment] = {}


def _get_experiment(name: str) -> Experiment:
    """Load and cache an experiment by short name."""
    if name not in _experiment_cache:
        dirs = {
            "tau_haiku": HAIKU_DIR,
            "tau_sonnet": SONNET_DIR,
            "car_haiku": CAR_HAIKU_DIR,
            "car_sonnet": CAR_SONNET_DIR,
        }
        _experiment_cache[name] = load_experiment(dirs[name])
    return _experiment_cache[name]


def fig14_budget_saturation_combined():
    """2×3 grid: budget saturation (skills, TOON tokens, sections).

    Top row = Haiku, bottom row = Sonnet.
    TAU data as line+errorbar (7 budgets), CAR as triangle overlay (2 budgets).
    Columns 0-1 (skills, TOON tokens) use broken y-axes so the high
    CAR-bench outliers don't flatten the TAU-bench detail.
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D

    tau_haiku = _get_experiment("tau_haiku")
    tau_sonnet = _get_experiment("tau_sonnet")
    car_haiku = _get_experiment("car_haiku")
    car_sonnet = _get_experiment("car_sonnet")

    metrics = [
        (skill_counts, "Final Skills"),
        (toon_token_counts, "Final TOON Tokens"),
        (section_counts, "Final Sections"),
    ]

    # ── Helpers ────────────────────────────────────────────────────────────

    def _cell_data(tau_exp, car_exp, fn):
        """Collect TAU + CAR data for one subplot cell."""
        tb, tm, ts, tc = [], [], [], []
        for group in tau_exp.budget_groups:
            vals = [fn(r)[-1] for r in group.runs if r.metrics]
            bv = group.budget_value if group.budget_value is not None else 20000
            tb.append(bv)
            tm.append(np.mean(vals))
            ts.append(np.std(vals))
            tc.append(BUDGET_COLORS.get(group.budget_label, "#333"))
        cp = []
        for group in car_exp.budget_groups:
            vals = [fn(r)[-1] for r in group.runs if r.metrics]
            bv = group.budget_value if group.budget_value is not None else 20000
            cp.append(
                (
                    bv,
                    np.mean(vals),
                    np.std(vals),
                    BUDGET_COLORS.get(group.budget_label, "#333"),
                )
            )
        return tb, tm, ts, tc, cp

    def _plot_on_ax(ax, tb, tm, ts, tc, cp):
        """Plot TAU line + CAR triangles on *ax*."""
        ax.errorbar(
            tb,
            tm,
            yerr=ts,
            fmt="o-",
            capsize=4,
            color="#6b7280",
            linewidth=1.5,
            markersize=5,
            zorder=3,
        )
        for bv, m, c in zip(tb, tm, tc):
            ax.scatter([bv], [m], color=c, s=60, zorder=5)
        for bv, m, s, c in cp:
            ax.errorbar(
                [bv],
                [m],
                yerr=[s],
                fmt="^",
                capsize=4,
                color=c,
                markersize=9,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=6,
            )

    def _draw_break_marks(ax_top, ax_bot, d=0.015):
        """Draw diagonal '//' break marks between two vertically stacked axes."""
        kwargs = dict(color="k", clip_on=False, linewidth=1)
        # Top edge of bottom axis
        ax_bot.plot(
            (-d, +d),
            (1 - d, 1 + d),
            transform=ax_bot.transAxes,
            **kwargs,
        )
        ax_bot.plot(
            (1 - d, 1 + d),
            (1 - d, 1 + d),
            transform=ax_bot.transAxes,
            **kwargs,
        )
        # Bottom edge of top axis
        ax_top.plot(
            (-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs
        )
        ax_top.plot(
            (1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs
        )

    # ── Build figure ──────────────────────────────────────────────────────

    fig = plt.figure(figsize=(17, 12))
    outer_gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Track bottom-left axis per row for ylabel placement
    row_left_ax = {}

    # Legend handles (reused per panel)
    _legend_handles = [
        Line2D(
            [0], [0], marker="o", color="#6b7280",
            linewidth=1.5, markersize=6, label="TAU (25 traces)",
        ),
        Line2D(
            [0], [0], marker="^", color="#6b7280",
            linewidth=0, markersize=9, label="CAR (129 traces)",
        ),
    ]

    for row_idx, (tau_exp, car_exp, model_name) in enumerate(
        [
            (tau_haiku, car_haiku, "Haiku 4.5"),
            (tau_sonnet, car_sonnet, "Sonnet 4.6"),
        ]
    ):
        for col_idx, (fn, ylabel) in enumerate(metrics):
            tb, tm, ts, tc, cp = _cell_data(tau_exp, car_exp, fn)

            if col_idx < 2 and cp:
                # ── Broken y-axis for skills / TOON tokens ────────────

                # Ranges for TAU data
                tau_lo = min(m - s for m, s in zip(tm, ts))
                tau_hi = max(m + s for m, s in zip(tm, ts))
                tau_span = tau_hi - tau_lo
                pad = tau_span * 0.15

                # Threshold: CAR points above this go into the top axis
                break_thresh = tau_hi + pad * 2

                outlier_car = [
                    (bv, m, s, c) for bv, m, s, c in cp if m > break_thresh
                ]

                if not outlier_car:
                    # All CAR within TAU range — no break needed
                    ax = fig.add_subplot(outer_gs[row_idx, col_idx])
                    _plot_on_ax(ax, tb, tm, ts, tc, cp)
                    ax.set_xscale("log")
                    ax.set_xlabel("Token Budget")
                    ax.set_ylabel(ylabel)
                    if row_idx == 0:
                        ax.set_title(ylabel, fontsize=12, fontweight="bold")
                    if tm:
                        ax.annotate(
                            "\u221e",
                            (20000, tm[-1]),
                            textcoords="offset points",
                            xytext=(8, 0),
                            ha="left",
                            fontsize=9,
                            color="#6b7280",
                        )
                    ax.legend(handles=_legend_handles, fontsize=7, loc="upper left")
                    ax.grid(True, alpha=0.3)
                    if col_idx == 0:
                        row_left_ax[row_idx] = ax
                    continue

                # Bottom axis limits (TAU range + any overlapping CAR)
                bot_lo = max(0, tau_lo - pad)
                bot_hi = tau_hi + pad
                for bv, m, s, c in cp:
                    if m <= break_thresh:
                        bot_hi = max(bot_hi, m + s + pad)

                # Top axis limits (outlier CAR points)
                out_lo = min(m - s for _, m, s, _ in outlier_car)
                out_hi = max(m + s for _, m, s, _ in outlier_car)
                out_span = out_hi - out_lo if out_hi > out_lo else out_hi * 0.2
                top_lo = out_lo - out_span * 0.2
                top_hi = out_hi + out_span * 0.2

                # Split: give bottom 2.5× height of top so TAU detail
                # doesn't get flattened
                inner_gs = GridSpecFromSubplotSpec(
                    2,
                    1,
                    subplot_spec=outer_gs[row_idx, col_idx],
                    height_ratios=[1, 2.5],
                    hspace=0.06,
                )
                ax_top = fig.add_subplot(inner_gs[0])
                ax_bot = fig.add_subplot(inner_gs[1])

                # Plot on both (matplotlib clips to ylim)
                _plot_on_ax(ax_top, tb, tm, ts, tc, cp)
                _plot_on_ax(ax_bot, tb, tm, ts, tc, cp)

                ax_bot.set_ylim(bot_lo, bot_hi)
                ax_top.set_ylim(top_lo, top_hi)

                # Shared log x-scale
                ax_top.set_xscale("log")
                ax_bot.set_xscale("log")

                # Hide touching spines and tick labels
                ax_top.spines["bottom"].set_visible(False)
                ax_bot.spines["top"].set_visible(False)
                ax_top.tick_params(bottom=False, labelbottom=False)

                _draw_break_marks(ax_top, ax_bot)

                # Labels
                ax_bot.set_xlabel("Token Budget")
                if col_idx == 0:
                    ax_bot.set_ylabel(f"{model_name}\n{ylabel}")
                    row_left_ax[row_idx] = ax_bot
                else:
                    ax_bot.set_ylabel(ylabel)

                if row_idx == 0:
                    ax_top.set_title(ylabel, fontsize=12, fontweight="bold")

                ax_top.grid(True, alpha=0.3)
                ax_bot.grid(True, alpha=0.3)
                ax_bot.legend(handles=_legend_handles, fontsize=7, loc="upper left")

                # ∞ annotations
                if tm:
                    ax_bot.annotate(
                        "\u221e",
                        (20000, tm[-1]),
                        textcoords="offset points",
                        xytext=(8, 0),
                        ha="left",
                        fontsize=9,
                        color="#6b7280",
                    )
                for bv, m, s, c in outlier_car:
                    if bv == 20000:
                        ax_top.annotate(
                            "\u221e",
                            (20000, m),
                            textcoords="offset points",
                            xytext=(8, 0),
                            ha="left",
                            fontsize=9,
                            color="#6b7280",
                        )

            else:
                # ── Regular axis (sections column) ────────────────────
                ax = fig.add_subplot(outer_gs[row_idx, col_idx])
                _plot_on_ax(ax, tb, tm, ts, tc, cp)
                ax.set_xscale("log")
                ax.set_xlabel("Token Budget")
                ax.set_ylabel(ylabel)
                if row_idx == 0:
                    ax.set_title(ylabel, fontsize=12, fontweight="bold")
                if tm:
                    ax.annotate(
                        "\u221e",
                        (20000, tm[-1]),
                        textcoords="offset points",
                        xytext=(8, 0),
                        ha="left",
                        fontsize=9,
                        color="#6b7280",
                    )
                ax.legend(handles=_legend_handles, fontsize=7, loc="upper left")
                ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Budget Saturation: TAU-bench vs CAR-bench",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "budget_saturation_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig15_growth_curves_combined():
    """2×2 grid: skill growth curves.

    Left = TAU (25 traces), right = CAR (129 traces).
    Top = Haiku, bottom = Sonnet.
    Shared y-axis per row.
    """
    tau_haiku = _get_experiment("tau_haiku")
    tau_sonnet = _get_experiment("tau_sonnet")
    car_haiku = _get_experiment("car_haiku")
    car_sonnet = _get_experiment("car_sonnet")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    for row, (tau_exp, car_exp, model_name) in enumerate(
        [
            (tau_haiku, car_haiku, "Haiku 4.5"),
            (tau_sonnet, car_sonnet, "Sonnet 4.6"),
        ]
    ):
        for col, (exp, bench_name) in enumerate(
            [(tau_exp, "TAU-bench"), (car_exp, "CAR-bench")]
        ):
            ax = axes[row, col]
            for group in exp.budget_groups:
                color = BUDGET_COLORS.get(group.budget_label, "#333")
                arrays = [np.array(skill_counts(r)) for r in group.runs]
                if not arrays:
                    continue
                # Derive num_traces from actual data
                num_traces = max(len(a) for a in arrays)
                x = np.arange(num_traces)
                # Pad shorter arrays to max length (shouldn't happen but be safe)
                padded = []
                for a in arrays:
                    if len(a) < num_traces:
                        a = np.pad(a, (0, num_traces - len(a)), constant_values=a[-1])
                    padded.append(a)
                mean = np.mean(padded, axis=0)
                std = np.std(padded, axis=0)
                ax.plot(
                    x,
                    mean,
                    color=color,
                    linewidth=2,
                    label=group.budget_label,
                )
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.1)

            ax.set_xlabel("Trace")
            if col == 0:
                ax.set_ylabel(f"{model_name}\nSkills")
            if row == 0:
                traces_str = (
                    f"{max(len(skill_counts(r)) for g in exp.budget_groups for r in g.runs)}"
                    if exp.budget_groups
                    else "?"
                )
                ax.set_title(
                    f"{bench_name} ({traces_str} traces)",
                    fontsize=12,
                    fontweight="bold",
                )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "Skill Growth Curves: TAU-bench vs CAR-bench",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "growth_curves_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig16_churn_combined():
    """2×2 grid: skill churn (scatter + bar).

    Left = scatter (created vs surviving), right = bar (churn rate).
    Top = Haiku, bottom = Sonnet.
    TAU as circles, CAR as triangles.
    """
    tau_haiku = _get_experiment("tau_haiku")
    tau_sonnet = _get_experiment("tau_sonnet")
    car_haiku = _get_experiment("car_haiku")
    car_sonnet = _get_experiment("car_sonnet")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    for row, (tau_exp, car_exp, model_name) in enumerate(
        [
            (tau_haiku, car_haiku, "Haiku 4.5"),
            (tau_sonnet, car_sonnet, "Sonnet 4.6"),
        ]
    ):
        # ── Scatter panel: created vs surviving ──
        ax = axes[row, 0]
        for exp, marker, bench in [
            (tau_exp, "o", "TAU"),
            (car_exp, "^", "CAR"),
        ]:
            for group in exp.budget_groups:
                color = BUDGET_COLORS.get(group.budget_label, "#333")
                for run in group.runs:
                    if not run.metrics or not run.next_ids:
                        continue
                    final_nid = run.next_ids[-1]
                    final_sk = run.metrics[-1].skill_count
                    ax.scatter(
                        final_nid,
                        final_sk,
                        color=color,
                        marker=marker,
                        s=50,
                        alpha=0.7,
                    )

        lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="No churn")
        ax.set_xlabel("next_id (total skills ever created)")
        ax.set_ylabel("Final skill count (surviving)")
        ax.set_title(f"{model_name}: Skills Created vs Surviving")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # ── Bar panel: churn rate ──
        ax = axes[row, 1]

        # Collect TAU churn rates
        tau_labels = []
        tau_rates = []
        tau_stds = []
        tau_colors = []
        for group in tau_exp.budget_groups:
            run_rates = []
            for run in group.runs:
                if not run.next_ids or not run.metrics:
                    continue
                nid = run.next_ids[-1]
                sk = run.metrics[-1].skill_count
                run_rates.append((nid - sk) / nid if nid > 0 else 0)
            tau_labels.append(group.budget_label)
            tau_rates.append(np.mean(run_rates) if run_rates else 0)
            tau_stds.append(np.std(run_rates) if run_rates else 0)
            tau_colors.append(BUDGET_COLORS.get(group.budget_label, "#333"))

        # Collect CAR churn rates (only budget-500 and no-budget)
        car_data: dict[str, tuple[float, float]] = {}
        for group in car_exp.budget_groups:
            run_rates = []
            for run in group.runs:
                if not run.next_ids or not run.metrics:
                    continue
                nid = run.next_ids[-1]
                sk = run.metrics[-1].skill_count
                run_rates.append((nid - sk) / nid if nid > 0 else 0)
            if run_rates:
                car_data[group.budget_label] = (
                    np.mean(run_rates),
                    np.std(run_rates),
                )

        n = len(tau_labels)
        x = np.arange(n)
        bar_w = 0.35

        # TAU bars
        ax.bar(
            x - bar_w / 2,
            tau_rates,
            bar_w,
            yerr=tau_stds,
            color=tau_colors,
            capsize=3,
            alpha=0.85,
            label="TAU-bench",
            edgecolor="white",
            linewidth=0.5,
        )

        # CAR bars (only where data exists)
        car_rates = []
        car_stds_vals = []
        car_colors_vals = []
        for lbl in tau_labels:
            if lbl in car_data:
                car_rates.append(car_data[lbl][0])
                car_stds_vals.append(car_data[lbl][1])
                car_colors_vals.append(BUDGET_COLORS.get(lbl, "#333"))
            else:
                car_rates.append(0)
                car_stds_vals.append(0)
                car_colors_vals.append("#ffffff")

        # Only plot CAR bars where data exists
        for i, lbl in enumerate(tau_labels):
            if lbl in car_data:
                ax.bar(
                    x[i] + bar_w / 2,
                    car_rates[i],
                    bar_w,
                    yerr=car_stds_vals[i],
                    color=car_colors_vals[i],
                    capsize=3,
                    alpha=0.5,
                    hatch="//",
                    edgecolor=car_colors_vals[i],
                    linewidth=0.5,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(tau_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Churn rate (removed / created)")
        ax.set_title(f"{model_name}: Skill Churn Rate")
        ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_elements = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="#6b7280",
            markersize=8, label="TAU-bench",
        ),
        Line2D(
            [0], [0], marker="^", color="w", markerfacecolor="#6b7280",
            markersize=8, label="CAR-bench",
        ),
        Patch(facecolor="#6b7280", alpha=0.5, hatch="//", label="CAR-bench (bar)"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=3,
        fontsize=10,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Skill Churn: TAU-bench vs CAR-bench",
        fontsize=14,
        fontweight="bold",
        y=1.06,
    )
    fig.tight_layout()
    out = OUT_DIR / "churn_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figures 17-18: Combined compression figures ──────────────────────────


def fig17_compression_distribution_combined():
    """2×3 grid: Opus compression distribution (skills, tokens, % of original).

    Top row = Haiku 4.5, bottom row = Sonnet 4.6.
    Each panel shows raw (uncompressed), Opus (individual mean±std),
    and Opus (consensus) on a log-scaled budget axis.
    """
    budget_vals = []
    for b in SH_BUDGET_ORDER:
        bv = 20000 if b == "no-budget" else int(b.split("-")[1])
        budget_vals.append(bv)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row_idx, (exp_dir, model_name) in enumerate(
        [(HAIKU_DIR, "Haiku 4.5"), (SONNET_DIR, "Sonnet 4.6")]
    ):
        comp_metrics = load_compression_metrics(exp_dir)
        comp_dist = compression_distribution_data(exp_dir)

        # Panel 1: Skills
        ax = axes[row_idx, 0]
        raw_sk = [comp_dist[b]["raw_skills_mean"] for b in SH_BUDGET_ORDER]
        raw_sk_std = [comp_dist[b]["raw_skills_std"] for b in SH_BUDGET_ORDER]
        opus_sk = [comp_dist[b]["skills_mean"] for b in SH_BUDGET_ORDER]
        opus_sk_std = [comp_dist[b]["skills_std"] for b in SH_BUDGET_ORDER]
        cons_sk = [
            comp_metrics.get(f"consensus_{b}", {}).get("skills", 0)
            for b in SH_BUDGET_ORDER
        ]

        ax.errorbar(
            budget_vals, raw_sk, yerr=raw_sk_std, fmt="o-", capsize=4,
            label="Raw (uncompressed)", color="#9ca3af",
        )
        ax.errorbar(
            budget_vals, opus_sk, yerr=opus_sk_std, fmt="o-", capsize=4,
            label="Opus (individual)", color="#2563eb",
        )
        ax.plot(
            budget_vals, cons_sk, "s--", color="#dc2626",
            label="Opus (consensus)", markersize=7,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Token Budget")
        ax.set_ylabel("Skills")
        ax.set_title(f"{model_name}: Skill Counts")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Panel 2: MD Tokens
        ax = axes[row_idx, 1]
        raw_toks = [comp_dist[b]["raw_md_tokens_mean"] for b in SH_BUDGET_ORDER]
        raw_toks_std = [comp_dist[b]["raw_md_tokens_std"] for b in SH_BUDGET_ORDER]
        opus_toks = [
            comp_dist[b]["md_tokens_tiktoken_mean"] for b in SH_BUDGET_ORDER
        ]
        opus_toks_std = [
            comp_dist[b]["md_tokens_tiktoken_std"] for b in SH_BUDGET_ORDER
        ]
        cons_toks = [
            comp_metrics.get(f"consensus_{b}", {}).get("md_tokens_tiktoken", 0)
            for b in SH_BUDGET_ORDER
        ]

        ax.errorbar(
            budget_vals, raw_toks, yerr=raw_toks_std, fmt="o-", capsize=4,
            label="Raw (uncompressed)", color="#9ca3af",
        )
        ax.errorbar(
            budget_vals, opus_toks, yerr=opus_toks_std, fmt="o-", capsize=4,
            label="Opus (individual)", color="#2563eb",
        )
        ax.plot(
            budget_vals, cons_toks, "s--", color="#dc2626",
            label="Opus (consensus)", markersize=7,
        )
        ax.set_xscale("log")
        ax.set_xlabel("Token Budget")
        ax.set_ylabel("Tiktoken Tokens")
        ax.set_title(f"{model_name}: MD Token Counts")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Panel 3: % of Original
        ax = axes[row_idx, 2]
        comp_pcts = [comp_dist[b]["compression_pct_mean"] for b in SH_BUDGET_ORDER]
        comp_pcts_std = [
            comp_dist[b]["compression_pct_std"] for b in SH_BUDGET_ORDER
        ]

        cons_comp = []
        for b in SH_BUDGET_ORDER:
            c = comp_metrics.get(f"consensus_{b}", {})
            raw = c.get("raw_md_tokens", 0)
            opus = c.get("md_tokens_tiktoken", 0)
            cons_comp.append(opus / raw * 100 if raw > 0 else 0)

        ax.errorbar(
            budget_vals, comp_pcts, yerr=comp_pcts_std, fmt="o-", capsize=4,
            label="Individual runs (mean ± std)", color="#2563eb",
        )
        ax.plot(
            budget_vals, cons_comp, "s--", color="#dc2626",
            label="Consensus", markersize=7,
        )
        ax.axhline(45, color="#9ca3af", linestyle="--", alpha=0.5, label="~45% average")
        ax.set_xscale("log")
        ax.set_xlabel("Token Budget")
        ax.set_ylabel("% of Original")
        ax.set_title(f"{model_name}: Opus Compression (% of original)")
        ax.set_ylim(0, 70)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Opus Compression: Individual Runs vs Consensus",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "compression_distribution_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig18_conciseness_combined():
    """1×2 grid: per-skill verbosity (tokens/skill) over traces.

    Left = Haiku 4.5, right = Sonnet 4.6.
    One line per budget with mean±std shading.
    """
    tau_haiku = _get_experiment("tau_haiku")
    tau_sonnet = _get_experiment("tau_sonnet")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for col_idx, (exp, model_name) in enumerate(
        [(tau_haiku, "Haiku 4.5"), (tau_sonnet, "Sonnet 4.6")]
    ):
        ax = axes[col_idx]
        for group in exp.budget_groups:
            color = BUDGET_COLORS.get(group.budget_label, "#333")
            arrays = [np.array(tokens_per_skill(r)) for r in group.runs if r.metrics]
            if not arrays:
                continue
            max_len = max(len(a) for a in arrays)
            # Pad shorter arrays with NaN for alignment
            padded = np.full((len(arrays), max_len), np.nan)
            for i, a in enumerate(arrays):
                padded[i, : len(a)] = a
            mean = np.nanmean(padded, axis=0)
            std = np.nanstd(padded, axis=0)
            x = np.arange(max_len)
            ax.plot(x, mean, color=color, linewidth=2, label=group.budget_label)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.1)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Tokens / Skill")
        ax.set_title(f"{model_name}: Per-Skill Verbosity")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Per-Skill Verbosity: Tokens/Skill Over Traces by Budget",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "conciseness_combined.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 19: Dimension Effect Sizes Across Evaluations ──────────────────


def fig19_dimension_effects():
    """Two-panel figure: (A) per-level Δ baseline, (B) pairwise effect sizes.

    Left panel: each dimension level as a line showing pass^4 minus the
    evaluation's baseline, so readers can see absolute positioning.
    Right panel: pairwise effect sizes with propagated error bars.
    """
    # Dimension-level averages and std devs (pass^4, in pp) from the study
    # Format: (mean, std)
    tau = dict(
        sonnet=(18.1, 4.6), haiku=(16.9, 3.7),
        median=(18.8, 2.5), consensus=(16.2, 7.5), opus_median=(18.8, 2.5),
        nobudget=(18.8, 3.5), b500=(16.2, 4.4),
    )
    car_mixed = dict(
        sonnet=(26.0, 2.0), haiku=(21.7, 4.3),
        median=(21.6, 4.8), consensus=(25.5, 3.1), opus_median=(24.3, 3.6),
        nobudget=(23.9, 4.0), b500=(23.8, 4.2),
    )
    car_base = dict(
        sonnet=(55.3, 5.6), haiku=(46.7, 4.4),
        median=(52.0, 9.9), consensus=(50.0, 5.5), opus_median=(51.0, 1.7),
        nobudget=(49.3, 5.6), b500=(52.7, 7.2),
    )
    car_halluc = dict(
        sonnet=(39.7, 8.1), haiku=(41.3, 5.6),
        median=(37.5, 3.4), consensus=(39.0, 9.3), opus_median=(45.0, 5.0),
        nobudget=(39.0, 9.0), b500=(42.0, 3.6),
    )

    evals_data = [tau, car_mixed, car_base, car_halluc]
    eval_labels = ["TAU", "CAR-mix", "CAR-base", "CAR-hal"]
    baselines = [15.0, 31.3, 48.0, 36.0]  # pass^4 baselines

    x = np.arange(len(eval_labels))

    # ── Panel A: grouped bar chart, per-level Δ baseline ────────────────
    # 7 bars per evaluation, grouped by dimension with a small gap between
    # dimension sub-groups.
    bar_specs = [
        # (label, key, color, hatch)  — ordered by dimension group
        # Source model
        ("Sonnet", "sonnet", C_SONNET, None),
        ("Haiku", "haiku", "#f4a261", None),
        # Budget
        ("No-budget", "nobudget", C_CONSENSUS, None),
        ("Budget-500", "b500", "#6bcf9f", None),
        # Skillbook type (blue-purple family)
        ("Opus-median", "opus_median", "#6366f1", None),
        ("Consensus", "consensus", "#818cf8", None),
        ("Median", "median", "#c7d2fe", None),
    ]
    n_bars = len(bar_specs)
    bar_w = 0.1
    # Positions: 7 bars with small gaps between dimension groups
    # Group offsets: [0,1] gap [2,3] gap [4,5,6]
    bar_offsets = np.array([
        -3.3, -2.2,   # source model
        -0.8, 0.3,    # budget
        1.7, 2.8, 3.9,  # skillbook type
    ]) * bar_w

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(17, 7), gridspec_kw={"width_ratios": [1.1, 1]}
    )

    for j, (label, key, color, hatch) in enumerate(bar_specs):
        deltas = [e[key][0] - bl for e, bl in zip(evals_data, baselines)]
        ax_a.bar(
            x + bar_offsets[j], deltas, bar_w,
            color=color, alpha=0.85, label=label,
            edgecolor="white", linewidth=0.5, zorder=3,
        )

    ax_a.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(eval_labels, fontsize=11)
    ax_a.set_ylabel("\u0394 baseline (pp, pass\u2074)", fontsize=11)
    ax_a.set_title("A.  Per-Level Performance vs Baseline", fontweight="bold")
    ax_a.legend(fontsize=8, framealpha=0.9, ncol=3, loc="lower left")

    # Shade above/below zero
    ylim_a = ax_a.get_ylim()
    ax_a.axhspan(0, ylim_a[1], alpha=0.03, color="green", zorder=0)
    ax_a.axhspan(ylim_a[0], 0, alpha=0.03, color="red", zorder=0)

    # ── Panel B: pairwise effect sizes with error bars ────────────────────

    def effect_and_err(edata, key_a, key_b):
        effs, errs = [], []
        for e in edata:
            ma, sa = e[key_a]
            mb, sb = e[key_b]
            effs.append(ma - mb)
            errs.append(np.sqrt(sa**2 + sb**2))
        return effs, errs

    effect_specs = [
        ("Source model\n(Sonnet \u2212 Haiku)", "sonnet", "haiku"),
        ("Budget\n(no-budget \u2212 500)", "nobudget", "b500"),
        ("Opus compression\n(opus-med \u2212 median)", "opus_median", "median"),
        ("Consensus\n(consensus \u2212 median)", "consensus", "median"),
    ]

    eff_colors = [C_SONNET, C_CONSENSUS, "#6366f1", "#818cf8"]
    eff_markers = ["o", "s", "D", "^"]
    jitter = [-0.08, -0.04, 0.0, 0.04]

    for i, (label, ka, kb) in enumerate(effect_specs):
        effs, errs = effect_and_err(evals_data, ka, kb)
        xj = x + jitter[i]
        ax_b.errorbar(
            xj, effs, yerr=errs,
            marker=eff_markers[i], color=eff_colors[i], linewidth=2,
            markersize=9, label=label, alpha=0.9, zorder=3,
            capsize=4, capthick=1.5, elinewidth=1.2,
        )
        for xi, v, err in zip(xj, effs, errs):
            if v >= 0:
                offset_y = err + 10
            else:
                offset_y = -(err + 14)
            ax_b.annotate(
                f"{v:+.1f}",
                (xi, v),
                textcoords="offset points",
                xytext=(0, offset_y),
                ha="center", fontsize=8, fontweight="bold", color=eff_colors[i],
            )

    # ── Training composition effect (task-separated − mixed) ──────────
    # Difference-in-differences: (mean Δbaseline in III.3) − (mean Δbaseline
    # in III.2), controlling for baseline shifts between evaluation runs.
    # Only defined for CAR-base and CAR-hal (indices 2, 3).
    #   Base:   III.3 mean Δbl = +3.0pp (SD 6.7), III.2 = −16.8pp (SD 9.6)
    #   Halluc: III.3 mean Δbl = +4.5pp (SD 6.4), III.2 = −6.7pp (SD 7.0)
    tc_eff = [19.8, 11.2]  # base, halluc
    tc_err = [
        np.sqrt(9.6**2 + 6.7**2),  # base: propagated SD
        np.sqrt(7.0**2 + 6.4**2),  # halluc: propagated SD
    ]
    tc_x = x[2:4] + 0.08  # CAR-base and CAR-hal only
    tc_color = C_HIGHLIGHT
    ax_b.errorbar(
        tc_x, tc_eff, yerr=tc_err,
        marker="*", color=tc_color, linewidth=2.5,
        markersize=14, label="Training composition\n(task-sep \u2212 mixed)",
        alpha=0.95, zorder=4,
        capsize=5, capthick=2, elinewidth=1.5,
    )
    for xi, v, err in zip(tc_x, tc_eff, tc_err):
        ax_b.annotate(
            f"{v:+.1f}",
            (xi, v),
            textcoords="offset points",
            xytext=(0, err + 10),
            ha="center", fontsize=9, fontweight="bold", color=tc_color,
        )

    ax_b.axhline(0, color=C_BASELINE, linestyle="--", linewidth=1.2, alpha=0.7,
                 zorder=1)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(eval_labels, fontsize=11)
    ax_b.set_ylabel("Effect size (pp, pass\u2074)", fontsize=11)
    ax_b.set_title("B.  Pairwise Dimension Effects (\u00b1 propagated SD)",
                    fontweight="bold")
    ax_b.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    ylim_b = ax_b.get_ylim()
    ax_b.axhspan(0, ylim_b[1], alpha=0.03, color="green", zorder=0)
    ax_b.axhspan(ylim_b[0], 0, alpha=0.03, color="red", zorder=0)

    fig.tight_layout()
    out = OUT_DIR / "dimension_effects.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 0: Summary Panel (hero figure for TL;DR) ──────────────────────


def fig0_summary_panel():
    """3-panel composite summary: fluff rate, what matters, downstream impact.

    Panel A: Opus retention by budget (Haiku flat ~45%, Sonnet gradient)
    Panel B: Dimension effect sizes (training composition dominates)
    Panel C: Best downstream results across benchmarks (grouped bar)
    """
    fig, (ax_a, ax_c, ax_b) = plt.subplots(
        1, 3, figsize=(20, 6.5), gridspec_kw={"width_ratios": [1, 1.1, 1]}
    )

    # ── Panel A: Fluff rate — Opus retention by budget ──────────────────
    # Haiku: flat ~45%, Sonnet: gradient 27%→44%
    budgets_x = np.arange(7)
    haiku_retention = [45.2, 45.7, 44.9, 44.8, 43.5, 48.1, 45.4]
    haiku_std = [4.8, 7.5, 5.6, 5.4, 8.9, 5.0, 10.5]
    sonnet_retention = [27.3, 29.0, 26.4, 32.0, 37.5, 44.6, 44.1]
    sonnet_std = [5.2, 3.2, 5.7, 4.6, 5.3, 6.6, 4.2]

    ax_a.fill_between(
        budgets_x,
        [h - s for h, s in zip(haiku_retention, haiku_std)],
        [h + s for h, s in zip(haiku_retention, haiku_std)],
        alpha=0.15, color=C_HAIKU,
    )
    ax_a.fill_between(
        budgets_x,
        [s - e for s, e in zip(sonnet_retention, sonnet_std)],
        [s + e for s, e in zip(sonnet_retention, sonnet_std)],
        alpha=0.15, color=C_SONNET,
    )
    ax_a.plot(
        budgets_x, haiku_retention, "o-",
        color=C_HAIKU, linewidth=2.5, markersize=7, label="Haiku 4.5",
    )
    ax_a.plot(
        budgets_x, sonnet_retention, "s-",
        color=C_SONNET, linewidth=2.5, markersize=7, label="Sonnet 4.6",
    )
    ax_a.axhline(45, color=C_HAIKU, linestyle=":", alpha=0.5, linewidth=1)
    ax_a.annotate(
        "Haiku flat ~45%", (5.5, 46.5),
        fontsize=9, color=C_HAIKU, fontstyle="italic",
    )
    ax_a.annotate(
        "Sonnet gradient\n27%→44%", (0.5, 30),
        fontsize=9, color=C_SONNET, fontstyle="italic",
    )

    # Shade fluff region
    ax_a.axhspan(0, 50, alpha=0.03, color="red", zorder=0)
    ax_a.axhspan(50, 100, alpha=0.03, color="green", zorder=0)
    ax_a.annotate(
        "← FLUFF removed by Opus", (3, 18), fontsize=9, color="#999",
        fontweight="bold", ha="center",
    )

    ax_a.set_xticks(budgets_x)
    ax_a.set_xticklabels(BUDGET_LABELS)
    ax_a.set_xlabel("Token Budget")
    ax_a.set_ylabel("Opus Retention (%)")
    ax_a.set_ylim(15, 60)
    ax_a.set_title("A. ~55% of a Skillbook Is Fluff", fontweight="bold")
    ax_a.legend(fontsize=9, loc="upper left")

    # ── Panel B: What matters — dimension effect sizes ──────────────────
    # Pairwise effects (pp, pass^4) across 4 evaluations
    dimensions = [
        "Training\ncomposition",
        "Source\nmodel",
        "Opus\ncompression",
        "Consensus\nvs median",
        "Token\nbudget",
    ]
    # Mean absolute effect across evaluations where measured
    # Training comp: only CAR-base (+19.8) and CAR-hal (+11.2) → mean 15.5
    # Source model: TAU +1.2, CAR-mix +4.3, CAR-base +8.6, CAR-hal -1.6 → mean |effect| = 3.9
    # Opus compression: TAU 0.0, CAR-mix +2.7, CAR-base -1.0, CAR-hal +7.5 → mean = 2.3
    # Consensus: TAU -2.6, CAR-mix +3.9, CAR-base -2.0, CAR-hal +1.5 → mean |effect| = 2.5
    # Budget: TAU +2.6, CAR-mix +0.1, CAR-base -3.4, CAR-hal -3.0 → mean |effect| = 2.3
    effect_means = [15.5, 3.9, 2.3, 2.5, 2.3]
    # Show direction consistency: training comp always positive, others flip
    effect_colors = [C_HIGHLIGHT, C_SONNET, "#6366f1", "#818cf8", C_CONSENSUS]

    bars = ax_b.barh(
        np.arange(len(dimensions)), effect_means,
        color=effect_colors, alpha=0.85, edgecolor="white", linewidth=0.5,
        height=0.6,
    )
    # Annotate values
    for i, (bar, val) in enumerate(zip(bars, effect_means)):
        ax_b.text(
            val + 0.3, i, f"{val:.1f}pp",
            va="center", fontsize=10, fontweight="bold",
            color=effect_colors[i],
        )

    ax_b.set_yticks(np.arange(len(dimensions)))
    ax_b.set_yticklabels(dimensions, fontsize=10)
    ax_b.set_xlabel("Mean |Effect Size| (pp, pass⁴)")
    ax_b.set_title("C. Training Data Composition\n    Dominates All Other Dimensions", fontweight="bold")
    ax_b.set_xlim(0, 20)
    ax_b.invert_yaxis()

    # ── Panel C: Downstream impact — best results per benchmark ─────────
    benchmarks = ["TAU-bench", "CAR-base\n(task-sep)", "CAR-halluc\n(task-sep)", "CAR-mixed"]
    baseline_pass4 = [15.0, 48.0, 36.0, 31.3]
    best_pass4 = [25.0, 66.0, 52.0, 29.3]
    best_labels = [
        "sonnet-nobudget\nconsensus",
        "sonnet-500\nmedian",
        "haiku-nobudget\nopus-median",
        "sonnet-nobudget\nopus-median",
    ]
    rel_improvement = [
        f"+{(b - bl) / bl * 100:.0f}%" if b > bl else f"{(b - bl) / bl * 100:.0f}%"
        for b, bl in zip(best_pass4, baseline_pass4)
    ]

    x_c = np.arange(len(benchmarks))
    w = 0.35
    bars_bl = ax_c.bar(
        x_c - w / 2, baseline_pass4, w,
        color=C_BASELINE, alpha=0.7, label="Baseline (no skillbook)",
        edgecolor="white", linewidth=0.5,
    )
    best_colors = [C_CONSENSUS, C_MEDIAN, C_OPUS, C_SONNET]
    bars_best = ax_c.bar(
        x_c + w / 2, best_pass4, w,
        color=best_colors, alpha=0.85,
        edgecolor="white", linewidth=0.5,
    )
    # Manual legend entry for "Best ACE config"
    from matplotlib.patches import Patch
    ax_c.legend(
        handles=[
            bars_bl[0],
            Patch(facecolor=C_HIGHLIGHT, alpha=0.6, label="Best ACE config"),
        ],
        labels=["Baseline (no skillbook)", "Best ACE config"],
        fontsize=9, loc="upper left",
    )

    # Annotate relative improvement
    for i, (bl, best, rel, lbl) in enumerate(
        zip(baseline_pass4, best_pass4, rel_improvement, best_labels)
    ):
        color = "green" if best > bl else "red"
        ax_c.annotate(
            rel,
            (i + w / 2, best + 1),
            ha="center", fontsize=11, fontweight="bold", color=color,
        )

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(benchmarks, fontsize=9.5)
    ax_c.set_ylabel("pass⁴ (%)")
    ax_c.set_title("B. Downstream Impact: Best Config vs Baseline", fontweight="bold")
    ax_c.set_ylim(0, 75)

    fig.tight_layout(w_pad=3)
    out = OUT_DIR / "summary_panel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── RR Study Figures ───────────────────────────────────────────────────────

RR_EVAL_DIR = RESULTS / "rr_car_bench_eval"
CAR_EVAL_DIR_MIXED = RESULTS / "car_bench_eval"

# Colors for RR study
C_RR = "#2563eb"  # blue — RR configs
C_TASKSEP = "#059669"  # green — task-separated
C_MIXED = "#e85d04"  # orange — mixed training


def _load_eval_json(eval_dir, config, task_type):
    """Load raw eval JSON from a config/task_type_test/ directory."""
    import glob as _glob

    pattern = str(eval_dir / config / f"{task_type}_test" / "**" / "*.json")
    files = _glob.glob(pattern, recursive=True)
    if not files:
        # Try without _test suffix
        pattern = str(eval_dir / config / task_type / "**" / "*.json")
        files = _glob.glob(pattern, recursive=True)
    if not files:
        return []
    all_data = []
    for f in files:
        all_data.extend(json.loads(Path(f).read_text()))
    return all_data


def _compute_pass_k(data, k):
    """Fraction of tasks where all first k trials succeed."""
    from collections import defaultdict

    tasks = defaultdict(list)
    for entry in data:
        tasks[entry["task_id"]].append(entry["reward"])
    n_pass = sum(
        1
        for rs in tasks.values()
        if len(rs) >= k and all(r > 0 for r in rs[:k])
    )
    return n_pass / len(tasks) if tasks else 0


def fig20_rr_pass4_comparison():
    """Grouped bar chart: pass^4 across RR, mixed, and task-separated methods.

    Two groups (Base | Hallucination), one bar per config.
    Baseline shown as dashed horizontal line.
    """
    # Config definitions: (label, base_source, halluc_source)
    # base_source = (eval_dir, config_name, task_type)
    # halluc_source = (eval_dir, config_name, task_type)
    configs = [
        {
            "label": "RR opus-median",
            "base": (RR_EVAL_DIR, "rr-opus-median", "base"),
            "halluc": (RR_EVAL_DIR, "rr-opus-median", "hallucination"),
            "color": C_RR,
            "hatch": "",
        },
        {
            "label": "RR run 5",
            "base": (RR_EVAL_DIR, "rr-run5", "base"),
            "halluc": (RR_EVAL_DIR, "rr-run5", "hallucination"),
            "color": C_RR,
            "hatch": "//",
        },
        {
            "label": "RR run 1",
            "base": (RR_EVAL_DIR, "rr-merged", "base"),
            "halluc": (RR_EVAL_DIR, "rr-merged", "hallucination"),
            "color": C_RR,
            "hatch": "\\\\",
        },
        {
            "label": "Task-sep best",
            "base": (CAR_BASE_ONLY_DIR, "sonnet-500-base50-median", "base"),
            "halluc": (
                CAR_HALLUC_ONLY_DIR,
                "sonnet-500-halluc48-consensus",
                "hallucination",
            ),
            "color": C_TASKSEP,
            "hatch": "",
        },
        {
            "label": "Task-sep opus-median",
            "base": (
                CAR_BASE_ONLY_DIR,
                "sonnet-500-base50-opus-median",
                "base",
            ),
            "halluc": (
                CAR_HALLUC_ONLY_DIR,
                "sonnet-500-halluc48-opus-median",
                "hallucination",
            ),
            "color": C_TASKSEP,
            "hatch": "//",
        },
        {
            "label": "Mixed opus-median",
            "base": (CAR_EVAL_DIR_MIXED, "sonnet-nobudget-opus-median", "base"),
            "halluc": (
                CAR_EVAL_DIR_MIXED,
                "sonnet-nobudget-opus-median",
                "hallucination",
            ),
            "color": C_MIXED,
            "hatch": "",
        },
        {
            "label": "Mixed consensus",
            "base": (CAR_EVAL_DIR_MIXED, "sonnet-500-consensus", "base"),
            "halluc": (
                CAR_EVAL_DIR_MIXED,
                "sonnet-500-consensus",
                "hallucination",
            ),
            "color": C_MIXED,
            "hatch": "//",
        },
    ]

    # Compute pass^4 for each config
    base_vals = []
    halluc_vals = []
    for cfg in configs:
        d = _load_eval_json(*cfg["base"])
        base_vals.append(_compute_pass_k(d, 4) * 100)
        d = _load_eval_json(*cfg["halluc"])
        halluc_vals.append(_compute_pass_k(d, 4) * 100)

    # Baseline
    bl_base_data = _load_eval_json(RR_EVAL_DIR, "baseline", "base")
    bl_halluc_data = _load_eval_json(RR_EVAL_DIR, "baseline", "hallucination")
    bl_base = _compute_pass_k(bl_base_data, 4) * 100
    bl_halluc = _compute_pass_k(bl_halluc_data, 4) * 100

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))

    n = len(configs)
    bar_w = 0.11
    group_positions = np.array([0, 1.2])  # Base, Halluc groups

    for i, cfg in enumerate(configs):
        offset = (i - n / 2 + 0.5) * bar_w
        vals = [base_vals[i], halluc_vals[i]]
        bars = ax.bar(
            group_positions + offset,
            vals,
            bar_w * 0.9,
            label=cfg["label"],
            color=cfg["color"],
            hatch=cfg["hatch"],
            edgecolor="white" if not cfg["hatch"] else cfg["color"],
            alpha=0.85,
            linewidth=0.5,
        )
        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )

    # Baseline reference lines
    base_span = (
        group_positions[0] - n / 2 * bar_w - 0.02,
        group_positions[0] + n / 2 * bar_w + 0.02,
    )
    halluc_span = (
        group_positions[1] - n / 2 * bar_w - 0.02,
        group_positions[1] + n / 2 * bar_w + 0.02,
    )
    ax.hlines(
        bl_base, *base_span, colors=C_BASELINE, linestyles="--", linewidth=2,
        label=f"Baseline ({bl_base:.0f}% / {bl_halluc:.0f}%)", zorder=5,
    )
    ax.hlines(
        bl_halluc, *halluc_span, colors=C_BASELINE, linestyles="--",
        linewidth=2, zorder=5,
    )

    ax.set_xticks(group_positions)
    ax.set_xticklabels(["Base Tasks", "Hallucination Tasks"], fontsize=12)
    ax.set_ylabel("pass^4 (%)")
    ax.set_title(
        "pass^4 Accuracy: RR vs Task-Separated vs Mixed Training",
        fontweight="bold",
    )
    ax.set_ylim(0, 75)
    ax.legend(fontsize=8.5, loc="upper right", ncol=2)

    fig.tight_layout()
    out = OUT_DIR / "rr_pass4_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig21_rr_passk_stability():
    """Line chart: pass^k decay curves (k=1..4), two panels (Base | Halluc).

    Shows consistency/stability differences across methods.
    """
    configs = [
        {
            "label": "Baseline",
            "base": (RR_EVAL_DIR, "baseline", "base"),
            "halluc": (RR_EVAL_DIR, "baseline", "hallucination"),
            "color": C_BASELINE,
            "ls": "--",
            "lw": 2.5,
            "marker": "s",
        },
        {
            "label": "RR opus-median",
            "base": (RR_EVAL_DIR, "rr-opus-median", "base"),
            "halluc": (RR_EVAL_DIR, "rr-opus-median", "hallucination"),
            "color": C_RR,
            "ls": "-",
            "lw": 2.5,
            "marker": "o",
        },
        {
            "label": "RR run 5",
            "base": (RR_EVAL_DIR, "rr-run5", "base"),
            "halluc": (RR_EVAL_DIR, "rr-run5", "hallucination"),
            "color": C_RR,
            "ls": "--",
            "lw": 1.5,
            "marker": "^",
        },
        {
            "label": "RR run 1",
            "base": (RR_EVAL_DIR, "rr-merged", "base"),
            "halluc": (RR_EVAL_DIR, "rr-merged", "hallucination"),
            "color": C_RR,
            "ls": ":",
            "lw": 1.5,
            "marker": "v",
        },
        {
            "label": "Task-sep best",
            "base": (CAR_BASE_ONLY_DIR, "sonnet-500-base50-median", "base"),
            "halluc": (
                CAR_HALLUC_ONLY_DIR,
                "sonnet-500-halluc48-consensus",
                "hallucination",
            ),
            "color": C_TASKSEP,
            "ls": "-",
            "lw": 2.5,
            "marker": "D",
        },
        {
            "label": "Task-sep opus-median",
            "base": (
                CAR_BASE_ONLY_DIR,
                "sonnet-500-base50-opus-median",
                "base",
            ),
            "halluc": (
                CAR_HALLUC_ONLY_DIR,
                "sonnet-500-halluc48-opus-median",
                "hallucination",
            ),
            "color": C_TASKSEP,
            "ls": "--",
            "lw": 1.5,
            "marker": "P",
        },
        {
            "label": "Mixed opus-median",
            "base": (CAR_EVAL_DIR_MIXED, "sonnet-nobudget-opus-median", "base"),
            "halluc": (
                CAR_EVAL_DIR_MIXED,
                "sonnet-nobudget-opus-median",
                "hallucination",
            ),
            "color": C_MIXED,
            "ls": "-",
            "lw": 2.5,
            "marker": "X",
        },
        {
            "label": "Mixed consensus",
            "base": (CAR_EVAL_DIR_MIXED, "sonnet-500-consensus", "base"),
            "halluc": (
                CAR_EVAL_DIR_MIXED,
                "sonnet-500-consensus",
                "hallucination",
            ),
            "color": C_MIXED,
            "ls": "--",
            "lw": 1.5,
            "marker": "h",
        },
    ]

    ks = [1, 2, 3, 4]

    fig, (ax_base, ax_halluc) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for cfg in configs:
        # Base panel
        data = _load_eval_json(*cfg["base"])
        base_curve = [_compute_pass_k(data, k) * 100 for k in ks]
        ax_base.plot(
            ks, base_curve,
            color=cfg["color"], linestyle=cfg["ls"], linewidth=cfg["lw"],
            marker=cfg["marker"], markersize=7, label=cfg["label"],
            alpha=0.9,
        )

        # Halluc panel
        data = _load_eval_json(*cfg["halluc"])
        halluc_curve = [_compute_pass_k(data, k) * 100 for k in ks]
        ax_halluc.plot(
            ks, halluc_curve,
            color=cfg["color"], linestyle=cfg["ls"], linewidth=cfg["lw"],
            marker=cfg["marker"], markersize=7, label=cfg["label"],
            alpha=0.9,
        )

    for ax, title in [(ax_base, "Base Tasks"), (ax_halluc, "Hallucination Tasks")]:
        ax.set_xticks(ks)
        ax.set_xticklabels(["pass^1", "pass^2", "pass^3", "pass^4"])
        ax.set_xlabel("Metric Strictness")
        ax.set_title(title, fontweight="bold")
        ax.set_ylim(0, 75)

    ax_base.set_ylabel("Accuracy (%)")
    ax_halluc.legend(fontsize=8, loc="lower left")

    fig.suptitle(
        "pass^k Stability: RR vs Task-Separated vs Mixed Training",
        fontweight="bold", fontsize=14, y=1.02,
    )
    fig.tight_layout()
    out = OUT_DIR / "rr_passk_stability.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}\n")
    fig0_summary_panel()
    fig1_tau_bench()
    fig2_cross_model()
    fig3_stability_spectrum()
    fig4_consensus_variance()
    fig5_compression_waterfall()
    fig6_opus_gap()
    fig7_budget_vs_compression()
    fig8_threshold_calibration()
    fig9_cross_domain()
    fig10_car_bench()
    fig11_car_bench_task_types()
    fig12_task_separated_combined()
    fig14_budget_saturation_combined()
    fig15_growth_curves_combined()
    fig16_churn_combined()
    fig17_compression_distribution_combined()
    fig18_conciseness_combined()
    fig19_dimension_effects()
    fig20_rr_pass4_comparison()
    fig21_rr_passk_stability()
    print(f"\nAll figures saved to {OUT_DIR}")
