"""Generate social-media-ready figure for Post 6: topic stability vs skill wording variance."""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path("results/study_figures/topic_vs_skill_stability.png")

# --- Data (no-budget, 5 identical runs each) ---
#
# Haiku 4.5:
#   Topics: 12 unique themes, 11 in ≥3/5 runs = 91.7% stable
#   Skills: 370 total → 176 clusters, 29 in ≥3/5 runs = 16.5% stable (83.5% unique)
#
# Sonnet 4.6:
#   Topics: 8 unique themes, 7 in ≥3/5 runs = 87.5% stable
#   Skills: 270 total → 111 clusters, 33 in ≥3/5 runs = 29.7% stable (70.3% unique)

models = {
    "Haiku 4.5": {
        "topic_stable": 91.7,
        "skill_stable": 16.5,
        "color": "#2563eb",       # blue
        "color_light": "#93b4f5",
    },
    "Sonnet 4.6": {
        "topic_stable": 87.5,
        "skill_stable": 29.7,
        "color": "#ea580c",       # orange
        "color_light": "#f5a873",
    },
}

C_UNSTABLE = "#e5e7eb"  # light gray

# --- Plot: 1×2 grid ---
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)

y_pos = [1, 0]
y_labels = [
    "Broad topics\n(cancellation, escalation, pricing, ...)",
    "Individual skill\nformulations",
]

for ax, (model_name, d) in zip(axes, models.items()):
    stable = [d["topic_stable"], d["skill_stable"]]
    unstable = [100 - s for s in stable]
    color = d["color"]

    # Background (full bar = 100%)
    ax.barh(y_pos, [100, 100], height=0.5, color=C_UNSTABLE,
            edgecolor="#d1d5db", linewidth=0.5, zorder=1)
    # Stable portion
    ax.barh(y_pos, stable, height=0.5, color=color,
            edgecolor="white", linewidth=0.5, zorder=2, alpha=0.9)

    # Labels inside bars
    for i, (s, u) in enumerate(zip(stable, unstable)):
        if s > 12:
            ax.text(s / 2, y_pos[i], f"{s:.0f}%\nstable",
                    ha="center", va="center", fontsize=11, fontweight="bold",
                    color="white")
        if u > 20:
            ax.text(s + u / 2, y_pos[i], f"{u:.0f}%\nvaried across runs",
                    ha="center", va="center", fontsize=10, color="#6b7280")

    ax.set_xlim(0, 105)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(left=False)
    ax.set_title(model_name, fontsize=12, fontweight="bold", pad=8)

# Y-axis labels only on left panel
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(y_labels, fontsize=10, fontweight="medium")

# Suptitle
fig.suptitle(
    "Same data, same model, 5 runs: What's reproducible?",
    fontsize=14, fontweight="bold", y=1.04,
)

# Footer annotation
fig.text(
    0.98, -0.06,
    "25 traces  ·  no token budget  ·  skill clusters via embedding similarity  ·  stable = found in ≥3 of 5 runs",
    ha="right", va="top", fontsize=7.5, color="#9ca3af", style="italic",
)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
