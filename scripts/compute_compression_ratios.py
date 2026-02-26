#!/usr/bin/env python3
"""Compute per-run compression ratios for the variance experiment.

For each of the 35 runs (7 budgets x 5 runs):
  - Original MD tokens: tiktoken cl100k_base count of skills_final.md
  - Compressed MD tokens: md_tokens_tiktoken from compression_metrics.json
  - Compression %: (compressed / original) * 100

Prints per-run data and a summary table grouped by budget.
"""

import json
import statistics
from pathlib import Path

import tiktoken

BASE = Path(
    "/scratch/tzerweck/other/Kayba/feature-tzerweck-token-budget/results/variance_experiment"
)
BUDGETS = [
    "budget-500",
    "budget-1000",
    "budget-2000",
    "budget-3000",
    "budget-5000",
    "budget-10000",
    "no-budget",
]
RUNS = range(1, 6)

enc = tiktoken.get_encoding("cl100k_base")

# Load compressed metrics
with open(BASE / "opus_compressed" / "compression_metrics.json") as f:
    metrics = json.load(f)

# Collect per-run data
rows = []
for budget in BUDGETS:
    for run in RUNS:
        # Read original file
        md_path = BASE / budget / f"run_{run}" / "skills_final.md"
        md_text = md_path.read_text()
        original_tokens = len(enc.encode(md_text))

        # Get compressed tokens from metrics
        key = f"{budget}_run_{run}"
        entry = metrics[key]
        assert (
            entry["type"] == "individual"
        ), f"Expected individual, got {entry['type']}"
        compressed_tokens = entry["md_tokens_tiktoken"]

        compression_pct = (compressed_tokens / original_tokens) * 100

        rows.append(
            {
                "budget": budget,
                "run": run,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "compression_pct": compression_pct,
            }
        )

# Print per-run data
print("=" * 90)
print("PER-RUN DATA")
print("=" * 90)
print(
    f"{'Budget':<15} {'Run':>4} {'Original MD':>14} {'Compressed MD':>15} {'Compression %':>15}"
)
print("-" * 90)
for r in rows:
    print(
        f"{r['budget']:<15} {r['run']:>4} {r['original_tokens']:>14,} {r['compressed_tokens']:>15,} {r['compression_pct']:>14.1f}%"
    )

# Group by budget and compute stats
print()
print("=" * 100)
print("SUMMARY TABLE (mean +/- std)")
print("=" * 100)

header = (
    f"| {'Budget':<14} "
    f"| {'Original MD Tokens (mean +/- std)':>35} "
    f"| {'Compressed MD Tokens (mean +/- std)':>37} "
    f"| {'Compression % (mean +/- std)':>30} |"
)
sep = "|" + "-" * 16 + "|" + "-" * 37 + "|" + "-" * 39 + "|" + "-" * 32 + "|"

print(header)
print(sep)

for budget in BUDGETS:
    budget_rows = [r for r in rows if r["budget"] == budget]
    orig_vals = [r["original_tokens"] for r in budget_rows]
    comp_vals = [r["compressed_tokens"] for r in budget_rows]
    pct_vals = [r["compression_pct"] for r in budget_rows]

    orig_mean = statistics.mean(orig_vals)
    orig_std = statistics.stdev(orig_vals)
    comp_mean = statistics.mean(comp_vals)
    comp_std = statistics.stdev(comp_vals)
    pct_mean = statistics.mean(pct_vals)
    pct_std = statistics.stdev(pct_vals)

    orig_str = f"{orig_mean:,.0f} +/- {orig_std:,.0f}"
    comp_str = f"{comp_mean:,.0f} +/- {comp_std:,.0f}"
    pct_str = f"{pct_mean:.1f}% +/- {pct_std:.1f}%"

    print(f"| {budget:<14} | {orig_str:>35} | {comp_str:>37} | {pct_str:>30} |")

print()
print("Note: Compression % = (compressed_tokens / original_tokens) * 100")
print("      Lower % means more compression.")
