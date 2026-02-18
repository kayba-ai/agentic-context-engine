#!/usr/bin/env python3
"""Merge partial TAU-bench re-run results into an existing detailed JSON.

Replaces failed/dead task entries in the original with fresh results from
a partial re-run, then recomputes pass^k metrics over all tasks.

Usage:
    uv run python scripts/merge_tau_results.py \
        --original tau_benchmark_results/tau_airline_..._223116_detailed.json \
        --patch    tau_benchmark_results/tau_airline_..._HHMMSS_detailed.json \
        --output   tau_benchmark_results/tau_airline_..._merged
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path


def pass_hat_k(n: int, s: int, k: int) -> float:
    """Compute pass^k = C(s, k) / C(n, k).

    Args:
        n: Total number of trials.
        s: Number of successes.
        k: k value for pass^k.

    Returns:
        Combinatorial probability that all k chosen trials succeed.
    """
    if k > n or k > s:
        return 0.0
    return comb(s, k) / comb(n, k)


def merge(original: dict, patch: dict) -> dict:
    """Replace task entries in original with matching entries from patch.

    Matching is by task_id. Only tasks present in the patch are replaced.
    """
    patch_by_id = {r["task_id"]: r for r in patch["results"]}

    merged_results = []
    replaced = []
    for task_result in original["results"]:
        tid = task_result["task_id"]
        if tid in patch_by_id:
            merged_results.append(patch_by_id[tid])
            replaced.append(tid)
        else:
            merged_results.append(task_result)

    print(f"Replaced {len(replaced)} tasks: {replaced}")

    # Recompute pass^k metrics
    k = original["k"]
    n_tasks = len(merged_results)
    pass_sums = {str(j): 0.0 for j in range(1, k + 1)}

    for task_result in merged_results:
        trials = task_result["trials"]
        n_trials = len(trials)
        n_successes = sum(1 for t in trials if t.get("success", False))

        # Recompute per-task pass_k_values
        task_pass_k = {}
        for j in range(1, k + 1):
            task_pass_k[str(j)] = pass_hat_k(n_trials, n_successes, j)
        task_result["pass_k_values"] = task_pass_k
        task_result["passed_all"] = all(t.get("success", False) for t in trials)

        for j in range(1, k + 1):
            pass_sums[str(j)] += task_pass_k[str(j)]

    metrics = {}
    for j in range(1, k + 1):
        metrics[f"pass_{j}"] = pass_sums[str(j)] / n_tasks if n_tasks > 0 else 0.0

    merged = {
        "tasks_evaluated": n_tasks,
        "k": k,
        "pass_sums": pass_sums,
        "metrics": metrics,
        "results": merged_results,
    }

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--original", required=True, help="Path to the original detailed JSON"
    )
    parser.add_argument(
        "--patch", required=True, help="Path to the partial re-run detailed JSON"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path prefix (will create _detailed.json and _summary.json)",
    )
    args = parser.parse_args()

    original = json.loads(Path(args.original).read_text())
    patch = json.loads(Path(args.patch).read_text())

    merged = merge(original, patch)

    # Validate: no tasks with 0 steps remaining
    zero_step_tasks = [
        r["task_id"]
        for r in merged["results"]
        if all(t.get("steps", 0) == 0 for t in r["trials"])
    ]
    if zero_step_tasks:
        print(
            f"WARNING: {len(zero_step_tasks)} tasks still have all-zero steps: {zero_step_tasks}"
        )

    # Save detailed
    detailed_path = Path(f"{args.output}_detailed.json")
    detailed_path.write_text(json.dumps(merged, indent=2, default=str))
    print(f"Saved detailed: {detailed_path}")

    # Save summary
    summary = {
        "tasks_evaluated": merged["tasks_evaluated"],
        "k": merged["k"],
        "pass_sums": merged["pass_sums"],
        "metrics": merged["metrics"],
    }
    summary_path = Path(f"{args.output}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")

    # Print metrics
    print(f"\nMerged pass^k metrics ({merged['tasks_evaluated']} tasks):")
    for j in range(1, merged["k"] + 1):
        print(f"  pass^{j}: {merged['metrics'][f'pass_{j}']:.2%}")


if __name__ == "__main__":
    main()
