#!/usr/bin/env python3
"""Build consensus skillbooks at alternative thresholds (t≥2, t≥4).

The default analysis (analyze_variance_compression.py) builds consensus at t≥3.
This script reuses its data-loading and clustering pipeline to generate
consensus skillbooks at other thresholds, enabling TAU-bench comparison.

Usage:
    uv run python scripts/build_consensus_thresholds.py \
        --experiment-dir results/variance_experiment_haiku_4.5 --thresholds 2 4
    uv run python scripts/build_consensus_thresholds.py \
        --experiment-dir results/variance_experiment_sonnet_4.6 --thresholds 2 4
    uv run python scripts/build_consensus_thresholds.py \
        --experiment-dir results/variance_experiment_car_haiku_4.5 \
        --budgets no-budget budget-500 --thresholds 2 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse functions from the variance analysis script
import scripts.analyze_variance_compression as avc

from ace.deduplication.config import DeduplicationConfig
from ace.deduplication.detector import SimilarityDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        required=True,
        help="Experiment directory (e.g. results/variance_experiment_haiku_4.5)",
    )
    parser.add_argument(
        "--thresholds",
        type=int,
        nargs="+",
        default=[2, 4],
        help="Consensus thresholds to build (default: 2 4)",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=None,
        help=(
            "Budget labels to process (e.g., no-budget budget-500). "
            "Default: discover from disk or use standard budgets."
        ),
    )
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.is_absolute():
        exp_dir = PROJECT_ROOT / exp_dir

    # Point the module globals at this experiment
    avc.RESULTS_DIR = exp_dir
    avc.CONSENSUS_OUTPUT_DIR = exp_dir / "consensus"

    if args.budgets:
        avc.BUDGETS[:] = args.budgets
        avc.BUDGET_ORDER[:] = args.budgets
    else:
        discovered = avc._discover_budgets(exp_dir)
        if discovered:
            avc.BUDGETS[:] = discovered
            avc.BUDGET_ORDER[:] = discovered

    print(f"Experiment: {exp_dir}")
    print(f"Budgets: {avc.BUDGETS}")
    print(f"Thresholds: {args.thresholds}")

    # Step 1: Load data
    print("\n[1/3] Loading data...")
    data = avc.load_all_data()
    total_runs = sum(len(v) for v in data.values())
    print(f"  Loaded {total_runs} runs across {len(avc.BUDGETS)} budgets")

    # Step 2: Compute embedding clusters
    print("\n[2/3] Computing embedding clusters...")
    config = DeduplicationConfig(
        embedding_provider="litellm",
        embedding_model="text-embedding-3-small",
        similarity_threshold=avc.SIMILARITY_THRESHOLD,
        within_section_only=False,
    )
    detector = SimilarityDetector(config)
    embedding_results = avc.compute_skill_embedding_similarity(data, detector)

    # Step 3: Build & save consensus at each threshold
    print("\n[3/3] Building consensus skillbooks...")
    avc.CONSENSUS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Discover runs dynamically from the first budget with data
    all_runs: set[int] = set()
    for budget in avc.BUDGETS:
        runs = avc.get_valid_runs(budget)
        if runs:
            all_runs = set(runs)
            break
    if not all_runs:
        all_runs = {1, 2, 3, 4, 5}  # fallback

    # Summary table header
    rows = []

    for threshold in args.thresholds:
        print(f"\n  --- threshold t≥{threshold} ---")
        for budget in avc.BUDGETS:
            clusters = embedding_results[budget]["clusters"]
            sb = avc._build_subset_consensus(clusters, all_runs, threshold)

            n_skills = len(sb.skills())
            toon_prompt = sb.as_prompt()
            toon_tokens = avc.count_tokens(toon_prompt)

            # Save JSON
            json_path = (
                avc.CONSENSUS_OUTPUT_DIR / f"consensus_{budget}_t{threshold}.json"
            )
            sb.save_to_file(str(json_path), exclude_embeddings=True)

            # Save markdown
            md_path = avc.CONSENSUS_OUTPUT_DIR / f"consensus_{budget}_t{threshold}.md"
            md_path.write_text(str(sb))

            short = budget.replace("budget-", "").replace("no-budget", "None")
            rows.append((short, threshold, n_skills, toon_tokens))
            print(f"    {short}: {n_skills} skills, {toon_tokens} TOON tokens")

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Budget':<12} {'t':>3} {'Skills':>8} {'TOON tokens':>12}")
    print("-" * 60)
    for budget_short, t, skills, tokens in rows:
        print(f"{budget_short:<12} {t:>3} {skills:>8} {tokens:>12}")
    print("=" * 60)
    print("\nDone!")


if __name__ == "__main__":
    main()
