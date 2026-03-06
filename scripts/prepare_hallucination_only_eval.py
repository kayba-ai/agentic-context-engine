#!/usr/bin/env python3
"""Prepare hallucination-only skillbooks for CAR-bench evaluation.

After running the variance experiment on hallucination-only traces
(results/traces_car_bench_hallucination/), this script:

  1. Loads the final skillbook from all 5 runs per config
  2. Identifies the median run (closest to mean TOON tokens)
  3. Builds consensus (t>=3) via embedding similarity clustering
  4. Saves median markdown, consensus markdown, and prepared wikis

Opus compression is handled separately (via LLM subagents after this script).

Prerequisites:
    # Run variance experiments first (48 hallucination traces, 2 budgets, 2 models):
    uv run python scripts/run_variance_experiment.py \
        --model "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0" \
        --traces-dir results/traces_car_bench_hallucination \
        --output-dir results/variance_experiment_car_hallucination_haiku_4.5 \
        --budgets none 500
    uv run python scripts/run_variance_experiment.py \
        --model "bedrock/us.anthropic.claude-sonnet-4-6" \
        --traces-dir results/traces_car_bench_hallucination \
        --output-dir results/variance_experiment_car_hallucination_sonnet_4.6 \
        --budgets none 500

Usage:
    uv run python scripts/prepare_hallucination_only_eval.py
    uv run python scripts/prepare_hallucination_only_eval.py --dry-run  # Print info only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import tiktoken

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace.deduplication.config import DeduplicationConfig
from ace.deduplication.detector import SimilarityDetector
from ace.skillbook import Skillbook

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_HALLUCINATION_TRACES = 48
SIMILARITY_THRESHOLD = 0.7
MIN_RUNS_FOR_STABLE = 3  # skill must appear in >= 3/5 runs

MODELS = {
    "haiku": "variance_experiment_car_hallucination_haiku_4.5",
    "sonnet": "variance_experiment_car_hallucination_sonnet_4.6",
}
BUDGETS = ["no-budget", "budget-500"]
RUNS = [1, 2, 3, 4, 5]

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "car_bench_eval_hallucination_only" / "skillbooks"
WIKI_ORIG = (
    PROJECT_ROOT
    / "benchmarks"
    / "car-bench"
    / "car-bench"
    / "car_bench"
    / "envs"
    / "car_voice_assistant"
    / "wiki.md"
)
PREPARED_WIKIS_DIR = PROJECT_ROOT / "benchmarks" / "car-bench" / "prepared_wikis"

_enc = tiktoken.encoding_for_model("gpt-4")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


# ---------------------------------------------------------------------------
# Step 1: Load all hallucination-only skillbooks (final snapshots)
# ---------------------------------------------------------------------------


def load_hallucination_skillbooks() -> dict[str, dict[str, dict[int, Skillbook]]]:
    """Load final skillbook from each hallucination-only variance run.

    The final skillbook index is NUM_HALLUCINATION_TRACES - 1 = 47.

    Returns: {model_short: {budget: {run_num: Skillbook}}}
    """
    final_index = NUM_HALLUCINATION_TRACES - 1
    data: dict[str, dict[str, dict[int, Skillbook]]] = {}
    for model_short, model_dir in MODELS.items():
        data[model_short] = {}
        for budget in BUDGETS:
            data[model_short][budget] = {}
            for run_num in RUNS:
                # Try zero-padded first, then plain
                candidates = [
                    RESULTS_DIR
                    / model_dir
                    / budget
                    / f"run_{run_num}"
                    / f"skillbook_{final_index:02d}.json",
                    RESULTS_DIR
                    / model_dir
                    / budget
                    / f"run_{run_num}"
                    / f"skillbook_{final_index}.json",
                ]
                sb_path = None
                for c in candidates:
                    if c.exists():
                        sb_path = c
                        break

                if sb_path is None:
                    print(
                        f"  WARNING: Missing final skillbook for "
                        f"{model_short}/{budget}/run_{run_num} "
                        f"(tried skillbook_{final_index:02d}.json and "
                        f"skillbook_{final_index}.json)"
                    )
                    continue
                sb = Skillbook.load_from_file(str(sb_path))
                data[model_short][budget][run_num] = sb
                print(
                    f"  Loaded {model_short}/{budget}/run_{run_num}: "
                    f"{len(sb.skills())} skills ({sb_path.name})"
                )
    return data


# ---------------------------------------------------------------------------
# Step 2: Identify median runs
# ---------------------------------------------------------------------------


def identify_median_runs(
    data: dict[str, dict[str, dict[int, Skillbook]]],
) -> dict[str, dict[str, int]]:
    """For each config, pick the run closest to mean TOON tokens.

    Returns: {model_short: {budget: run_num}}
    """
    medians: dict[str, dict[str, int]] = {}
    for model_short in MODELS:
        medians[model_short] = {}
        for budget in BUDGETS:
            runs = data[model_short][budget]
            if not runs:
                continue
            toon_tokens = {}
            for run_num, sb in runs.items():
                prompt = sb.as_prompt()
                toon_tokens[run_num] = len(prompt) // 4  # rough char-to-token
            mean_toon = sum(toon_tokens.values()) / len(toon_tokens)
            closest_run = min(
                toon_tokens.keys(), key=lambda r: abs(toon_tokens[r] - mean_toon)
            )
            medians[model_short][budget] = closest_run
            print(
                f"  {model_short}/{budget}: median = run_{closest_run} "
                f"(TOON ~{toon_tokens[closest_run]}, mean ~{mean_toon:.0f})"
            )
    return medians


# ---------------------------------------------------------------------------
# Step 3: Build consensus skillbooks
# ---------------------------------------------------------------------------


def build_consensus_skillbooks(
    data: dict[str, dict[str, dict[int, Skillbook]]],
    detector: SimilarityDetector,
) -> dict[str, dict[str, Skillbook]]:
    """Build consensus skillbook per config via embedding clustering.

    Returns: {model_short: {budget: Skillbook}}
    """
    consensus: dict[str, dict[str, Skillbook]] = {}
    for model_short in MODELS:
        consensus[model_short] = {}
        for budget in BUDGETS:
            runs = data[model_short][budget]
            n_runs = len(runs)
            label = f"{model_short}/{budget}"
            print(f"\n  Clustering {label} ({n_runs} runs)...")

            # Collect all skills with run labels
            all_skills: list[tuple[int, object]] = []
            all_texts: list[str] = []
            for run_num in sorted(runs.keys()):
                sb = runs[run_num]
                for skill in sb.skills():
                    all_skills.append((run_num, skill))
                    all_texts.append(skill.content)

            if not all_texts:
                consensus[model_short][budget] = Skillbook()
                continue

            # Compute embeddings (use cached where available)
            cached = 0
            to_embed_indices: list[int] = []
            to_embed_texts: list[str] = []
            embeddings: list = [None] * len(all_skills)

            for i, (_, skill) in enumerate(all_skills):
                if skill.embedding is not None:
                    embeddings[i] = skill.embedding
                    cached += 1
                else:
                    to_embed_indices.append(i)
                    to_embed_texts.append(all_texts[i])

            if to_embed_texts:
                print(
                    f"    Embedding {len(to_embed_texts)} skills "
                    f"({cached} cached)..."
                )
                new_embeddings = detector.compute_embeddings_batch(to_embed_texts)
                for j, idx in enumerate(to_embed_indices):
                    embeddings[idx] = new_embeddings[j]
                    all_skills[idx][1].embedding = new_embeddings[j]
            else:
                print(f"    All {cached} skills have cached embeddings")

            # Greedy clustering
            clusters: list[dict] = []
            assigned: set[int] = set()

            for i in range(len(all_skills)):
                if i in assigned or embeddings[i] is None:
                    continue
                cluster = {"representative_idx": i, "members": [(i, 1.0)]}
                assigned.add(i)

                for j in range(i + 1, len(all_skills)):
                    if j in assigned or embeddings[j] is None:
                        continue
                    sim = detector.cosine_similarity(embeddings[i], embeddings[j])
                    if sim >= SIMILARITY_THRESHOLD:
                        cluster["members"].append((j, sim))
                        assigned.add(j)
                clusters.append(cluster)

            # Build consensus: keep clusters with coverage >= MIN_RUNS_FOR_STABLE
            sb = Skillbook()
            stable_count = 0

            for cluster in clusters:
                members = cluster["members"]
                runs_in_cluster = set()
                member_details = []
                for idx, sim in members:
                    run_num, skill = all_skills[idx]
                    runs_in_cluster.add(run_num)
                    member_details.append((run_num, skill, sim))

                if len(runs_in_cluster) < MIN_RUNS_FOR_STABLE:
                    continue

                stable_count += 1
                # Pick representative: highest helpful, then longest content
                rep_idx = max(
                    [idx for idx, _ in members],
                    key=lambda i: (
                        all_skills[i][1].helpful,
                        len(all_skills[i][1].content),
                    ),
                )
                _, rep_skill = all_skills[rep_idx]

                total_helpful = sum(s.helpful for _, s, _ in member_details)
                total_harmful = sum(s.harmful for _, s, _ in member_details)
                total_neutral = sum(s.neutral for _, s, _ in member_details)

                sb.add_skill(
                    section=rep_skill.section,
                    content=rep_skill.content,
                    metadata={
                        "helpful": total_helpful,
                        "harmful": total_harmful,
                        "neutral": total_neutral,
                    },
                    justification=rep_skill.justification,
                    evidence=rep_skill.evidence,
                )

            consensus[model_short][budget] = sb
            print(
                f"    {len(all_skills)} skills -> {len(clusters)} clusters, "
                f"{stable_count} stable (consensus)"
            )

    return consensus


# ---------------------------------------------------------------------------
# Step 4: Save skillbooks and prepare wikis
# ---------------------------------------------------------------------------


def config_label(model_short: str, budget: str) -> str:
    """Create a short label like 'haiku-nobudget' or 'sonnet-500'."""
    budget_short = budget.replace("budget-", "").replace("no-budget", "nobudget")
    return f"{model_short}-{budget_short}"


def save_outputs(
    data: dict[str, dict[str, dict[int, Skillbook]]],
    medians: dict[str, dict[str, int]],
    consensus: dict[str, dict[str, Skillbook]],
) -> dict[str, Path]:
    """Save median and consensus skillbooks, prepare wiki variants.

    Returns: {label: skillbook_md_path} for all saved skillbooks.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREPARED_WIKIS_DIR.mkdir(parents=True, exist_ok=True)
    wiki_text = WIKI_ORIG.read_text()

    saved: dict[str, Path] = {}

    for model_short in MODELS:
        for budget in BUDGETS:
            label = config_label(model_short, budget)

            # --- Median skillbook ---
            median_run = medians[model_short][budget]
            median_sb = data[model_short][budget][median_run]
            median_md = str(median_sb)

            median_path = OUTPUT_DIR / f"{label}-halluc48-median.md"
            median_path.write_text(median_md)
            saved[f"{label}-halluc48-median"] = median_path

            median_toon = count_tokens(median_sb.as_prompt())
            median_md_tokens = count_tokens(median_md)
            print(
                f"  {label}-halluc48-median: run_{median_run}, "
                f"{len(median_sb.skills())} skills, "
                f"{median_toon} TOON tokens, {median_md_tokens} MD tokens"
            )

            # Wiki variant for median
            wiki_path = PREPARED_WIKIS_DIR / f"wiki_{label}-halluc48-median.md"
            wiki_path.write_text(
                wiki_text + "\n\n## ACE Learned Skills\n\n" + median_md
            )

            # --- Consensus skillbook ---
            cons_sb = consensus[model_short][budget]
            cons_md = str(cons_sb)

            cons_path = OUTPUT_DIR / f"{label}-halluc48-consensus.md"
            cons_path.write_text(cons_md)
            saved[f"{label}-halluc48-consensus"] = cons_path

            cons_toon = count_tokens(cons_sb.as_prompt())
            cons_md_tokens = count_tokens(cons_md)
            print(
                f"  {label}-halluc48-consensus: "
                f"{len(cons_sb.skills())} skills, "
                f"{cons_toon} TOON tokens, {cons_md_tokens} MD tokens"
            )

            # Wiki variant for consensus
            wiki_path = PREPARED_WIKIS_DIR / f"wiki_{label}-halluc48-consensus.md"
            wiki_path.write_text(
                wiki_text + "\n\n## ACE Learned Skills\n\n" + cons_md
            )

    # Baseline wiki (copy original if not exists)
    baseline_wiki = PREPARED_WIKIS_DIR / "wiki_baseline.md"
    if not baseline_wiki.exists():
        baseline_wiki.write_text(wiki_text)
        print("  Created baseline wiki")

    return saved


# ---------------------------------------------------------------------------
# Step 5: Save JSON metadata for downstream use
# ---------------------------------------------------------------------------


def save_metadata(
    medians: dict[str, dict[str, int]],
    data: dict[str, dict[str, dict[int, Skillbook]]],
    consensus: dict[str, dict[str, Skillbook]],
):
    """Save metadata JSON with median runs and skill counts."""
    meta: dict[str, dict] = {"medians": {}, "configs": {}}

    for model_short in MODELS:
        for budget in BUDGETS:
            label = config_label(model_short, budget)
            median_run = medians[model_short][budget]
            median_sb = data[model_short][budget][median_run]
            cons_sb = consensus[model_short][budget]

            meta["medians"][label] = median_run
            meta["configs"][label] = {
                "median_run": median_run,
                "median_skills": len(median_sb.skills()),
                "median_toon_tokens": count_tokens(median_sb.as_prompt()),
                "consensus_skills": len(cons_sb.skills()),
                "consensus_toon_tokens": count_tokens(cons_sb.as_prompt()),
            }

    meta_path = OUTPUT_DIR / "halluc48_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"\n  Saved metadata to {meta_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Prepare hallucination-only skillbooks for CAR-bench evaluation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print info without saving files",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Hallucination-Only Skillbook Preparation")
    print(f"  Traces: {NUM_HALLUCINATION_TRACES} hallucination traces")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    # Step 1: Load
    print("\n--- Step 1: Loading hallucination-only skillbooks ---")
    data = load_hallucination_skillbooks()

    # Step 2: Identify medians
    print("\n--- Step 2: Identifying median runs ---")
    medians = identify_median_runs(data)

    if args.dry_run:
        print("\n--- DRY RUN: Skipping consensus building and file saving ---")
        for model_short in MODELS:
            for budget in BUDGETS:
                label = config_label(model_short, budget)
                run = medians[model_short][budget]
                sb = data[model_short][budget][run]
                print(f"  {label}: median=run_{run}, {len(sb.skills())} skills")
        return

    # Step 3: Build consensus
    print("\n--- Step 3: Building consensus skillbooks ---")
    dedup_config = DeduplicationConfig(
        embedding_model="text-embedding-3-small",
        similarity_threshold=SIMILARITY_THRESHOLD,
    )
    detector = SimilarityDetector(dedup_config)
    consensus = build_consensus_skillbooks(data, detector)

    # Step 4: Save outputs
    print("\n--- Step 4: Saving skillbooks and preparing wikis ---")
    saved = save_outputs(data, medians, consensus)

    # Step 5: Save metadata
    print("\n--- Step 5: Saving metadata ---")
    save_metadata(medians, data, consensus)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Skillbooks saved to: {OUTPUT_DIR}")
    print(f"  Wiki variants saved to: {PREPARED_WIKIS_DIR}")
    print(f"  Total configs: {len(saved)} (+ baseline)")
    print("\n  Saved files:")
    for label, path in sorted(saved.items()):
        print(f"    {label}: {path.name}")
    print(
        "\n  Next: Run Opus compression on median skillbooks, then "
        "prepare opus-median wiki variants."
    )


if __name__ == "__main__":
    main()
