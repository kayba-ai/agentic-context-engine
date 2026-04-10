#!/usr/bin/env python3
"""
Build consensus skillbook from multiple RR runs.

Loads merged skillbooks from N run directories, clusters skills by embedding
similarity (greedy leader-based, same algorithm as analyze_variance_compression.py),
and keeps clusters appearing in >= threshold runs.

Opus compression of the median is done separately (interactively via Claude chat
in a fresh context window — see RUNBOOK).

Usage:
    uv run python scripts/run_rr_car_bench_consensus.py \
        --run-dirs results/rr_car_bench_*/  \
        --output-dir results/rr_car_bench_consensus

    # Or specify runs explicitly:
    uv run python scripts/run_rr_car_bench_consensus.py \
        --run-dirs results/rr_car_bench_20260309_162529 \
                   results/rr_car_bench_20260310_010000 \
                   ...
        --output-dir results/rr_car_bench_consensus
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import groupby
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load env for OpenAI embeddings
ENV_FILE = Path("/scratch/tzerweck/other/Kayba/.env")
try:
    from dotenv import load_dotenv

    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)
    else:
        load_dotenv()
except ImportError:
    pass

from ace_next import Skillbook
from ace_next.deduplication.detector import SimilarityDetector
from ace_next.protocols.deduplication import DeduplicationConfig
from ace_next.implementations.prompts import wrap_skillbook_for_external_agent


# ---------------------------------------------------------------------------
# Consensus building (same algorithm as analyze_variance_compression.py)
# ---------------------------------------------------------------------------

SIMILARITY_THRESHOLD = 0.7
MIN_RUNS_FOR_STABLE = 3


def load_run_skillbooks(
    run_dirs: list[Path],
) -> list[tuple[int, Skillbook]]:
    """Load merged skillbooks from each run directory.

    Returns list of (run_index, skillbook) tuples.
    """
    result = []
    for i, d in enumerate(sorted(run_dirs)):
        path = d / "skillbook_merged.json"
        if not path.exists():
            print(f"  SKIP: {d.name} (no skillbook_merged.json)")
            continue
        sb = Skillbook.load_from_file(str(path))
        n = len(sb.skills())
        print(f"  Run {i}: {d.name} — {n} skills")
        result.append((i, sb))
    return result


def build_consensus(
    run_skillbooks: list[tuple[int, Skillbook]],
    threshold: int = MIN_RUNS_FOR_STABLE,
    similarity: float = SIMILARITY_THRESHOLD,
) -> Skillbook:
    """Build consensus skillbook from multiple runs.

    Algorithm (same as variance study analyze_variance_compression.py):
    1. Collect all skills across all runs with run labels
    2. Compute embeddings for all skills
    3. Greedy leader-based clustering (cosine >= similarity threshold)
    4. Keep clusters appearing in >= threshold distinct runs
    5. Pick representative: highest helpful count, breaking ties by longest content
    """
    all_skills: list[tuple[int, object]] = []
    for run_idx, sb in run_skillbooks:
        for skill in sb.skills():
            all_skills.append((run_idx, skill))

    n_runs = len(run_skillbooks)
    print(f"\nConsensus: {len(all_skills)} skills from {n_runs} runs")
    print(f"  Clustering threshold: {similarity}")
    print(f"  Min runs for stable: {threshold} of {n_runs}")

    # Compute embeddings
    config = DeduplicationConfig(
        enabled=True,
        similarity_threshold=similarity,
        embedding_model="text-embedding-3-small",
    )
    detector = SimilarityDetector(config)
    texts = [s.content for _, s in all_skills]
    embeddings = detector.compute_embeddings_batch(texts)

    # Greedy leader-based clustering
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
            if sim >= similarity:
                cluster["members"].append((j, sim))
                assigned.add(j)

        clusters.append(cluster)

    # Enrich clusters with run coverage
    enriched = []
    for cluster in clusters:
        runs_in_cluster = set()
        member_details = []
        for idx, sim in cluster["members"]:
            run_idx, skill = all_skills[idx]
            runs_in_cluster.add(run_idx)
            member_details.append((run_idx, skill, sim))

        rep_idx = max(
            [idx for idx, _ in cluster["members"]],
            key=lambda i: (all_skills[i][1].helpful, len(all_skills[i][1].content)),
        )
        _, rep_skill = all_skills[rep_idx]

        enriched.append({
            "representative": rep_skill,
            "members": member_details,
            "run_coverage": len(runs_in_cluster),
            "runs": runs_in_cluster,
        })

    stable = [c for c in enriched if c["run_coverage"] >= threshold]

    print(f"  Total clusters: {len(enriched)}")
    print(f"  Stable clusters (>={threshold} runs): {len(stable)}")

    # Coverage distribution
    coverage_dist: dict[int, int] = {}
    for c in enriched:
        rc = c["run_coverage"]
        coverage_dist[rc] = coverage_dist.get(rc, 0) + 1
    for rc in sorted(coverage_dist):
        print(f"    {rc} run(s): {coverage_dist[rc]} clusters")

    # Build consensus skillbook
    sb = Skillbook()
    for cluster in stable:
        rep = cluster["representative"]
        total_helpful = sum(s.helpful for _, s, _ in cluster["members"])
        total_harmful = sum(s.harmful for _, s, _ in cluster["members"])
        total_neutral = sum(s.neutral for _, s, _ in cluster["members"])

        sb.add_skill(
            section=rep.section,
            content=rep.content,
            metadata={
                "helpful": total_helpful,
                "harmful": total_harmful,
                "neutral": total_neutral,
            },
            justification=rep.justification,
            evidence=rep.evidence,
        )

    return sb


# ---------------------------------------------------------------------------
# Median selection
# ---------------------------------------------------------------------------

def select_median_run(
    run_skillbooks: list[tuple[int, Skillbook]],
    run_dirs: list[Path],
) -> tuple[int, Skillbook, Path]:
    """Select the median run by skill count (closest to mean)."""
    counts = [(idx, len(sb.skills())) for idx, sb in run_skillbooks]
    mean_count = float(np.mean([c for _, c in counts]))
    best_idx, best_count = min(counts, key=lambda x: abs(x[1] - mean_count))
    best_sb = next(sb for idx, sb in run_skillbooks if idx == best_idx)
    best_dir = sorted(run_dirs)[best_idx]
    print(f"\nMedian run: {best_dir.name} ({best_count} skills, mean={mean_count:.1f})")
    return best_idx, best_sb, best_dir


# ---------------------------------------------------------------------------
# Save utilities
# ---------------------------------------------------------------------------

def save_skillbook_md(sb: Skillbook, path: Path, title: str = "Skillbook"):
    """Save skillbook as markdown."""
    skills = sb.skills()
    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Skills: {len(skills)}\n\n")
        for section, section_skills in groupby(
            sorted(skills, key=lambda s: s.section), key=lambda s: s.section
        ):
            f.write(f"## {section}\n\n")
            for skill in section_skills:
                f.write(f"- {skill.content}\n")
            f.write("\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-dirs", type=Path, nargs="+", required=True,
        help="Directories containing per-run skillbook_merged.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for consensus skillbook",
    )
    parser.add_argument(
        "--threshold", type=int, default=MIN_RUNS_FOR_STABLE,
        help=f"Min runs for consensus (default: {MIN_RUNS_FOR_STABLE})",
    )
    parser.add_argument(
        "--similarity", type=float, default=SIMILARITY_THRESHOLD,
        help=f"Clustering similarity threshold (default: {SIMILARITY_THRESHOLD})",
    )
    args = parser.parse_args()

    # Filter to valid run dirs
    run_dirs = [
        d for d in args.run_dirs
        if d.is_dir() and (d / "skillbook_merged.json").exists()
    ]

    if len(run_dirs) < args.threshold:
        print(f"ERROR: Need at least {args.threshold} valid runs, found {len(run_dirs)}")
        print(f"Checked: {[str(d) for d in args.run_dirs]}")
        sys.exit(1)

    print(f"Found {len(run_dirs)} valid runs:")
    run_skillbooks = load_run_skillbooks(run_dirs)

    if len(run_skillbooks) < args.threshold:
        print(f"ERROR: Only {len(run_skillbooks)} runs loaded, need {args.threshold}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Consensus ---
    consensus = build_consensus(
        run_skillbooks,
        threshold=args.threshold,
        similarity=args.similarity,
    )
    consensus.save_to_file(
        str(args.output_dir / "consensus.json"), exclude_embeddings=True
    )
    save_skillbook_md(
        consensus,
        args.output_dir / "consensus.md",
        title=f"RR CAR-bench Consensus (t>={args.threshold}/{len(run_skillbooks)})",
    )

    # External agent injection for consensus
    inj = wrap_skillbook_for_external_agent(consensus)
    if inj:
        (args.output_dir / "consensus_injection.txt").write_text(inj)

    print(f"\n  Saved: {args.output_dir}/consensus.json")
    print(f"  Saved: {args.output_dir}/consensus.md")

    # --- Median selection ---
    median_idx, median_sb, median_dir = select_median_run(run_skillbooks, run_dirs)
    save_skillbook_md(
        median_sb,
        args.output_dir / "median.md",
        title=f"RR CAR-bench Median (run {median_dir.name})",
    )
    median_sb.save_to_file(
        str(args.output_dir / "median.json"), exclude_embeddings=True
    )
    print(f"  Saved: {args.output_dir}/median.md")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Runs: {len(run_skillbooks)}")
    print(f"Consensus (t>={args.threshold}): {len(consensus.skills())} skills")
    print(f"Median run: {median_dir.name} ({len(median_sb.skills())} skills)")
    print(f"Output: {args.output_dir}/")
    print()
    print("Next steps:")
    print("  1. Opus-compress the median skillbook interactively (fresh context):")
    print(f"     cat {args.output_dir}/median.md")
    print("     → paste into Claude chat with compression prompt")
    print(f"     → save output to {args.output_dir}/opus_median.md")
    print("  2. Run evaluation on consensus + opus-median configs")

    # Save metadata
    meta = {
        "n_runs": len(run_skillbooks),
        "run_dirs": [str(d) for d in sorted(run_dirs)],
        "consensus_threshold": args.threshold,
        "similarity_threshold": args.similarity,
        "consensus_skills": len(consensus.skills()),
        "median_run": str(median_dir),
        "median_skills": len(median_sb.skills()),
    }
    with open(args.output_dir / "consensus_config.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
