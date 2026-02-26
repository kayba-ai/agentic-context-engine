#!/usr/bin/env python3
"""Variance experiment compression analysis.

Loads all 34 valid skillbooks from the variance experiment (7 budgets x 5 runs,
excluding no-budget/run_1 which used dedup_interval=1), computes:

1. Cross-budget section convergence (Part 1)
2. Within-budget consistency — embedding similarity (Part 2)
3. Consensus skillbook construction + median run identification (Part 3 prep)

Outputs structured tables and saves consensus skillbooks.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ace.deduplication.config import DeduplicationConfig
from ace.deduplication.detector import SimilarityDetector
from ace.skillbook import Skillbook

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = PROJECT_ROOT / "results" / "variance_experiment"
BUDGETS = [
    "budget-500",
    "budget-1000",
    "budget-2000",
    "budget-3000",
    "budget-5000",
    "budget-10000",
    "no-budget",
]
BUDGET_ORDER = BUDGETS  # display order
SIMILARITY_THRESHOLD = 0.7  # matching dedup threshold used during generation
MIN_RUNS_FOR_STABLE = 3  # skill must appear in >= this many runs to be "stable"
CONSENSUS_OUTPUT_DIR = PROJECT_ROOT / "results" / "variance_experiment" / "consensus"


def get_valid_runs(budget: str) -> list[int]:
    """Return valid run numbers for a budget (excludes no-budget/run_1)."""
    if budget == "no-budget":
        return [2, 3, 4, 5]
    return [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------


def load_all_data() -> dict:
    """Load all skillbooks and run_info files.

    Returns dict: {budget: {run_num: {"skillbook": Skillbook, "info": dict, "md": str}}}
    """
    data = {}
    for budget in BUDGETS:
        data[budget] = {}
        for run_num in get_valid_runs(budget):
            run_dir = RESULTS_DIR / budget / f"run_{run_num}"
            sb_path = run_dir / "skillbook_24.json"
            info_path = run_dir / "run_info.json"
            md_path = run_dir / "skills_final.md"

            if not sb_path.exists():
                print(f"WARNING: Missing {sb_path}")
                continue

            sb = Skillbook.load_from_file(str(sb_path))
            info = json.loads(info_path.read_text()) if info_path.exists() else {}
            md = md_path.read_text() if md_path.exists() else ""

            data[budget][run_num] = {
                "skillbook": sb,
                "info": info,
                "md": md,
            }
    return data


# ---------------------------------------------------------------------------
# Step 2: Identify median runs
# ---------------------------------------------------------------------------


def identify_median_runs(data: dict) -> dict[str, int]:
    """For each budget, pick the run closest to mean TOON tokens."""
    median_runs = {}
    for budget in BUDGETS:
        runs = data[budget]
        if not runs:
            continue
        toon_tokens = {}
        for run_num, run_data in runs.items():
            prompt = run_data["skillbook"].as_prompt()
            toon_tokens[run_num] = len(prompt) // 4  # rough estimate
        mean_toon = sum(toon_tokens.values()) / len(toon_tokens)
        # Pick run closest to mean
        closest_run = min(
            toon_tokens.keys(), key=lambda r: abs(toon_tokens[r] - mean_toon)
        )
        median_runs[budget] = closest_run
        print(
            f"  {budget}: median run = run_{closest_run} "
            f"(TOON ~{toon_tokens[closest_run]}, mean ~{mean_toon:.0f})"
        )
    return median_runs


# ---------------------------------------------------------------------------
# Step 3: Cross-budget section analysis (Part 1)
# ---------------------------------------------------------------------------


def cross_budget_section_analysis(data: dict) -> dict:
    """Analyze section presence across budgets.

    Returns:
        {
            "section_presence": {section: {budget: fraction_of_runs}},
            "budget_sections": {budget: {run: set_of_sections}},
            "core_sections": set,  # present in >= 6/7 budgets
            "fringe_sections": set,  # present in <= 2 budgets
        }
    """
    budget_sections = {}  # budget -> {run -> set(sections)}
    all_sections = set()

    for budget in BUDGETS:
        budget_sections[budget] = {}
        for run_num, run_data in data[budget].items():
            sections = set(s.section for s in run_data["skillbook"].skills())
            budget_sections[budget][run_num] = sections
            all_sections.update(sections)

    # For each section, compute fraction of runs that include it per budget
    section_presence = {}
    for section in sorted(all_sections):
        section_presence[section] = {}
        for budget in BUDGETS:
            runs = budget_sections[budget]
            if not runs:
                section_presence[section][budget] = 0.0
                continue
            count = sum(1 for secs in runs.values() if section in secs)
            section_presence[section][budget] = count / len(runs)

    # Core sections: appear in >= 6/7 budgets with >= 50% run frequency
    core_sections = set()
    fringe_sections = set()
    for section, presence in section_presence.items():
        budgets_with_section = sum(1 for b in BUDGETS if presence.get(b, 0) >= 0.5)
        if budgets_with_section >= 6:
            core_sections.add(section)
        elif budgets_with_section <= 2:
            fringe_sections.add(section)

    return {
        "section_presence": section_presence,
        "budget_sections": budget_sections,
        "core_sections": core_sections,
        "fringe_sections": fringe_sections,
        "all_sections": all_sections,
    }


def print_section_heatmap(analysis: dict):
    """Print section presence heatmap."""
    presence = analysis["section_presence"]
    core = analysis["core_sections"]
    fringe = analysis["fringe_sections"]

    # Column headers
    short_names = {
        "budget-500": "500",
        "budget-1000": "1000",
        "budget-2000": "2000",
        "budget-3000": "3000",
        "budget-5000": "5000",
        "budget-10000": "10000",
        "no-budget": "None",
    }

    header = (
        f"{'Section':<35} "
        + " ".join(f"{short_names[b]:>5}" for b in BUDGET_ORDER)
        + "  Category"
    )
    print(header)
    print("-" * len(header))

    for section in sorted(presence.keys()):
        row = f"{section:<35} "
        for budget in BUDGET_ORDER:
            frac = presence[section].get(budget, 0)
            if frac >= 0.8:
                cell = "████"
            elif frac >= 0.5:
                cell = "▓▓▓ "
            elif frac > 0:
                cell = "░░  "
            else:
                cell = "    "
            row += f"{cell:>5} "

        if section in core:
            row += " CORE"
        elif section in fringe:
            row += " FRINGE"
        else:
            row += " MID"
        print(row)


# ---------------------------------------------------------------------------
# Step 4: Within-budget consistency (Part 2)
# ---------------------------------------------------------------------------


def compute_skill_embedding_similarity(
    data: dict, detector: SimilarityDetector, threshold: float = SIMILARITY_THRESHOLD
) -> dict[str, dict]:
    """Compute embedding-based skill similarity within each budget.

    For each budget:
    - Embed all skills across all runs
    - Match cross-run skills by similarity >= threshold
    - Compute % of skills that are "stable" (appear in >= MIN_RUNS_FOR_STABLE runs)

    Returns: {budget: {"total_skills": int, "stable_skills": int, "stability_pct": float,
                        "clusters": list_of_clusters}}
    Where each cluster is: {"representative": Skill, "members": [(run_num, Skill, similarity)],
                            "run_coverage": int}
    """
    results = {}
    for budget in BUDGETS:
        runs = data[budget]
        n_runs = len(runs)
        print(f"\n  Processing {budget} ({n_runs} runs)...")

        # Collect all skills with run labels
        all_skills = []  # (run_num, skill)
        all_texts = []
        for run_num in sorted(runs.keys()):
            sb = runs[run_num]["skillbook"]
            for skill in sb.skills():
                all_skills.append((run_num, skill))
                all_texts.append(skill.content)

        if not all_texts:
            results[budget] = {
                "total_skills": 0,
                "stable_skills": 0,
                "stability_pct": 0.0,
                "clusters": [],
                "unique_skills_per_run": {},
            }
            continue

        # Batch embed all skills
        print(f"    Embedding {len(all_texts)} skills...")
        embeddings = detector.compute_embeddings_batch(all_texts)

        # Assign embeddings
        for i, (run_num, skill) in enumerate(all_skills):
            skill.embedding = embeddings[i]

        # Greedy clustering: assign each skill to first matching cluster
        clusters = (
            []
        )  # each: {"representative_idx": int, "members": [(idx, similarity)]}
        assigned = set()

        for i in range(len(all_skills)):
            if i in assigned or embeddings[i] is None:
                continue

            cluster = {"representative_idx": i, "members": [(i, 1.0)]}
            assigned.add(i)

            for j in range(i + 1, len(all_skills)):
                if j in assigned or embeddings[j] is None:
                    continue

                sim = detector.cosine_similarity(embeddings[i], embeddings[j])
                if sim >= threshold:
                    cluster["members"].append((j, sim))
                    assigned.add(j)

            clusters.append(cluster)

        # Analyze clusters
        enriched_clusters = []
        total_unique_per_run = defaultdict(int)
        stable_skill_count = 0

        for cluster in clusters:
            members = cluster["members"]
            # Which runs are represented?
            runs_in_cluster = set()
            member_details = []
            for idx, sim in members:
                run_num, skill = all_skills[idx]
                runs_in_cluster.add(run_num)
                member_details.append((run_num, skill, sim))

            run_coverage = len(runs_in_cluster)

            # Pick representative: longest content or highest helpful count
            rep_idx = max(
                [idx for idx, _ in members],
                key=lambda i: (all_skills[i][1].helpful, len(all_skills[i][1].content)),
            )
            rep_run, rep_skill = all_skills[rep_idx]

            enriched = {
                "representative": rep_skill,
                "representative_run": rep_run,
                "members": member_details,
                "run_coverage": run_coverage,
                "runs": runs_in_cluster,
            }
            enriched_clusters.append(enriched)

            if run_coverage >= MIN_RUNS_FOR_STABLE:
                stable_skill_count += 1

        total_skills = len(all_skills)
        n_clusters = len(enriched_clusters)

        results[budget] = {
            "total_skills": total_skills,
            "n_clusters": n_clusters,
            "stable_skills": stable_skill_count,
            "stability_pct": (
                (stable_skill_count / n_clusters * 100) if n_clusters > 0 else 0.0
            ),
            "clusters": enriched_clusters,
        }

        print(
            f"    {total_skills} skills -> {n_clusters} clusters, "
            f"{stable_skill_count} stable ({results[budget]['stability_pct']:.1f}%)"
        )

    return results


# ---------------------------------------------------------------------------
# Step 5: Build consensus skillbooks (Part 3 prep)
# ---------------------------------------------------------------------------


def build_consensus_skillbooks(
    data: dict, embedding_results: dict[str, dict]
) -> dict[str, Skillbook]:
    """Build consensus skillbook per budget.

    Keep only clusters with members from >= MIN_RUNS_FOR_STABLE runs.
    Select representative with longest content or highest helpful count.
    """
    consensus = {}
    for budget in BUDGETS:
        result = embedding_results[budget]
        sb = Skillbook()

        stable_clusters = [
            c for c in result["clusters"] if c["run_coverage"] >= MIN_RUNS_FOR_STABLE
        ]

        for cluster in stable_clusters:
            rep = cluster["representative"]
            # Sum helpful/harmful across all members
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

        consensus[budget] = sb
        print(
            f"  {budget}: {len(stable_clusters)} consensus skills "
            f"(from {result['n_clusters']} total clusters)"
        )

    return consensus


def save_consensus_skillbooks(consensus: dict[str, Skillbook]):
    """Save consensus skillbooks to disk."""
    CONSENSUS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for budget, sb in consensus.items():
        path = CONSENSUS_OUTPUT_DIR / f"consensus_{budget}.json"
        sb.save_to_file(str(path), exclude_embeddings=True)
        print(f"  Saved {path}")

    # Also save markdown versions
    for budget, sb in consensus.items():
        md_path = CONSENSUS_OUTPUT_DIR / f"consensus_{budget}.md"
        md_path.write_text(str(sb))
        print(f"  Saved {md_path}")


# ---------------------------------------------------------------------------
# Step 6: Output summary tables
# ---------------------------------------------------------------------------


def print_summary_table(
    data: dict,
    median_runs: dict[str, int],
    section_analysis: dict,
    embedding_results: dict[str, dict],
    consensus: dict[str, Skillbook],
):
    """Print the comprehensive summary table."""

    print("\n" + "=" * 100)
    print("SUMMARY TABLE: Per-Budget Metrics")
    print("=" * 100)

    header = (
        f"{'Budget':<12} {'Runs':>4} {'Median':>6} "
        f"{'Avg Skills':>10} {'Avg Sections':>12} {'Avg TOON':>10} "
        f"{'Clusters':>8} {'Stable%':>8} "
        f"{'Consensus':>10} {'Cons TOON':>10}"
    )
    print(header)
    print("-" * len(header))

    for budget in BUDGET_ORDER:
        runs = data[budget]
        n_runs = len(runs)
        med = median_runs.get(budget, "?")

        # Averages
        avg_skills = sum(len(r["skillbook"].skills()) for r in runs.values()) / n_runs
        avg_sections = (
            sum(
                len(set(s.section for s in r["skillbook"].skills()))
                for r in runs.values()
            )
            / n_runs
        )
        avg_toon = (
            sum(len(r["skillbook"].as_prompt()) // 4 for r in runs.values()) / n_runs
        )

        emb = embedding_results[budget]
        n_clusters = emb["n_clusters"]
        stability = emb["stability_pct"]

        cons_sb = consensus[budget]
        cons_skills = len(cons_sb.skills())
        cons_toon = len(cons_sb.as_prompt()) // 4

        short = budget.replace("budget-", "").replace("no-budget", "None")
        print(
            f"{short:<12} {n_runs:>4} {med:>6} "
            f"{avg_skills:>10.1f} {avg_sections:>12.1f} {avg_toon:>10.0f} "
            f"{n_clusters:>8} {stability:>7.1f}% "
            f"{cons_skills:>10} {cons_toon:>10}"
        )


def print_stability_breakdown(embedding_results: dict[str, dict]):
    """Print per-budget cluster stability breakdown."""
    print("\n" + "=" * 100)
    print("STABILITY BREAKDOWN: Cluster Run Coverage")
    print("=" * 100)

    for budget in BUDGET_ORDER:
        emb = embedding_results[budget]
        clusters = emb["clusters"]
        n_runs = len(get_valid_runs(budget))

        # Count clusters by run coverage
        coverage_counts = defaultdict(int)
        for c in clusters:
            coverage_counts[c["run_coverage"]] += 1

        short = budget.replace("budget-", "").replace("no-budget", "None")
        print(f"\n  {short} ({emb['n_clusters']} clusters from {n_runs} runs):")
        for cov in sorted(coverage_counts.keys()):
            pct = coverage_counts[cov] / emb["n_clusters"] * 100
            bar = "█" * int(pct / 2)
            label = "STABLE" if cov >= MIN_RUNS_FOR_STABLE else "stochastic"
            print(
                f"    {cov}/{n_runs} runs: {coverage_counts[cov]:>3} clusters ({pct:>5.1f}%) {bar} [{label}]"
            )


def print_median_run_details(data: dict, median_runs: dict[str, int]):
    """Print details about median runs for compression."""
    print("\n" + "=" * 100)
    print("MEDIAN RUNS FOR OPUS COMPRESSION")
    print("=" * 100)

    for budget in BUDGET_ORDER:
        med = median_runs[budget]
        run_data = data[budget][med]
        sb = run_data["skillbook"]
        n_skills = len(sb.skills())
        n_sections = len(set(s.section for s in sb.skills()))
        toon = len(sb.as_prompt()) // 4
        md_chars = len(run_data["md"])

        short = budget.replace("budget-", "").replace("no-budget", "None")
        print(
            f"  {short}: run_{med} — {n_skills} skills, {n_sections} sections, "
            f"~{toon} TOON tokens, {md_chars} MD chars"
        )


def print_consensus_details(consensus: dict[str, Skillbook]):
    """Print details about consensus skillbooks."""
    print("\n" + "=" * 100)
    print("CONSENSUS SKILLBOOKS")
    print("=" * 100)

    for budget in BUDGET_ORDER:
        sb = consensus[budget]
        n_skills = len(sb.skills())
        n_sections = len(set(s.section for s in sb.skills()))
        toon = len(sb.as_prompt()) // 4
        md_chars = len(str(sb))

        short = budget.replace("budget-", "").replace("no-budget", "None")
        print(
            f"  {short}: {n_skills} skills, {n_sections} sections, "
            f"~{toon} TOON tokens, {md_chars} MD chars"
        )


def print_core_fringe_sections(section_analysis: dict):
    """Print core and fringe section lists."""
    print("\n" + "=" * 100)
    print("CORE SECTIONS (present in >= 6/7 budgets at >= 50% run frequency)")
    print("=" * 100)
    for s in sorted(section_analysis["core_sections"]):
        print(f"  - {s}")

    print("\nFRINGE SECTIONS (present in <= 2 budgets at >= 50% run frequency)")
    print("-" * 60)
    for s in sorted(section_analysis["fringe_sections"]):
        # Show which budgets have it
        budgets_with = [
            b.replace("budget-", "").replace("no-budget", "None")
            for b in BUDGETS
            if section_analysis["section_presence"][s].get(b, 0) >= 0.5
        ]
        print(f"  - {s} (in: {', '.join(budgets_with)})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 100)
    print("VARIANCE EXPERIMENT COMPRESSION ANALYSIS")
    print("=" * 100)

    # Initialize embedding detector
    config = DeduplicationConfig(
        embedding_provider="litellm",
        embedding_model="text-embedding-3-small",
        similarity_threshold=SIMILARITY_THRESHOLD,
        within_section_only=False,  # Compare across sections for cross-run matching
    )
    detector = SimilarityDetector(config)

    # Step 1: Load data
    print("\n[1/6] Loading data...")
    data = load_all_data()
    total_runs = sum(len(v) for v in data.values())
    print(f"  Loaded {total_runs} runs across {len(BUDGETS)} budgets")

    # Step 2: Identify median runs
    print("\n[2/6] Identifying median runs...")
    median_runs = identify_median_runs(data)

    # Step 3: Cross-budget section analysis
    print("\n[3/6] Cross-budget section analysis...")
    section_analysis = cross_budget_section_analysis(data)
    print(f"  {len(section_analysis['all_sections'])} unique sections found")
    print(
        f"  {len(section_analysis['core_sections'])} core, "
        f"{len(section_analysis['fringe_sections'])} fringe"
    )

    # Step 4: Within-budget consistency
    print("\n[4/6] Within-budget consistency (embeddings)...")
    embedding_results = compute_skill_embedding_similarity(data, detector)

    # Step 5: Build consensus skillbooks
    print("\n[5/6] Building consensus skillbooks...")
    consensus = build_consensus_skillbooks(data, embedding_results)

    # Step 6: Save and report
    print("\n[6/6] Saving consensus skillbooks...")
    save_consensus_skillbooks(consensus)

    # Print all tables
    print_summary_table(
        data, median_runs, section_analysis, embedding_results, consensus
    )
    print_section_heatmap(section_analysis)
    print_stability_breakdown(embedding_results)
    print_median_run_details(data, median_runs)
    print_consensus_details(consensus)
    print_core_fringe_sections(section_analysis)

    # Save machine-readable results
    results_path = CONSENSUS_OUTPUT_DIR / "analysis_results.json"
    results = {
        "median_runs": median_runs,
        "embedding_similarity": {
            b: {
                "total_skills": v["total_skills"],
                "n_clusters": v["n_clusters"],
                "stable_skills": v["stable_skills"],
                "stability_pct": v["stability_pct"],
            }
            for b, v in embedding_results.items()
        },
        "consensus_stats": {
            b: {
                "skills": len(sb.skills()),
                "sections": len(set(s.section for s in sb.skills())),
                "toon_tokens": len(sb.as_prompt()) // 4,
                "md_chars": len(str(sb)),
            }
            for b, sb in consensus.items()
        },
        "section_analysis": {
            "core": sorted(section_analysis["core_sections"]),
            "fringe": sorted(section_analysis["fringe_sections"]),
            "total_unique": len(section_analysis["all_sections"]),
        },
    }
    results_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nSaved analysis results to {results_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
