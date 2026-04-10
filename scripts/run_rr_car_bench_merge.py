#!/usr/bin/env python3
"""
Merge per-batch RR skillbooks into a single deduplicated skillbook.

Reads all <batch>/skillbook.json from the output directory, concatenates
skills (prefixing sections with batch source), then runs embedding-based
deduplication.

Usage:
    uv run python scripts/run_rr_car_bench_merge.py --output-dir results/rr_car_bench_<timestamp>
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import groupby
from pathlib import Path

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

BATCH_ASSIGNMENT = ROOT / "results" / "traces_car_bench" / "batch_assignment.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory containing per-batch subdirectories with skillbook.json",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.7,
        help="Dedup similarity threshold (default: 0.7)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.exists():
        print(f"ERROR: Output directory not found: {output_dir}")
        sys.exit(1)

    # Load batch names from assignment
    with open(BATCH_ASSIGNMENT) as f:
        assignment = json.load(f)
    batch_names = list(assignment["batches"].keys())

    # Load per-batch skillbooks
    per_batch = {}
    for name in batch_names:
        sb_path = output_dir / name / "skillbook.json"
        if not sb_path.exists():
            print(f"  SKIP: {name} (no skillbook.json)")
            continue
        sb = Skillbook.load_from_file(str(sb_path))
        n_skills = len(sb.skills())
        per_batch[name] = sb
        print(f"  Loaded: {name} — {n_skills} skills")

    if not per_batch:
        print("ERROR: No skillbooks found to merge.")
        sys.exit(1)

    # Merge: concatenate all skills with batch-prefixed sections
    merged = Skillbook()
    total_before = 0
    for batch_name, sb in per_batch.items():
        for skill in sb.skills():
            section = f"[{batch_name}] {skill.section}"
            merged.add_skill(
                section=section,
                content=skill.content,
                metadata={
                    "helpful": skill.helpful,
                    "harmful": skill.harmful,
                    "neutral": skill.neutral,
                },
                justification=skill.justification,
                evidence=skill.evidence,
            )
            total_before += 1

    print(f"\nMerged: {total_before} skills from {len(per_batch)} batches")

    # Deduplicate using embedding similarity
    # The DeduplicationManager works through the SkillManager pipeline (LLM-driven),
    # so for standalone merge we use SimilarityDetector directly: find pairs above
    # threshold, remove the shorter/lower-quality skill from each pair.
    config = DeduplicationConfig(
        enabled=True,
        similarity_threshold=args.threshold,
        embedding_model="text-embedding-3-small",
        within_section_only=False,  # cross-batch dedup
    )
    detector = SimilarityDetector(config)
    detector.ensure_embeddings(merged)
    pairs = detector.detect_similar_pairs(merged, threshold=args.threshold)
    print(f"Found {len(pairs)} similar pairs (threshold={args.threshold})")

    # Remove the shorter skill from each pair (heuristic: longer = more detailed)
    to_remove: set[str] = set()
    for skill_a, skill_b, sim in pairs:
        if skill_a.id in to_remove or skill_b.id in to_remove:
            continue  # already marked
        # Keep the longer one
        if len(skill_a.content) >= len(skill_b.content):
            to_remove.add(skill_b.id)
        else:
            to_remove.add(skill_a.id)

    for skill_id in to_remove:
        merged.remove_skill(skill_id)

    removed = len(to_remove)
    total_after = len(merged.skills())
    print(f"Dedup: {total_before} → {total_after} skills ({removed} removed)")

    # Save
    merged.save_to_file(str(output_dir / "skillbook_merged.json"))

    # Markdown
    skills = merged.skills()
    md_path = output_dir / "skills_merged.md"
    with open(md_path, "w") as f:
        f.write("# RR CAR-bench — Merged Skillbook\n\n")
        f.write(f"Batches: {len(per_batch)} | Skills: {total_after}\n")
        f.write(f"Before dedup: {total_before} | Removed: {removed}\n")
        f.write(f"Threshold: {args.threshold}\n\n")
        for section, section_skills in groupby(
            sorted(skills, key=lambda s: s.section), key=lambda s: s.section
        ):
            f.write(f"## {section}\n\n")
            for skill in section_skills:
                f.write(f"- {skill.content}\n")
            f.write("\n")

    # External agent injection
    injection = wrap_skillbook_for_external_agent(merged)
    if injection:
        inj_path = output_dir / "external_agent_injection.txt"
        with open(inj_path, "w") as f:
            f.write(injection)

    print(f"\nSaved:")
    print(f"  {output_dir}/skillbook_merged.json")
    print(f"  {output_dir}/skills_merged.md")
    print(f"  {output_dir}/external_agent_injection.txt")


if __name__ == "__main__":
    main()
