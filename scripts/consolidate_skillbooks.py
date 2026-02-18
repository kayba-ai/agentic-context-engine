#!/usr/bin/env python3
"""
Consolidate multiple ACE skillbooks into a single high-quality playbook.

1. Naive-merge all skills from input files (handling ID conflicts, summing counters)
2. Use an LLM to intelligently deduplicate, resolve conflicts, and restructure

Usage:
    uv run python scripts/consolidate_skillbooks.py file1.json file2.json -o output.json
    uv run python scripts/consolidate_skillbooks.py file1.json file2.json -o output.json --model gpt-4.1-2025-04-14
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from ace import Skillbook
from ace.llm_providers import LiteLLMClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="+", help="Skillbook JSON files to consolidate")
    parser.add_argument(
        "-o", "--output", required=True, help="Output path for consolidated skillbook"
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-2025-04-14",
        help="LLM model for intelligent consolidation (default: gpt-4.1-2025-04-14)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="LLM temperature (default: 0.3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only naive-merge, skip LLM consolidation",
    )
    return parser.parse_args()


def naive_merge(skillbooks: List[Skillbook]) -> Skillbook:
    """Merge all skills from multiple skillbooks, handling ID conflicts."""
    merged = Skillbook()
    seen_ids: dict[str, int] = {}

    for sb in skillbooks:
        for skill in sb.skills():
            # Handle ID conflicts by appending a suffix
            base_id = skill.id
            if base_id in seen_ids:
                seen_ids[base_id] += 1
                new_id = f"{base_id}_v{seen_ids[base_id]}"
            else:
                seen_ids[base_id] = 1
                new_id = base_id

            merged.add_skill(
                section=skill.section,
                content=skill.content,
                skill_id=new_id,
                metadata={
                    "helpful": skill.helpful,
                    "harmful": skill.harmful,
                    "neutral": skill.neutral,
                },
                justification=skill.justification,
                evidence=skill.evidence,
            )

    return merged


def llm_consolidate(merged: Skillbook, model: str, temperature: float) -> Skillbook:
    """Use LLM to intelligently consolidate a naive-merged skillbook."""
    client = LiteLLMClient(
        model=model, temperature=temperature, max_tokens=8192, timeout=120
    )

    skills_text = merged._as_markdown_debug()
    skill_count = len(merged.skills())

    prompt = f"""You are an expert at consolidating knowledge bases for AI agents.

Below are {skill_count} skills merged from multiple training runs on airline customer service tasks (TAU-bench).
Many skills overlap, contradict, or could be combined into clearer, more actionable guidance.

Your job: Produce a consolidated set of high-quality skills. For each output skill, provide:
- section: A concise category name
- content: Clear, actionable guidance (1-2 sentences)
- helpful: Sum of helpful counts from merged source skills (or best estimate)
- harmful: Sum of harmful counts from merged source skills (or best estimate)

Guidelines:
1. MERGE duplicates: Combine skills that say the same thing into one stronger skill
2. RESOLVE conflicts: If skills contradict, keep the one with better evidence (higher helpful, lower harmful)
3. GENERALIZE: Turn specific examples into general principles where possible
4. PRUNE: Remove skills that are too vague, harmful, or unsupported
5. Target 15-30 high-quality skills (fewer is better if they cover all key patterns)

## Input Skills

{skills_text}

## Output Format

Return ONLY a JSON array of skill objects. No markdown, no explanation. Example:
[
  {{"section": "booking_changes", "content": "Always verify the passenger name and booking ID before making any changes.", "helpful": 5, "harmful": 0}},
  ...
]"""

    print(f"  Calling {model} for consolidation...")
    response = client.complete(prompt)
    response_text = response.text.strip()

    # Parse JSON from response (handle markdown code blocks)
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        skills_data = json.loads(response_text)
    except json.JSONDecodeError:
        # Try to extract JSON array from the response
        start = response_text.find("[")
        end = response_text.rfind("]") + 1
        if start >= 0 and end > start:
            skills_data = json.loads(response_text[start:end])
        else:
            print(f"Error: Could not parse LLM response as JSON")
            print(f"Response: {response_text[:500]}")
            sys.exit(1)

    # Build consolidated skillbook
    consolidated = Skillbook()
    for i, skill in enumerate(skills_data):
        section = skill.get("section", "general")
        content = skill.get("content", "")
        if not content:
            continue
        consolidated.add_skill(
            section=section,
            content=content,
            metadata={
                "helpful": skill.get("helpful", 1),
                "harmful": skill.get("harmful", 0),
                "neutral": skill.get("neutral", 0),
            },
        )

    return consolidated


def main() -> None:
    args = parse_args()

    # Validate input files
    for f in args.files:
        if not Path(f).exists():
            print(f"Error: File not found: {f}")
            sys.exit(1)

    # Load skillbooks
    print(f"Loading {len(args.files)} skillbook(s)...")
    skillbooks = []
    for f in args.files:
        sb = Skillbook.load_from_file(f)
        skill_count = len(sb.skills())
        print(f"  {f}: {skill_count} skills")
        skillbooks.append(sb)

    # Naive merge
    print("\nNaive merging...")
    merged = naive_merge(skillbooks)
    print(f"  Merged skillbook: {len(merged.skills())} skills")

    if args.dry_run:
        consolidated = merged
        print("  Skipping LLM consolidation (--dry-run)")
    else:
        # LLM consolidation
        print(f"\nLLM consolidation with {args.model}...")
        consolidated = llm_consolidate(merged, args.model, args.temperature)
        print(
            f"  Consolidated: {len(merged.skills())} → {len(consolidated.skills())} skills"
        )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    consolidated.save_to_file(str(output_path))
    print(f"\nSaved to: {output_path}")

    # Print summary
    print(f"\nConsolidated skillbook summary:")
    print(consolidated._as_markdown_debug())


if __name__ == "__main__":
    main()
