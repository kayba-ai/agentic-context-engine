#!/usr/bin/env python3
"""End-to-end test: insight source tracing with AWS Bedrock.

Runs a real ACE learning loop via Bedrock and verifies that InsightSource
metadata (epoch, sample question, trace refs) is populated on learned skills.

Usage:
    uv run python examples/litellm/bedrock_insight_source_test.py

Requires BEDROCK_API_KEY in .env or environment (bearer token).
"""

import os
import sys
import json

from dotenv import load_dotenv

from ace import ACELiteLLM, Sample, SimpleEnvironment

load_dotenv()


def main() -> None:
    api_key = os.environ.get("BEDROCK_API_KEY")
    if not api_key:
        print("Error: set BEDROCK_API_KEY in .env or environment")
        sys.exit(1)

    model = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    print(f"Model: {model}")

    # 1. Create ACE agent (Bedrock bearer-token auth via api_key)
    agent = ACELiteLLM(model=model, api_key=api_key)

    # 2. Training samples (mix of easy/tricky to trigger varied error types)
    samples = [
        Sample(question="What is 15 * 17?", ground_truth="255"),
        Sample(question="What is the capital of Australia?", ground_truth="Canberra"),
        Sample(
            question="Sort these numbers ascending: 9, 3, 7, 1",
            ground_truth="1, 3, 7, 9",
        ),
        Sample(
            question="What is the chemical symbol for gold?",
            ground_truth="Au",
        ),
    ]

    # 3. Train
    environment = SimpleEnvironment()
    print(f"\nTraining on {len(samples)} samples ...")
    results = agent.learn(samples, environment, epochs=1)

    print(f"Processed {len(results)} samples")
    print(f"Skillbook has {len(agent.skillbook.skills())} skills")

    # 4. Verify insight sources
    source_map = agent.skillbook.source_map()
    summary = agent.skillbook.source_summary()

    print("\n--- Source Map ---")
    print(json.dumps(source_map, indent=2, default=str))

    print("\n--- Source Summary ---")
    print(json.dumps(summary, indent=2))

    # 4b. Show error_identification from sources
    print("\n--- Error Identification Samples ---")
    for skill_id, sources in source_map.items():
        for src in sources:
            err_id = src.get("error_identification")
            if err_id:
                print(f"  {skill_id}: {err_id[:120]}...")

    # 4c. Demo source_filter()
    print("\n--- Source Filter Demo ---")
    for epoch in summary.get("epochs", {}):
        filtered = agent.skillbook.source_filter(epoch=epoch)
        print(f"  epoch={epoch}: {sum(len(v) for v in filtered.values())} source(s)")

    # 4d. Show sample_questions distribution
    print(f"\nSample questions: {summary.get('sample_questions', {})}")

    # Assertions
    skills_with_sources = sum(1 for s in agent.skillbook.skills() if s.sources)
    print(
        f"\nSkills with sources: {skills_with_sources}/{len(agent.skillbook.skills())}"
    )

    if summary["total_sources"] == 0:
        print("FAIL: no insight sources attached to any skill")
        sys.exit(1)

    print("PASS: insight source metadata is populated")

    # 5. Test ask()
    answer = agent.ask("What is 12 * 13?")
    print(f"\nTest: What is 12 * 13?  ->  {answer}")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
