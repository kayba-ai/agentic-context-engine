#!/usr/bin/env python3
"""
ACE Deduplication Example

Demonstrates that deduplication works during learning by:
1. Loading a skillbook with known duplicate skills
2. Running a learning cycle (ask + learn)
3. Verifying that similar skills were detected and consolidated

Requires: OPENAI_API_KEY (for embeddings), ANTHROPIC_API_KEY (for LLM)
"""

import os
from dotenv import load_dotenv

from ace import ACELiteLLM, Sample, SimpleEnvironment, DeduplicationConfig, Skillbook

load_dotenv()


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY for embeddings")
        return

    print("=" * 60)
    print("ACELiteLLM DEDUPLICATION DEMO")
    print("=" * 60)

    # Step 1: Load skillbook with known duplicates
    skillbook_path = os.path.join(os.path.dirname(__file__), "test_duplicates.json")

    print("\n1. Loading skillbook with known duplicates...")
    skillbook = Skillbook.load_from_file(skillbook_path)
    skills_before = skillbook.skills()

    print(f"   Loaded {len(skills_before)} skills:")
    for s in skills_before:
        print(f"   [{s.section}] {s.content}")

    # Step 2: Configure agent with deduplication
    dedup_config = DeduplicationConfig(
        enabled=True,
        similarity_threshold=0.60,  # Lowered to catch semantic duplicates
        embedding_model="text-embedding-3-small",
    )

    agent = ACELiteLLM(
        model="claude-sonnet-4-5-20250929",
        dedup_config=dedup_config,
        is_learning=True,
    )
    agent.skillbook = skillbook  # Use our duplicate skillbook

    print(f"\n2. Running learning with deduplication enabled...")
    print(f"   - Similarity threshold: {dedup_config.similarity_threshold}")
    print(f"   - Embedding model: {dedup_config.embedding_model}")

    # Step 3: Run learning (this triggers deduplication)
    samples = [
        Sample(question="What is the capital of France?", ground_truth="Paris"),
    ]
    environment = SimpleEnvironment()

    results = agent.learn(samples, environment, epochs=1)

    # Step 4: Check results
    skills_after = agent.skillbook.skills()

    print(f"\n3. Results:")
    print(f"   - Skills before: {len(skills_before)}")
    print(f"   - Skills after:  {len(skills_after)}")

    print(f"\n   Current skillbook:")
    for s in skills_after:
        print(f"   [{s.section}] {s.content}")

    # Step 5: Verify
    print("\n" + "=" * 60)
    if len(skills_after) < len(skills_before):
        reduction = len(skills_before) - len(skills_after)
        print(f"SUCCESS: Deduplication removed {reduction} duplicate skill(s)")
    elif len(skills_after) == len(skills_before):
        print("INFO: No duplicates removed (similarity may be below threshold)")
        print(
            "   This is expected if embeddings find the skills sufficiently different"
        )
    else:
        print("INFO: Learning added new skills")
    print("=" * 60)


if __name__ == "__main__":
    main()
