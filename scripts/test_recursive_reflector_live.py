#!/usr/bin/env python3
"""Live test script for RecursiveReflector with a real LLM.

This script tests whether the recursive reflector:
1. Explores traces effectively using code execution
2. Generates useful analysis code
3. Produces insights the SkillManager can use

Usage:
    # Requires OPENAI_API_KEY or ANTHROPIC_API_KEY
    uv run python scripts/test_recursive_reflector_live.py
"""

import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ace import Skillbook
from ace.roles import AgentOutput
from ace.reflector import RecursiveReflector, RecursiveConfig
from ace.llm_providers.litellm_client import LiteLLMClient

# Enable debug logging to see REPL iterations
logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-8s %(name)s: %(message)s",
)
# Suppress noisy loggers, keep only reflector
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def run_test():
    """Run the recursive reflector test with a real LLM."""

    # Use a capable model - gpt-4o-mini is fast and cost-effective
    # Alternative: claude-3-haiku-20240307 for Anthropic
    model = os.environ.get("ACE_TEST_MODEL", "gpt-4o-mini")
    print(f"\n{'='*60}")
    print(f"Testing RecursiveReflector with model: {model}")
    print(f"{'='*60}\n")

    llm = LiteLLMClient(model=model)

    # Create a realistic test case with a WRONG answer
    # The agent made a common mistake: assumed largest/most famous = capital
    agent_output = AgentOutput(
        reasoning="""
Let me solve this step by step:
1. The question asks for the capital of Australia
2. Australia is a large country in the southern hemisphere
3. The most famous Australian city is Sydney
4. Sydney is known for its Opera House and Harbour Bridge
5. Sydney has the largest population in Australia
6. Therefore, the capital must be Sydney
        """.strip(),
        final_answer="Sydney",
        skill_ids=["skill-001"],
    )

    # Create a skillbook with a relevant skill that SHOULD have helped
    skillbook = Skillbook()
    skill = skillbook.add_skill(
        section="geography",
        content="When answering geography questions, consider that capitals are not always the largest or most famous cities. Many countries have separate administrative and commercial capitals.",
        skill_id="skill-001",
        metadata={"helpful": 5, "harmful": 0},
    )

    # Configure the recursive reflector
    config = RecursiveConfig(
        max_iterations=5,  # Allow up to 5 REPL iterations
        max_llm_calls=10,  # Allow sub-LLM queries
        timeout=30.0,  # 30 second timeout per code execution
    )

    reflector = RecursiveReflector(llm, config=config)

    print("Test Case:")
    print(f"  Question: What is the capital of Australia?")
    print(f"  Agent's Answer: {agent_output.final_answer}")
    print(f"  Ground Truth: Canberra")
    print(f"  Skill Used: skill-001 (should have helped!)")
    print()

    print("Running recursive reflection...\n")
    print("-" * 60)

    result = reflector.reflect(
        question="What is the capital of Australia?",
        agent_output=agent_output,
        skillbook=skillbook,
        ground_truth="Canberra",
        feedback="Incorrect. The capital of Australia is Canberra, not Sydney. Sydney is the largest city but not the capital.",
    )

    print("-" * 60)
    print("\n=== REFLECTOR OUTPUT ===\n")

    print(f"Reasoning:\n{result.reasoning}\n")
    print(f"Error Identification:\n{result.error_identification}\n")
    print(f"Root Cause Analysis:\n{result.root_cause_analysis}\n")
    print(f"Correct Approach:\n{result.correct_approach}\n")
    print(f"Key Insight:\n{result.key_insight}\n")

    print("Extracted Learnings:")
    if result.extracted_learnings:
        for i, learning in enumerate(result.extracted_learnings, 1):
            print(f"  {i}. {learning.learning}")
            if learning.atomicity_score:
                print(f"     (atomicity: {learning.atomicity_score})")
            if learning.evidence:
                print(f"     Evidence: {learning.evidence}")
    else:
        print("  (none)")
    print()

    print("Skill Tags:")
    if result.skill_tags:
        for tag in result.skill_tags:
            print(f"  - {tag.id}: {tag.tag}")
    else:
        print("  (none)")
    print()

    # Analyze the output
    print("\n=== ANALYSIS ===\n")

    # Check if error was identified
    if (
        "sydney" in result.error_identification.lower()
        or "canberra" in result.error_identification.lower()
    ):
        print("[PASS] Error identification mentions the key cities")
    else:
        print("[WARN] Error identification may be vague")

    # Check if root cause was found
    if (
        "largest" in result.root_cause_analysis.lower()
        or "famous" in result.root_cause_analysis.lower()
        or "population" in result.root_cause_analysis.lower()
    ):
        print("[PASS] Root cause identifies the flawed reasoning pattern")
    else:
        print("[WARN] Root cause may not capture the reasoning flaw")

    # Check skill tagging
    skill_001_tag = next((t for t in result.skill_tags if t.id == "skill-001"), None)
    if skill_001_tag:
        print(f"[INFO] skill-001 tagged as: {skill_001_tag.tag}")
        if skill_001_tag.tag == "helpful":
            print("[PASS] Correctly identified skill would have helped if followed")
        elif skill_001_tag.tag == "harmful":
            print("[FAIL] Skill wrongly tagged as harmful (it should have helped!)")
        else:
            print("[WARN] Skill tagged as neutral/ignored")
    else:
        print("[WARN] skill-001 not tagged")

    # Check for useful learnings
    if result.extracted_learnings:
        has_relevant_learning = any(
            "capital" in l.learning.lower()
            or "largest" in l.learning.lower()
            or "famous" in l.learning.lower()
            for l in result.extracted_learnings
        )
        if has_relevant_learning:
            print("[PASS] Learnings contain relevant insights")
        else:
            print("[WARN] Learnings may not be specific enough")
    else:
        print("[WARN] No learnings extracted")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)

    return result


if __name__ == "__main__":
    run_test()
