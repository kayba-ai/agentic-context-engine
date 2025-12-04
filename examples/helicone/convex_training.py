#!/usr/bin/env python3
"""
Convex Error Pattern Training with ACE

Trains ACE on Convex backend error→fix patterns using the new sample-based
ReplayAgent. Demonstrates how to use historical data without dict lookups.

Features:
- Sample-based ReplayAgent (reads response from sample.metadata)
- Time tracking for training duration
- Learns from real Convex production errors
- Simple validation environment

Setup:
1. Ensure you've run convert_convex_to_ace.py to generate ace_convex_samples.jsonl
2. Set ANTHROPIC_API_KEY for Reflector and SkillManager
3. Run: python convex_training.py

Output:
- Prints training duration and learned insights
- Saves skillbook to convex_learned_skillbook.json
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List
from dotenv import load_dotenv


from ace import (
    ReplayAgent,
    Reflector,
    SkillManager,
    OfflineACE,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    Skillbook,
    LiteLLMClient,
)
from ace.prompts_v2_1 import PromptManager

# Load environment variables
load_dotenv(override=True)


class ConvexEnvironment(TaskEnvironment):
    """
    Simple environment for evaluating Convex error→fix patterns.

    Validates that replayed responses have meaningful content.
    """

    def evaluate(self, sample: Sample, agent_output):
        """
        Evaluate the replayed response.

        Since these are historical responses, we just validate they exist
        and have sufficient content.
        """
        response = agent_output.final_answer.strip()

        # Basic validation
        is_valid = len(response) > 0
        has_content = (
            len(response) > 50
        )  # Convex patterns should have substantial content
        has_sections = any(
            marker in response for marker in ["##", "Initial Attempt", "Error", "Fixed"]
        )

        if is_valid and has_content and has_sections:
            feedback = "✓ Valid error→fix pattern with all sections"
            success = True
        elif is_valid and has_content:
            feedback = "⚠ Response has content but may be missing standard sections"
            success = True
        else:
            feedback = "✗ Response is too short or invalid"
            success = False

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=sample.ground_truth,
            metrics={
                "valid": is_valid,
                "has_content": has_content,
                "has_sections": has_sections,
                "success": success,
            },
        )


def load_convex_samples(jsonl_path: str, max_lines: int = 20) -> List[Sample]:
    """
    Load Convex training samples from JSONL file.

    Args:
        jsonl_path: Path to ace_convex_samples.jsonl
        max_lines: Number of lines to read (default: 20)

    Returns:
        List of Sample objects with response in metadata
    """
    samples = []

    print(f"📂 Loading samples from: {jsonl_path}")
    print(f"   Reading first {max_lines} lines...\n")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break

            try:
                data = json.loads(line)

                # Create Sample object
                # The 'response' field is already in the data and will be in metadata
                sample = Sample(
                    question=data["question"],
                    context=data.get("context", ""),
                    ground_truth=data.get("ground_truth", ""),
                    metadata={
                        "response": data[
                            "response"
                        ],  # This is the key for ReplayAgent!
                        "line_number": data.get("metadata", {}).get("line_number"),
                        "categories": data.get("metadata", {}).get("categories", []),
                        "has_buggy_code": data.get("metadata", {}).get(
                            "has_buggy_code", False
                        ),
                        "has_error_msg": data.get("metadata", {}).get(
                            "has_error_msg", False
                        ),
                        "has_fixed_code": data.get("metadata", {}).get(
                            "has_fixed_code", False
                        ),
                    },
                )
                samples.append(sample)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Skipping malformed line {i + 1}: {e}")
                continue

    print(f"✅ Loaded {len(samples)} samples\n")
    return samples


def main():
    """Main training loop."""

    print("\n" + "=" * 70)
    print("ACE Training on Convex Error Patterns")
    print("=" * 70 + "\n")

    # Configuration
    DATA_PATH = "../../.private/helicone/ace_convex_training/ace_convex_samples.jsonl"
    MAX_LINES = 400  # Read first 20 lines
    EPOCHS = 1
    MODEL = "claude-sonnet-4-5-20250929"  # Using Sonnet 4.5 for better JSON reliability
    SKILLBOOK_OUTPUT = "convex_learned_skillbook.json"
    CHECKPOINT_INTERVAL = 10  # Save checkpoint every N samples
    CHECKPOINTS_DIR = "checkpoints"

    # Check for API key (needed for Reflector and SkillManager)
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Please set ANTHROPIC_API_KEY in your .env file")
        print("   Required for Reflector and SkillManager\n")
        return

    # 1. Load training samples
    script_dir = Path(__file__).parent
    data_path = script_dir / DATA_PATH

    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print(f"   Run convert_convex_to_ace.py first to generate training data\n")
        return

    samples = load_convex_samples(str(data_path), max_lines=MAX_LINES)

    if not samples:
        print("❌ No samples loaded\n")
        return

    # Create checkpoints directory
    checkpoints_dir = script_dir / CHECKPOINTS_DIR
    checkpoints_dir.mkdir(exist_ok=True)
    print(f"📁 Checkpoints will be saved to: {checkpoints_dir}/\n")

    # 2. Create ACE components
    print("🧠 Initializing ACE components...")
    print(f"   Agent: ReplayAgent (sample-based mode - NEW!)")
    print(f"   Reflector: {MODEL} (with v2 prompts)")
    print(f"   SkillManager: {MODEL} (with v2 prompts)\n")

    # Create LLM client for Reflector and SkillManager
    llm = LiteLLMClient(model=MODEL, temperature=0.7, max_tokens=8000)

    # Create prompt manager for v2 prompts
    prompt_manager = PromptManager(default_version="2.0")

    # Create ReplayAgent with NO dict (sample-based mode!)
    # It will automatically read from sample.metadata['response']
    agent = ReplayAgent()

    # Create ACE adapter
    adapter = OfflineACE(
        skillbook=Skillbook(),
        agent=agent,
        reflector=Reflector(llm, prompt_template=prompt_manager.get_reflector_prompt()),
        skill_manager=SkillManager(
            llm, prompt_template=prompt_manager.get_skill_manager_prompt()
        ),
    )

    # 3. Create environment
    environment = ConvexEnvironment()

    # 4. Run training with time tracking and checkpoints
    print("🔄 Starting ACE training...")
    print(f"   Samples: {len(samples)}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Checkpoint interval: every {CHECKPOINT_INTERVAL} samples\n")

    print("⏱️  Starting timer...\n")
    start_time = time.time()

    # Run the training with automatic checkpointing!
    results = adapter.run(
        samples,
        environment,
        epochs=EPOCHS,
        checkpoint_interval=CHECKPOINT_INTERVAL,
        checkpoint_dir=str(checkpoints_dir),
    )

    end_time = time.time()
    duration_seconds = end_time - start_time
    duration_minutes = duration_seconds / 60

    # 5. Show results
    print("\n" + "=" * 70)
    print("📊 Training Results")
    print("=" * 70)
    total_attempted = len(samples) * EPOCHS
    failed_count = total_attempted - len(results)
    print(f"✅ Processed {len(results)}/{total_attempted} samples")
    if failed_count > 0:
        print(f"⚠️  Skipped {failed_count} failed samples (errors logged)")
    print(
        f"⏱️  Training duration: {duration_seconds:.2f} seconds ({duration_minutes:.2f} minutes)"
    )
    print(
        f"📚 Skillbook now has {len(adapter.skillbook.skills())} learned strategies\n"
    )

    # Calculate success metrics
    if results:
        successful = sum(
            1 for r in results if r.environment_result.metrics.get("success", False)
        )
        success_rate = (successful / len(results)) * 100
        print(f"📈 Success rate: {success_rate:.1f}% ({successful}/{len(results)})")

    # Show learned insights
    if adapter.skillbook.skills():
        print(f"\n🎯 Top Learned Insights:")
        print("-" * 70)

        for i, skill in enumerate(adapter.skillbook.skills()[:10], 1):
            score = skill.helpful - skill.harmful
            print(f"\n{i}. {skill.content}")
            print(f"   Score: +{skill.helpful}/-{skill.harmful} (net: {score:+d})")
    else:
        print("\n⚠️  No strategies learned (skillbook is empty)")

    # 6. Save skillbook
    print("\n" + "-" * 70)
    skillbook_path = script_dir / SKILLBOOK_OUTPUT
    adapter.skillbook.save_to_file(str(skillbook_path))
    print(f"💾 Skillbook saved to: {skillbook_path}")

    # Show skillbook stats
    stats = adapter.skillbook.stats()
    print(f"\n📊 Skillbook Statistics:")
    print(f"   Total skills: {stats['skills']}")
    print(f"   Sections: {stats['sections']}")
    print(f"   Total helpful: {stats['tags']['helpful']}")
    print(f"   Total harmful: {stats['tags']['harmful']}")

    print("\n" + "=" * 70)
    print("✨ Training complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
