#!/usr/bin/env python3
"""
Offline ACE Training with Helicone Replay Data

Demonstrates how to train ACE using historical Helicone traces by replaying
actual responses instead of generating new ones. This allows ACE to learn
from past successful interactions without making new LLM calls.

Features:
- Uses ReplayAgent (no LLM calls for generation - free!)
- Automatic Opik observability tracking for all operations
- Learns from real production data

Setup:
1. Place your Helicone JSON export in: ../../.private/helicone/oneline.json
2. Set ANTHROPIC_API_KEY for Reflector and SkillManager (they still need LLM)
3. (Optional) Install Opik for observability: pip install ace-framework[observability]
4. Run: python offline_training_replay.py

Observability:
If Opik is installed, you'll get automatic tracking of:
- Which historical responses are being replayed
- Question coverage (found vs. default responses)
- Reflector and SkillManager learning patterns
- View traces at: https://www.comet.com/opik
"""

import os
import sys
from pathlib import Path
from typing import Dict
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
from helicone_loader import HeliconeLoader

# Load environment variables
load_dotenv()


class SimpleHeliconeEnvironment(TaskEnvironment):
    """
    Simple environment that evaluates replayed responses.

    This is a placeholder - you can customize the evaluation logic
    based on your use case (e.g., checking for errors, quality metrics, etc.)
    """

    def evaluate(self, sample: Sample, agent_output):
        """
        Evaluate the replayed response.

        For now, we assume all replayed responses are valid since they
        came from actual production usage. You can add custom logic here.
        """
        response = agent_output.final_answer.strip()

        # Basic validation
        is_valid = len(response) > 0
        has_content = len(response) > 10

        if is_valid and has_content:
            feedback = "✓ Valid response with substantial content."
            success = True
        elif is_valid:
            feedback = "⚠ Response is short or lacks detail."
            success = False
        else:
            feedback = "✗ Empty or invalid response."
            success = False

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=sample.ground_truth,
            metrics={
                "valid": is_valid,
                "has_content": has_content,
                "success": success,
            },
        )


def build_response_mapping(helicone_trace) -> Dict[str, str]:
    """
    Extract question→response mapping from a Helicone trace.

    Args:
        helicone_trace: HeliconeTrace object from helicone_loader

    Returns:
        Dict mapping questions to their responses
    """
    responses = {}

    # Iterate through conversation turns
    for turn in helicone_trace.conversation:
        # Find user questions
        if turn.role == "user" and turn.text_content:
            question = turn.text_content[0] if turn.text_content else ""

            # Find the next assistant response
            next_turn_idx = turn.turn_index + 1
            if next_turn_idx < len(helicone_trace.conversation):
                next_turn = helicone_trace.conversation[next_turn_idx]
                if next_turn.role == "assistant" and next_turn.text_content:
                    response = (
                        next_turn.text_content[0] if next_turn.text_content else ""
                    )

                    if question and response:
                        responses[question] = response

    return responses


def create_samples_from_trace(helicone_trace) -> list[Sample]:
    """
    Create ACE training samples from a Helicone trace.

    Args:
        helicone_trace: HeliconeTrace object

    Returns:
        List of Sample objects for ACE training
    """
    samples = []

    for turn in helicone_trace.conversation:
        if turn.role == "user" and turn.text_content:
            question = turn.text_content[0] if turn.text_content else ""

            # Get the next assistant response as ground truth
            next_turn_idx = turn.turn_index + 1
            if next_turn_idx < len(helicone_trace.conversation):
                next_turn = helicone_trace.conversation[next_turn_idx]
                if next_turn.role == "assistant" and next_turn.text_content:
                    ground_truth = (
                        next_turn.text_content[0] if next_turn.text_content else ""
                    )

                    if question and ground_truth:
                        sample = Sample(
                            question=question,
                            ground_truth=ground_truth,
                            context=f"Model: {helicone_trace.model}",
                            metadata={
                                "trace_id": helicone_trace.trace_id,
                                "turn_index": turn.turn_index,
                                "model": helicone_trace.model,
                            },
                        )
                        samples.append(sample)

    return samples


def main():
    """Main training loop using Helicone replay data."""

    print("\n" + "=" * 70)
    print("ACE Offline Training with Helicone Replay")
    print("=" * 70 + "\n")

    # Configuration
    HELICONE_JSON_PATH = "../../.private/helicone/oneline.json"
    MAX_SAMPLES = 20  # Start small for testing
    EPOCHS = 2
    MODEL = "claude-sonnet-4-20250514"
    SKILLBOOK_OUTPUT = "helicone_replay_skillbook.json"

    # Check for API key (needed for Reflector and SkillManager)
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Please set ANTHROPIC_API_KEY in your .env file")
        print("   This is required for Reflector and SkillManager")
        return

    # 1. Load Helicone trace
    print(f"📂 Loading Helicone trace from: {HELICONE_JSON_PATH}")
    script_dir = Path(__file__).parent
    json_path = script_dir / HELICONE_JSON_PATH

    if not json_path.exists():
        print(f"❌ File not found: {json_path}")
        print(f"   Please place your Helicone JSON export at this location")
        return

    loader = HeliconeLoader(str(json_path))
    trace = loader.load()

    print(f"✅ Loaded trace: {trace.trace_id}")
    print(f"   Model: {trace.model}")
    print(f"   Total tokens: {trace.total_tokens:,}")
    print(f"   Cost: ${trace.cost:.4f}")
    print(f"   Turns: {len(trace.conversation)}\n")

    # 2. Build response mapping for ReplayAgent
    print("🔄 Building response mapping from trace...")
    response_mapping = build_response_mapping(trace)
    print(f"✅ Extracted {len(response_mapping)} question-response pairs\n")

    if not response_mapping:
        print("❌ No question-response pairs found in trace")
        return

    # 3. Create training samples
    print("📝 Creating training samples...")
    all_samples = create_samples_from_trace(trace)
    samples = all_samples[:MAX_SAMPLES] if MAX_SAMPLES else all_samples
    print(f"✅ Created {len(samples)} training samples\n")

    if not samples:
        print("❌ No samples created from trace")
        return

    # 4. Create ACE components
    print(f"🧠 Initializing ACE components...")
    print(f"   Agent: ReplayAgent (uses historical data)")
    print(f"   Reflector: {MODEL} (with v2 prompts)")
    print(f"   SkillManager: {MODEL} (with v2 prompts)\n")

    # Create LLM client for Reflector and SkillManager
    llm = LiteLLMClient(model=MODEL, temperature=0.7)

    # Create prompt manager for v2 prompts
    prompt_manager = PromptManager(default_version="2.0")

    # Create ACE adapter with ReplayAgent and v2 prompts
    adapter = OfflineACE(
        skillbook=Skillbook(),
        agent=ReplayAgent(response_mapping),
        reflector=Reflector(llm, prompt_template=prompt_manager.get_reflector_prompt()),
        skill_manager=SkillManager(
            llm, prompt_template=prompt_manager.get_skill_manager_prompt()
        ),
    )

    # 5. Create environment
    environment = SimpleHeliconeEnvironment()

    # 6. Run offline adaptation
    print(f"🔄 Starting ACE offline training...")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Samples per epoch: {len(samples)}\n")

    results = adapter.run(samples, environment, epochs=EPOCHS)

    # 7. Show results
    print("\n" + "=" * 70)
    print("📊 Training Results")
    print("=" * 70)
    print(f"✅ Processed {len(results)} samples")
    print(f"📚 Skillbook now has {len(adapter.skillbook.skills())} learned strategies")

    # Calculate success metrics
    if results:
        successful = sum(
            1 for r in results if r.environment_result.metrics.get("success", False)
        )
        success_rate = (successful / len(results)) * 100
        print(f"📈 Success rate: {success_rate:.1f}% ({successful}/{len(results)})")

    # Show top learned strategies
    if adapter.skillbook.skills():
        print(f"\n🎯 Top Learned Strategies:")
        for i, skill in enumerate(adapter.skillbook.skills()[:5], 1):
            score = skill.helpful - skill.harmful
            print(f"\n{i}. {skill.content[:100]}...")
            print(f"   Score: +{skill.helpful}/-{skill.harmful} (net: {score})")

    # 8. Save skillbook
    skillbook_path = Path(SKILLBOOK_OUTPUT)
    adapter.skillbook.save_to_file(str(skillbook_path))
    print(f"\n💾 Skillbook saved to: {skillbook_path}")

    # 9. Show trace analytics
    print(f"\n📊 Original Trace Analytics:")
    print(f"   Cache effectiveness: {trace.get_cache_effectiveness():.1f}%")
    print(f"   Cost per token: ${trace.get_cost_per_token():.6f}")
    print(f"   Latency: {trace.delay_ms}ms")

    tool_stats = trace.get_tool_stats()
    if tool_stats:
        print(f"\n🔧 Tool Usage:")
        for tool, count in sorted(tool_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"   {tool}: {count} calls")

    print("\n" + "=" * 70)
    print("✨ Training complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
