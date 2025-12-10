#!/usr/bin/env python3
"""
Helicone Data ACE Training Boilerplate

Loads Helicone JSONL request/response logs and trains ACE to learn from them.
Uses OfflineACE for batch learning over the dataset.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv


from ace import (
    LiteLLMClient,
    Agent,
    Reflector,
    SkillManager,
    OfflineACE,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    Skillbook,
)

# Load environment variables
load_dotenv()


class HeliconeEnvironment(TaskEnvironment):
    """
    Simple environment for evaluating responses from Helicone data.

    This is a placeholder that does basic validation.
    You can improve this later to add more sophisticated evaluation logic.
    """

    def evaluate(self, sample: Sample, agent_output):
        """
        Evaluate the generator's response against the ground truth.

        Current logic:
        - Check if response is non-empty
        - Simple similarity check (can be improved)
        """
        response = agent_output.final_answer.strip()
        ground_truth = sample.ground_truth.strip() if sample.ground_truth else ""

        # Basic validation
        is_valid = len(response) > 0

        # Simple correctness check (you can improve this)
        # For now, just check if response is similar in length and non-empty
        has_content = len(response) > 10

        if is_valid and has_content:
            feedback = "Generated a valid response with content."
        elif is_valid:
            feedback = "Response is too short or lacks detail."
        else:
            feedback = "No valid response generated."

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=ground_truth,
            metrics={
                "valid": is_valid,
                "has_content": has_content,
            },
        )


def load_helicone_samples(
    jsonl_path: str, max_samples: Optional[int] = 100
) -> List[Sample]:
    """
    Load and parse Helicone JSONL data into ACE samples.

    Args:
        jsonl_path: Path to the Helicone JSONL file
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        List of Sample objects
    """
    samples = []

    print(f"Loading Helicone data from: {jsonl_path}")

    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break

            try:
                data = json.loads(line)

                # Extract request and response
                request_body = data.get("request_body", {})
                response_body = data.get("response_body", {})

                # Extract the last user message as the question
                messages = request_body.get("messages", [])
                user_messages = [m for m in messages if m.get("role") == "user"]

                if not user_messages:
                    continue

                # Get the last user message content
                last_user_msg = user_messages[-1].get("content", "")

                # Extract text from message content (handle both string and list formats)
                if isinstance(last_user_msg, str):
                    question = last_user_msg
                elif isinstance(last_user_msg, list):
                    # Extract text blocks from content list
                    text_blocks = [
                        item.get("text", "")
                        for item in last_user_msg
                        if item.get("type") == "text"
                    ]
                    question = " ".join(text_blocks)
                else:
                    continue

                # Extract the assistant response as ground truth
                response_content = response_body.get("content", "")
                if isinstance(response_content, list):
                    # Handle array format (e.g., from Claude API)
                    text_blocks = [
                        item.get("text", "")
                        for item in response_content
                        if item.get("type") == "text"
                    ]
                    ground_truth = " ".join(text_blocks)
                else:
                    ground_truth = response_content

                if not question or not ground_truth:
                    continue

                # Create sample
                sample = Sample(
                    question=question,
                    ground_truth=ground_truth,
                    context=f"Model: {data.get('model', 'unknown')}",
                    metadata={
                        "request_id": data.get("request_id"),
                        "model": data.get("model"),
                        "total_tokens": data.get("total_tokens"),
                    },
                )
                samples.append(sample)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Skipping malformed line {i}: {e}")
                continue

    print(f"Loaded {len(samples)} samples from Helicone data")
    return samples


def main():
    """Main training loop."""

    # Configuration
    HELICONE_JSONL_PATH = "../../.private/helicone-data-past-7-days-prod-30-10-25.jsonl"
    MAX_SAMPLES = 50  # Start small for testing
    EPOCHS = 1
    MODEL = "claude-sonnet-4-20250514"  # Model from Helicone data
    SKILLBOOK_OUTPUT = "helicone_learned_skillbook.json"

    print("\n" + "=" * 60)
    print("Helicone Data ACE Training")
    print("=" * 60 + "\n")

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Please set ANTHROPIC_API_KEY in your .env file")
        print("   This is required to use Claude models")
        return

    # 1. Load Helicone data
    print(f"📂 Loading samples from Helicone data...")
    samples = load_helicone_samples(HELICONE_JSONL_PATH, max_samples=MAX_SAMPLES)

    if not samples:
        print("❌ No samples loaded. Check the JSONL file path and format.")
        return

    print(f"✅ Loaded {len(samples)} training samples\n")

    # 2. Create LLM client with the model from Helicone data
    print(f"🤖 Creating LLM client with model: {MODEL}")
    llm = LiteLLMClient(model=MODEL, temperature=0.7)

    # 3. Create ACE components
    print("🧠 Initializing ACE components...")
    adapter = OfflineACE(
        skillbook=Skillbook(),
        agent=Agent(llm),
        reflector=Reflector(llm),
        skill_manager=SkillManager(llm),
    )

    # 4. Create environment
    environment = HeliconeEnvironment()

    # 5. Run offline adaptation
    print(f"\n🔄 Starting ACE training for {EPOCHS} epoch(s)...")
    print(f"   Processing {len(samples)} samples...\n")

    results = adapter.run(samples, environment, epochs=EPOCHS)

    # 6. Show results
    print("\n" + "=" * 60)
    print("📊 Training Results")
    print("=" * 60)
    print(f"✅ Trained on {len(results)} samples")
    print(f"📚 Skillbook now has {len(adapter.skillbook.skills())} learned strategies")

    # Show learned strategies
    if adapter.skillbook.skills():
        print("\n🎯 Top Learned Strategies:")
        for i, skill in enumerate(adapter.skillbook.skills()[:5], 1):
            print(f"\n{i}. {skill.content}")
            print(f"   Helpful: {skill.helpful} | Harmful: {skill.harmful}")

    # 7. Save skillbook
    skillbook_path = Path(SKILLBOOK_OUTPUT)
    adapter.skillbook.save_to_file(str(skillbook_path))
    print(f"\n💾 Skillbook saved to: {skillbook_path}")

    print("\n" + "=" * 60)
    print("✨ Training complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
