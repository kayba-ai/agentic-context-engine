"""Example demonstrating skillbook save and load functionality."""

import os
import time

from ace import Skillbook, Sample, OfflineACE, Agent, Reflector, SkillManager
from ace.adaptation import TaskEnvironment, EnvironmentResult
from ace.llm_providers import LiteLLMClient


class SimpleTaskEnvironment(TaskEnvironment):
    """A simple environment for demonstration."""

    def evaluate(self, sample: Sample, agent_output) -> EnvironmentResult:
        # Simple evaluation: check if answer contains expected keyword
        is_correct = sample.ground_truth.lower() in agent_output.final_answer.lower()

        feedback = "Correct!" if is_correct else f"Expected '{sample.ground_truth}'"

        return EnvironmentResult(feedback=feedback, ground_truth=sample.ground_truth)


def train_and_save_skillbook():
    """Train a skillbook and save it to file."""
    print("\n" + "=" * 60)
    print("TRAINING PHASE")
    print("=" * 60)

    # Initialize components (v2.1 prompts are now the default)
    client = LiteLLMClient(model="claude-sonnet-4-5-20250929")
    agent = Agent(client)
    reflector = Reflector(client)
    skill_manager = SkillManager(client)

    # Create offline adapter
    adapter = OfflineACE(agent=agent, reflector=reflector, skill_manager=skill_manager)

    # Create training samples - reasoning questions that benefit from learned strategies
    samples = [
        Sample(
            question="A store sells apples for $2 each. If you buy 5 or more, you get 20% off. How much do 7 apples cost?",
            ground_truth="$11.20",
            context="Multi-step: calculate base price, apply discount",
        ),
        Sample(
            question="A train leaves Station A at 9:00 AM traveling at 60 mph. Another train leaves Station B (180 miles away) at 10:00 AM traveling at 90 mph toward Station A. When do they meet?",
            ground_truth="11:00 AM",
            context="Relative motion problem requiring setup of equations",
        ),
        Sample(
            question="If all birds can fly, and penguins are birds, can penguins fly? Explain the flaw.",
            ground_truth="The premise is false - penguins are birds but cannot fly",
            context="Logic and commonsense reasoning",
        ),
        Sample(
            question="You have 3 boxes labeled apples, oranges, mixed. All labels are wrong. Pick one fruit from one box to correctly label all. Which box?",
            ground_truth="Pick from the box labeled 'mixed' - it must contain only one type",
            context="Logic puzzle requiring deductive reasoning",
        ),
    ]

    # Train with 2 epochs
    environment = SimpleTaskEnvironment()
    print(f"\nTraining on {len(samples)} samples for 2 epochs...")

    start = time.time()
    results = adapter.run(samples, environment, epochs=2)
    train_time = time.time() - start

    # Save the trained skillbook
    skillbook_path = "trained_skillbook.json"
    adapter.skillbook.save_to_file(skillbook_path)

    print(f"\n✅ Training complete in {train_time:.2f}s")
    print(f"   - Samples processed: {len(results)}")
    print(f"   - Strategies learned: {len(adapter.skillbook.skills())}")
    print(f"   - Skillbook saved to: {skillbook_path}")

    return skillbook_path, train_time


def load_and_use_skillbook(skillbook_path):
    """Load a pre-trained skillbook and use it."""
    print("\n" + "=" * 60)
    print("INFERENCE PHASE (using saved skillbook)")
    print("=" * 60)

    # Load the saved skillbook
    print(f"\nLoading skillbook from {skillbook_path}...")
    skillbook = Skillbook.load_from_file(skillbook_path)

    print(f"✅ Loaded skillbook with {len(skillbook.skills())} strategies")

    # Use the loaded skillbook with a new adapter
    client = LiteLLMClient(model="claude-sonnet-4-5-20250929")
    agent = Agent(client)

    # Test with a new reasoning question (similar type to training)
    test_question = "A book costs $15. With a 30% discount, how much do you pay?"
    print(f"\nTest question: {test_question}")

    start = time.time()
    test_output = agent.generate(
        question=test_question,
        context="",
        skillbook=skillbook,
        reflection=None,
    )
    inference_time = time.time() - start

    print(f"   Answer: {test_output.final_answer}")
    print(f"   Strategies used: {len(test_output.skill_ids)}")
    print(f"   Inference time: {inference_time:.2f}s")

    return skillbook, inference_time


def demonstrate_skillbook_inspection(skillbook):
    """Show how to inspect a loaded skillbook."""
    print("\n" + "=" * 50)
    print("SKILLBOOK INSPECTION")
    print("=" * 50)

    # Print skillbook statistics
    stats = skillbook.stats()
    print(f"\nSkillbook Statistics:")
    print(f"  - Sections: {stats['sections']}")
    print(f"  - Total skills: {stats['skills']}")
    print(f"  - Helpful tags: {stats['tags']['helpful']}")
    print(f"  - Harmful tags: {stats['tags']['harmful']}")
    print(f"  - Neutral tags: {stats['tags']['neutral']}")

    # Show skillbook as prompt (first 500 chars)
    prompt_view = skillbook.as_prompt()
    if prompt_view:
        print(f"\nSkillbook as prompt (preview):")
        print("-" * 40)
        print(prompt_view[:500] + "..." if len(prompt_view) > 500 else prompt_view)

    # Show individual skills
    print(f"\nIndividual skills:")
    for skill in skillbook.skills()[:3]:  # Show first 3 skills
        print(f"  [{skill.id}] {skill.content[:60]}...")
        print(f"    Helpful: {skill.helpful}, Harmful: {skill.harmful}")


if __name__ == "__main__":
    print("=" * 60)
    print("SKILLBOOK PERSISTENCE DEMO")
    print("=" * 60)
    print("\nThis demo shows how to save and load trained skillbooks.")
    print("A trained skillbook can be reused across sessions.")

    try:
        # Step 1: Train and save a skillbook
        skillbook_path, train_time = train_and_save_skillbook()

        # Step 2: Load and use the saved skillbook
        loaded_skillbook, inference_time = load_and_use_skillbook(skillbook_path)

        # Step 3: Inspect the skillbook
        demonstrate_skillbook_inspection(loaded_skillbook)

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"\n📊 TIMING:")
        print(f"   - Training time: {train_time:.2f}s")
        print(f"   - Inference time: {inference_time:.2f}s")
        print(f"\n📚 SKILLBOOK:")
        print(f"   - Strategies learned: {len(loaded_skillbook.skills())}")
        print(f"   - Sections: {list(loaded_skillbook._sections.keys())}")
        print(f"\n✅ Skillbook can now be reused without retraining!")

        # Clean up
        if os.path.exists(skillbook_path):
            os.remove(skillbook_path)
            print(f"\n(Cleaned up {skillbook_path})")

    except ImportError:
        print("Note: This example requires an LLM provider to be configured.")
        print("Set your API key: export OPENAI_API_KEY='your-key'")
    except Exception as e:
        print(f"Example requires API keys to be set: {e}")

        # Demonstrate without API calls
        print("\n" + "=" * 50)
        print("DEMONSTRATING SAVE/LOAD WITHOUT API CALLS")
        print("=" * 50)

        # Create a skillbook manually
        skillbook = Skillbook()
        skillbook.add_skill(
            section="general",
            content="Always provide step-by-step explanations",
            metadata={"helpful": 3, "harmful": 0},
        )
        skillbook.add_skill(
            section="math",
            content="Show your calculations clearly",
            metadata={"helpful": 5, "harmful": 0},
        )

        # Save it
        test_path = "test_skillbook.json"
        skillbook.save_to_file(test_path)
        print(f"\n✓ Saved skillbook to {test_path}")

        # Load it back
        loaded = Skillbook.load_from_file(test_path)
        print(f"✓ Loaded skillbook with {len(loaded.skills())} skills")

        # Verify content matches
        for original, loaded_skill in zip(skillbook.skills(), loaded.skills()):
            assert original.content == loaded_skill.content
            assert original.helpful == loaded_skill.helpful
        print("✓ Content verified - save/load working correctly!")

        # Show the JSON structure
        print(f"\nJSON structure of saved skillbook:")
        with open(test_path, "r") as f:
            print(f.read())

        # Clean up
        # os.remove(test_path)
        print(f"\n✓ Cleaned up {test_path}")
