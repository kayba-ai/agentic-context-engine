#!/usr/bin/env python3
"""
ACELiteLLM Starter Template - Learning from Execution Feedback

This demonstrates how ACELiteLLM agents learn from task execution feedback
to improve their performance over time. The agent will:

1. Perform tasks with current knowledge
2. Learn from execution results and feedback
3. Build strategies that improve future performance
4. Persist learned knowledge across sessions

Key Concepts:
- Generator: Produces answers using learned strategies
- Reflector: Analyzes what worked/failed in execution
- Curator: Updates the strategy playbook based on analysis
- Playbook: Persistent store of learned execution strategies

Requirements:
    pip install ace-framework

Environment:
    export OPENAI_API_KEY="your-api-key"

    # Other supported providers (change model parameter):
    # export ANTHROPIC_API_KEY="your-key"  # model="claude-3-haiku-20240307"
    # export GOOGLE_API_KEY="your-key"     # model="gemini/gemini-1.5-flash"
    # export COHERE_API_KEY="your-key"     # model="command-r"
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from ace.integrations import ACELiteLLM
from ace import Sample, SimpleEnvironment

load_dotenv()


def main():
    print("=" * 60)
    print("ACELiteLLM Learning Example")
    print("=" * 60)

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: Please set OPENAI_API_KEY environment variable")
        print("\nSetup instructions:")
        print("1. Get an API key from https://platform.openai.com/api-keys")
        print("2. export OPENAI_API_KEY='your-api-key'")
        print("3. Run this script again")
        return

    print("Setting up ACE learning agent...")

    # Setup playbook persistence
    playbook_path = Path(__file__).parent / "litellm_ace_learned.json"

    # Create ACELiteLLM agent
    # Note: Change model for different providers (see docstring for examples)
    agent = ACELiteLLM(
        model="gpt-4o-mini",  # Fast, cost-effective OpenAI model
        max_tokens=512,  # Reasonable limit for simple Q&A
        temperature=0.2,  # Low temperature for consistent learning
        playbook_path=str(playbook_path) if playbook_path.exists() else None,
    )

    # Show current knowledge state
    if playbook_path.exists():
        print(f"Loaded playbook: {len(agent.playbook.bullets())} existing strategies")
    else:
        print("Starting fresh: No existing playbook found")

    print("\n" + "-" * 40)
    print("BEFORE LEARNING")
    print("-" * 40)

    # Test current performance
    test_question = "What is 2+2?"
    answer = agent.ask(test_question)
    print(f"Q: {test_question}")
    print(f"A: {answer}")

    print("\n" + "-" * 40)
    print("LEARNING PHASE")
    print("-" * 40)

    # Create training samples with ground truth
    # These represent good execution examples the agent should learn from
    samples = [
        Sample(question="What is 2+2?", ground_truth="4"),
        Sample(question="What is 3+3?", ground_truth="6"),
    ]

    print(f"Learning from {len(samples)} execution examples...")

    # Run ACE learning process
    # Generator -> Environment -> Reflector -> Curator -> Updated Playbook
    try:
        results = agent.learn(samples, SimpleEnvironment(), epochs=1)
        successful_samples = len([r for r in results if r.success])
        success_rate = successful_samples / len(results) if results else 0

        print(f"Learning completed:")
        print(
            f"  Success rate: {success_rate:.1%} ({successful_samples}/{len(results)})"
        )
        print(f"  Total strategies: {len(agent.playbook.bullets())}")
    except Exception as e:
        print(f"Learning failed: {e}")
        print("The agent will continue with existing knowledge.")

    print("\n" + "-" * 40)
    print("AFTER LEARNING")
    print("-" * 40)

    # Test improved performance
    test_question = "What is 4+4?"
    answer = agent.ask(test_question)
    print(f"Q: {test_question}")
    print(f"A: {answer}")

    # Show learned strategies (if any)
    if agent.playbook.bullets():
        print(f"\nLearned Strategies ({len(agent.playbook.bullets())} total):")
        for i, bullet in enumerate(agent.playbook.bullets()[:3], 1):  # Show first 3
            helpful, harmful = bullet.helpful, bullet.harmful
            effectiveness = "Effective" if helpful > harmful else "Needs improvement"
            print(f"  {i}. [{effectiveness}] {bullet.content[:50]}...")

    print("\n" + "-" * 40)
    print("PERSISTENCE")
    print("-" * 40)

    # Save learned knowledge
    agent.save_playbook(str(playbook_path))
    print(f"Strategies saved to: {playbook_path}")
    print("\nNext Steps:")
    print("  1. Run this script again to see incremental learning")
    print("  2. Check the saved playbook JSON file to see strategies")
    print("  3. Modify the model parameter to try different LLM providers")
    print("  4. Add your own training samples to teach specific tasks")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
