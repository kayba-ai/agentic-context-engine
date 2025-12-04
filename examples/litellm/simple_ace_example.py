#!/usr/bin/env python3
"""
Simplest possible ACE example with ACELiteLLM.

This shows the minimal code needed to use ACE with a production LLM.
"""

import os
from dotenv import load_dotenv

from ace import ACELiteLLM, Sample, SimpleEnvironment

# Load environment variables
load_dotenv()


def main():
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    # 1. Create ACE agent (bundles all components with v2.1 prompts)
    agent = ACELiteLLM(model="claude-sonnet-4-5-20250929")

    # 2. Create training samples
    samples = [
        Sample(question="What is 2+2?", ground_truth="4"),
        Sample(question="What color is the sky?", ground_truth="blue"),
        Sample(question="Capital of France?", ground_truth="Paris"),
    ]

    # 3. Train the agent
    environment = SimpleEnvironment()
    results = agent.learn(samples, environment, epochs=1)

    # 4. Check results
    print(f"Trained on {len(results)} samples")
    print(f"Skillbook now has {len(agent.skillbook.skills())} strategies")

    # Show a few learned strategies
    for skill in agent.skillbook.skills()[:3]:
        print(f"\nLearned: {skill.content}")

    # 5. Test with new question
    answer = agent.ask("What is 5 + 3?")
    print(f"\nTest question: What is 5 + 3?")
    print(f"Answer: {answer}")


if __name__ == "__main__":
    main()
