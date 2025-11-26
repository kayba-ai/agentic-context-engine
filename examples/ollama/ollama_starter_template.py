#!/usr/bin/env python3
"""
Quick start example using Ollama with ACELiteLLM integration.

This shows how to use ACELiteLLM with local Ollama models for
self-improving AI agents. The agent learns from examples and
improves its responses over time.

Requires:
- Ollama installed and running locally
- A model pulled (e.g., ollama pull llama2)

Features demonstrated:
- Creating ACELiteLLM agent with Ollama
- Learning from training samples
- Testing before/after learning
- Saving learned strategies for reuse
"""

import subprocess
from ace.integrations import ACELiteLLM
from ace import Sample, SimpleEnvironment
import os
from pathlib import Path


def check_ollama_running():
    """Check if Ollama is running and has models available."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, result.stdout
        return False, "No models found. Run 'ollama pull llama2' to get started."
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "Ollama not found. Install from https://ollama.ai"
    except Exception as e:
        return False, f"Error checking Ollama: {e}"


def main():
    # Check if Ollama is available
    is_running, message = check_ollama_running()
    if not is_running:
        print(f"❌ {message}")
        print("\nTo get started:")
        print("1. Install Ollama: https://ollama.ai")
        print("2. Pull a model: ollama pull llama2")
        print("3. Verify: ollama list")
        return

    print("✅ Ollama is running")
    print("Available models:")
    print(message)

    # 1. Create ACELiteLLM agent with Ollama
    print("\n🤖 Creating ACELiteLLM agent with Ollama...")
    playbook_path = Path(__file__).parent / "ollama_learned_strategies.json"
    agent = ACELiteLLM(
        model="ollama/gemma3:1b",
        max_tokens=1024,
        temperature=0.2,
        is_learning=True,
        playbook_path=str(playbook_path) if playbook_path.exists() else None,
    )

    # 2. Try asking questions before learning
    print("\n❓ Testing agent before learning:")
    test_question = "What is 2+2?"
    answer = agent.ask(test_question)
    print(f"Q: {test_question}")
    print(f"A: {answer}")

    # 3. Create training samples
    samples = [
        Sample(question="What is 2+2?", ground_truth="4"),
        Sample(question="What color is the sky?", ground_truth="blue"),
        Sample(question="Capital of France?", ground_truth="Paris"),
    ]

    # 4. Run learning
    print("\n🚀 Running ACE learning with Ollama...")
    print(
        "⚠️  Note: Small models like Gemma 1B may have issues with structured JSON output"
    )
    environment = SimpleEnvironment()

    try:
        results = agent.learn(samples, environment, epochs=1)
        successful_samples = len([r for r in results if r.success])
        print(f"✅ Successfully processed {successful_samples}/{len(results)} samples")
    except Exception as e:
        print(f"❌ Learning failed: {e}")
        print(
            "💡 Try using a larger model like 'ollama pull llama2:7b' for better JSON generation"
        )
        results = []

    # 5. Check results
    print(f"\n📊 Trained on {len(results)} samples")
    print(f"📚 Playbook now has {len(agent.playbook.bullets())} strategies")

    # 6. Test with learned knowledge
    print("\n🧠 Testing agent after learning:")
    for question in ["What is 3+3?", "What color is grass?"]:
        answer = agent.ask(question)
        print(f"Q: {question}")
        print(f"A: {answer}")

    # Show a few learned strategies
    if agent.playbook.bullets():
        print("\n💡 Learned strategies:")
        for bullet in agent.playbook.bullets()[:3]:
            helpful = bullet.helpful
            harmful = bullet.harmful
            score = f"(+{helpful}/-{harmful})"
            print(f"  • {bullet.content[:70]}... {score}")

    # 7. Save learned knowledge for future use
    agent.save_playbook(playbook_path)
    print(f"\n💾 Saved learned strategies to {playbook_path}")


if __name__ == "__main__":
    main()
