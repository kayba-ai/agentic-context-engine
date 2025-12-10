#!/usr/bin/env python3
"""
Test ACE with LM Studio (Ollama-compatible model).

LM Studio runs an OpenAI-compatible API server, so we use the
openai/ prefix with a custom base_url instead of ollama/.
"""

from ace.integrations import ACELiteLLM
from ace import Sample, SimpleEnvironment
from pathlib import Path


def main():
    print("🤖 Testing ACELiteLLM with LM Studio...")

    # LM Studio default endpoint
    lm_studio_url = "http://localhost:1234/v1"

    # 1. Create ACELiteLLM agent pointing to LM Studio
    # Note: Use "openai/" prefix with base_url for LM Studio
    print(f"\n📡 Connecting to LM Studio at {lm_studio_url}...")

    skillbook_path = Path("lm_studio_learned_strategies.json")

    try:
        agent = ACELiteLLM(
            model="openai/local-model",  # LM Studio serves any model as 'local-model'
            base_url=lm_studio_url,
            max_tokens=512,
            temperature=0.2,
            is_learning=True,
            skillbook_path=str(skillbook_path) if skillbook_path.exists() else None,
        )
    except Exception as e:
        print(f"❌ Failed to connect to LM Studio: {e}")
        print("\n💡 Make sure:")
        print("1. LM Studio is running")
        print("2. A model is loaded")
        print("3. Server is enabled (default: http://localhost:1234)")
        return

    # 2. Test basic question
    print("\n❓ Testing agent before learning:")
    test_question = "What is 2+2?"
    try:
        answer = agent.ask(test_question)
        print(f"Q: {test_question}")
        print(f"A: {answer}")
    except Exception as e:
        print(f"❌ Failed to get answer: {e}")
        return

    # 3. Create training samples
    samples = [
        Sample(question="What is 2+2?", ground_truth="4"),
        Sample(question="What color is the sky?", ground_truth="blue"),
        Sample(question="Capital of France?", ground_truth="Paris"),
    ]

    # 4. Run learning
    print("\n🚀 Running ACE learning...")
    print("⚠️  Note: This may take a while with smaller local models")
    environment = SimpleEnvironment()

    try:
        results = agent.learn(samples, environment, epochs=1)
        successful_samples = len(
            [r for r in results if "Correct" in r.environment_result.feedback]
        )
        print(f"✅ Successfully processed {successful_samples}/{len(results)} samples")
    except Exception as e:
        print(f"❌ Learning failed: {e}")
        print("💡 Small models may struggle with structured JSON output")
        print("   Consider using Instructor wrapper for better reliability")
        results = []

    # 5. Check results
    print(f"\n📊 Trained on {len(results)} samples")
    print(f"📚 Skillbook now has {len(agent.skillbook.skills())} strategies")

    # 6. Test with learned knowledge
    print("\n🧠 Testing agent after learning:")
    for question in ["What is 3+3?", "What color is grass?"]:
        try:
            answer = agent.ask(question)
            print(f"Q: {question}")
            print(f"A: {answer}")
        except Exception as e:
            print(f"❌ Error: {e}")

    # Show learned strategies
    if agent.skillbook.skills():
        print("\n💡 Learned strategies:")
        for skill in agent.skillbook.skills()[:3]:
            helpful = skill.helpful
            harmful = skill.harmful
            score = f"(+{helpful}/-{harmful})"
            content = (
                skill.content[:70] + "..." if len(skill.content) > 70 else skill.content
            )
            print(f"  • {content} {score}")

    # 7. Save learned knowledge
    agent.save_skillbook(skillbook_path)
    print(f"\n💾 Saved learned strategies to {skillbook_path}")


if __name__ == "__main__":
    main()
