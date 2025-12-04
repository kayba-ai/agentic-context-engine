#!/usr/bin/env python3
"""
Simplest possible ACE + browser-use example.

Shows how ACEAgent is a drop-in replacement for browser-use Agent
with automatic learning.

Requirements:
    pip install ace-framework[browser-use]
    export OPENAI_API_KEY="your-key"  # or other LLM API key
"""

import asyncio
from ace import ACEAgent
from browser_use import ChatBrowserUse


async def main():
    print("🤖 ACE + Browser-Use Simple Example")
    print("=" * 50)
    print("This demo shows ACE learning from browser automation tasks.\n")

    # Create ACE agent (just like browser-use Agent!)
    # Two LLMs are used:
    #   - llm: ChatBrowserUse for browser execution (fine-tuned for browser automation)
    #   - ace_model: "gpt-4o-mini" by default for ACE learning (Reflector/SkillManager)
    agent = ACEAgent(
        llm=ChatBrowserUse(),  # Browser execution LLM
        ace_model="gpt-4o-mini",  # ACE learning LLM (optional, this is the default)
    )

    print("📍 Running 3 tasks - watch ACE learn!\n")

    # Task 1 - Fresh start, no prior knowledge
    print("Task 1: Find top HN post")
    print("-" * 50)
    try:
        await agent.run(task="Find the number 1 post on Hacker News")
        print(f"✅ Completed! Learned {len(agent.skillbook.skills())} strategies\n")
    except Exception as e:
        print(f"⚠️  Task failed: {e}\n")

    # Task 2 - Uses Task 1 learnings
    print("Task 2: Find second HN post")
    print("-" * 50)
    try:
        await agent.run(task="Find the number 2 post on Hacker News")
        print(f"✅ Completed! Learned {len(agent.skillbook.skills())} strategies\n")
    except Exception as e:
        print(f"⚠️  Task failed: {e}\n")

    # Task 3 - Even smarter with accumulated knowledge
    print("Task 3: Find third HN post")
    print("-" * 50)
    try:
        await agent.run(task="Find the number 3 post on Hacker News")
        print(f"✅ Completed! Learned {len(agent.skillbook.skills())} strategies\n")
    except Exception as e:
        print(f"⚠️  Task failed: {e}\n")

    # Show learned strategies
    print("=" * 50)
    print("🎯 Learned Strategies:")
    print("=" * 50)
    if agent.skillbook.skills():
        for i, skill in enumerate(agent.skillbook.skills()[:5], 1):
            helpful = skill.helpful
            harmful = skill.harmful
            score = f"+{helpful}/-{harmful}"
            print(f"{i}. {skill.content}")
            print(f"   Score: {score}\n")
    else:
        print("No strategies learned yet.\n")

    # Save skillbook for reuse
    skillbook_path = "hn_expert.json"
    agent.save_skillbook(skillbook_path)
    print(f"💾 Skillbook saved to {skillbook_path}")
    print("\n✨ Next time, load this skillbook to start with learned strategies!")


async def demo_with_pretrained():
    """
    Demo: Using a pre-trained skillbook.

    Run this after the main() function has saved hn_expert.json
    """
    print("\n" + "=" * 50)
    print("📚 Demo: Using Pre-Trained Skillbook")
    print("=" * 50)

    # Create agent with pre-trained skillbook
    agent = ACEAgent(llm=ChatBrowserUse(), skillbook_path="hn_expert.json")

    print(f"Loaded {len(agent.skillbook.skills())} strategies from skillbook\n")

    # This task should be faster/better due to pre-trained knowledge
    print("Task: Find fourth HN post (using pre-trained strategies)")
    print("-" * 50)
    try:
        await agent.run(task="Find the number 4 post on Hacker News")
        print("✅ Completed with pre-trained knowledge!\n")
    except Exception as e:
        print(f"⚠️  Task failed: {e}\n")


async def demo_learning_toggle():
    """
    Demo: Toggling learning on/off.
    """
    print("\n" + "=" * 50)
    print("🔄 Demo: Toggle Learning On/Off")
    print("=" * 50)

    agent = ACEAgent(llm=ChatBrowserUse())

    # Task 1 with learning
    print("Task 1 (learning=ON): Find top post")
    await agent.run(task="Find the top post on Hacker News")
    print(f"Learned {len(agent.skillbook.skills())} strategies\n")

    # Disable learning
    agent.disable_learning()
    print("🔕 Learning disabled")

    # Task 2 without learning
    print("Task 2 (learning=OFF): Find second post")
    await agent.run(task="Find the second post on Hacker News")
    print(f"Strategies unchanged: {len(agent.skillbook.skills())}\n")

    # Re-enable learning
    agent.enable_learning()
    print("🔔 Learning re-enabled")

    # Task 3 with learning again
    print("Task 3 (learning=ON): Find third post")
    await agent.run(task="Find the third post on Hacker News")
    print(f"Learned {len(agent.skillbook.skills())} strategies\n")


if __name__ == "__main__":
    # Run main example
    asyncio.run(main())

    # Uncomment to run additional demos:
    # asyncio.run(demo_with_pretrained())
    # asyncio.run(demo_learning_toggle())
