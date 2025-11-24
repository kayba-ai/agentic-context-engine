#!/usr/bin/env python3
"""
ACE + Browser-Use Template (Bare Bones)

Minimal template for self-improving browser automation.
Copy this file and customize for your use case.
"""

import asyncio
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from ace import ACEAgent
from browser_use import ChatBrowserUse


async def main():
    """Minimal ACE browser automation example."""

    # Create ACE agent
    agent = ACEAgent(llm=ChatBrowserUse(), ace_model="gpt-4o-mini")

    # Define your tasks
    tasks = [
        "Go to google.com and search for 'python'",
        "Visit wikipedia.org and find the Python page",
    ]

    # Run tasks - ACE learns automatically
    for task in tasks:
        print(f"Running: {task}")

        try:
            history = await agent.run(task=task)
            print(f"✅ Completed")
        except Exception as e:
            print(f"❌ Failed: {e}")

    # Save learned strategies
    agent.save_playbook("playbook.json")
    print(f"💾 Saved {len(agent.playbook.bullets())} learned strategies")


if __name__ == "__main__":
    asyncio.run(main())
