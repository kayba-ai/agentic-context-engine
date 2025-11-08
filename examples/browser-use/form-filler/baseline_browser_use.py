#!/usr/bin/env python3
"""
Baseline Browser Agent (WITHOUT ACE)

Simple browser automation agent without any learning.
Compare this with ace_browser_use.py to see ACE's value.
"""

import asyncio
from typing import List, Dict
from dotenv import load_dotenv

from browser_use import Agent, Browser, ChatOpenAI
from pathlib import Path

load_dotenv()

# Import utilities from parent directory
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from debug import print_history_details


async def run_browser_task(
    task: str, model: str = "gpt-4o-mini", headless: bool = True
):
    """Run browser task without any learning."""
    browser = None
    try:
        # Start browser
        browser = Browser(headless=headless)
        await browser.start()

        # Create agent with basic task (no learning, no strategy optimization)
        llm = ChatOpenAI(model=model, temperature=0.0)

        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
        )

        # Run with timeout
        history = await asyncio.wait_for(agent.run(max_steps=10), timeout=240.0)

        # Print all history information in a nice format
        print_history_details(history)

        # Parse result
        output = history.final_result() if hasattr(history, "final_result") else ""
        steps = (
            len(history.action_names())
            if hasattr(history, "action_names") and history.action_names()
            else 0
        )

        # Determine status
        status = "ERROR"
        if "SUCCESS:" in output.upper():
            status = "SUCCESS"

        return {
            "status": status,
            "steps": steps,
            "output": output,
            "success": status == "SUCCESS",
        }

    except asyncio.TimeoutError:
        # Get actual steps even on timeout - history should exist
        try:
            steps = (
                history.number_of_steps()
                if "history" in locals() and hasattr(history, "number_of_steps")
                else 0
            )
        except:
            steps = 25  # max_steps if we can't determine
        return {"status": "ERROR", "steps": steps, "error": "Timeout", "success": False}
    except Exception as e:
        # Get actual steps even on error - history might exist
        try:
            steps = (
                history.number_of_steps()
                if "history" in locals() and hasattr(history, "number_of_steps")
                else 0
            )
        except:
            steps = 0
        return {"status": "ERROR", "steps": steps, "error": str(e), "success": False}
    finally:
        if browser:
            try:
                await browser.stop()
            except:
                pass


def main(task_file: str = "task1_flight_search.txt"):
    """Main function - basic browser automation without learning.

    Args:
        task_file: Path to the task file containing the browser task description.
                  Defaults to "task1_flight_search.txt".
    """

    print("\n🤖 Baseline Browser Agent (WITHOUT ACE)")
    print("🚫 No learning - same approach every time")
    print("=" * 40)

    print("\n🔄 Starting browser task (no learning)...\n")

    results = []

    with open(task_file, "r") as f:
        task = f.read()

    result = asyncio.run(run_browser_task(task=task, headless=False))
    results.append(result)

    # Show final results
    print("=" * 40)
    print("📊 Results:")

    # Summary
    successful = sum(1 for r in results if r["success"])
    total_steps = sum(r["steps"] for r in results)
    avg_steps = total_steps / len(results) if results else 0

    print(
        f"\n✅ Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)"
    )
    print(f"⚡ Average steps: {avg_steps:.1f}")
    print(f"🚫 No learning - same performance every time")

    print(f"\n💡 Compare with: python examples/browser-use/ace_browser_use.py")
    print(f"   ACE learns and improves after each task!")


if __name__ == "__main__":
    main()
