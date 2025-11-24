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


# Debug utility function (inlined from debug.py)
def print_history_details(history):
    """Print all history information in a nice formatted way."""
    print("\n" + "=" * 80)
    print("📊 BROWSER HISTORY DETAILS")
    print("=" * 80)

    # Access useful information
    print("\n🔗 VISITED URLS:")
    print("-" * 80)
    try:
        urls = history.urls() if hasattr(history, "urls") else []
        if urls:
            for i, url in enumerate(urls, 1):
                print(f"  {i}. {url}")
        else:
            print("  (no URLs)")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n📸 SCREENSHOT PATHS:")
    print("-" * 80)
    try:
        screenshot_paths = (
            history.screenshot_paths() if hasattr(history, "screenshot_paths") else []
        )
        if screenshot_paths:
            for i, path in enumerate(screenshot_paths, 1):
                print(f"  {i}. {path}")
        else:
            print("  (no screenshots)")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n🖼️  SCREENSHOTS (Base64):")
    print("-" * 80)
    try:
        screenshots = history.screenshots() if hasattr(history, "screenshots") else []
        if screenshots:
            for i, screenshot in enumerate(screenshots, 1):
                preview = (
                    screenshot[:50] + "..." if len(screenshot) > 50 else screenshot
                )
                print(f"  {i}. {preview} ({len(screenshot)} chars)")
        else:
            print("  (no screenshots)")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n⚙️  ACTION NAMES:")
    print("-" * 80)
    try:
        action_names = (
            history.action_names() if hasattr(history, "action_names") else []
        )
        if action_names:
            for i, name in enumerate(action_names, 1):
                print(f"  {i}. {name}")
        else:
            print("  (no actions)")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n📝 EXTRACTED CONTENT:")
    print("-" * 80)
    try:
        extracted = (
            history.extracted_content() if hasattr(history, "extracted_content") else []
        )
        if extracted:
            for i, content in enumerate(extracted, 1):
                preview = (
                    str(content)[:100] + "..."
                    if len(str(content)) > 100
                    else str(content)
                )
                print(f"  {i}. {preview}")
        else:
            print("  (no extracted content)")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n❌ ERRORS:")
    print("-" * 80)
    try:
        errors = history.errors() if hasattr(history, "errors") else []
        if errors:
            for i, error in enumerate(errors, 1):
                if error is not None:
                    print(f"  {i}. {error}")
                else:
                    print(f"  {i}. (no error)")
        else:
            print("  (no errors)")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n🎯 MODEL ACTIONS:")
    print("-" * 80)
    try:
        model_actions = (
            history.model_actions() if hasattr(history, "model_actions") else []
        )
        if model_actions:
            for i, action in enumerate(model_actions, 1):
                action_str = (
                    str(action)[:150] + "..." if len(str(action)) > 150 else str(action)
                )
                print(f"  {i}. {action_str}")
        else:
            print("  (no model actions)")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n🤖 MODEL OUTPUTS:")
    print("-" * 80)
    try:
        model_outputs = (
            history.model_outputs() if hasattr(history, "model_outputs") else []
        )
        if model_outputs:
            for i, output in enumerate(model_outputs, 1):
                output_str = (
                    str(output)[:150] + "..." if len(str(output)) > 150 else str(output)
                )
                print(f"  {i}. {output_str}")
        else:
            print("  (no model outputs)")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n🔄 LAST ACTION:")
    print("-" * 80)
    try:
        last_action = history.last_action() if hasattr(history, "last_action") else None
        if last_action:
            print(f"  {last_action}")
        else:
            print("  (no last action)")
    except Exception as e:
        print(f"  Error: {e}")

    # Analysis methods
    print("\n📊 ANALYSIS RESULTS:")
    print("-" * 80)

    try:
        final_result = (
            history.final_result() if hasattr(history, "final_result") else None
        )
        print(f"  Final Result: {final_result}")
    except Exception as e:
        print(f"  Final Result Error: {e}")

    try:
        is_done = history.is_done() if hasattr(history, "is_done") else None
        print(f"  Is Done: {is_done}")
    except Exception as e:
        print(f"  Is Done Error: {e}")

    try:
        is_successful = (
            history.is_successful() if hasattr(history, "is_successful") else None
        )
        print(f"  Is Successful: {is_successful}")
    except Exception as e:
        print(f"  Is Successful Error: {e}")

    try:
        has_errors = history.has_errors() if hasattr(history, "has_errors") else None
        print(f"  Has Errors: {has_errors}")
    except Exception as e:
        print(f"  Has Errors Error: {e}")

    try:
        model_thoughts = (
            history.model_thoughts() if hasattr(history, "model_thoughts") else []
        )
        print(f"  Model Thoughts Count: {len(model_thoughts)}")
        if model_thoughts:
            for i, thought in enumerate(model_thoughts[:3], 1):  # Show first 3
                thought_str = (
                    str(thought)[:100] + "..."
                    if len(str(thought)) > 100
                    else str(thought)
                )
                print(f"    {i}. {thought_str}")
    except Exception as e:
        print(f"  Model Thoughts Error: {e}")

    try:
        action_results = (
            history.action_results() if hasattr(history, "action_results") else []
        )
        print(f"  Action Results Count: {len(action_results)}")
    except Exception as e:
        print(f"  Action Results Error: {e}")

    try:
        action_history = (
            history.action_history() if hasattr(history, "action_history") else []
        )
        print(f"  Action History Count: {len(action_history)}")
        if action_history:
            for i, action in enumerate(action_history[:3], 1):  # Show first 3
                action_str = (
                    str(action)[:100] + "..." if len(str(action)) > 100 else str(action)
                )
                print(f"    {i}. {action_str}")
    except Exception as e:
        print(f"  Action History Error: {e}")

    try:
        num_steps = (
            history.number_of_steps() if hasattr(history, "number_of_steps") else None
        )
        print(f"  Number of Steps: {num_steps}")
    except Exception as e:
        print(f"  Number of Steps Error: {e}")

    try:
        duration = (
            history.total_duration_seconds()
            if hasattr(history, "total_duration_seconds")
            else None
        )
        print(
            f"  Total Duration: {duration} seconds"
            if duration is not None
            else "  Total Duration: N/A"
        )
    except Exception as e:
        print(f"  Total Duration Error: {e}")

    try:
        structured_output = (
            history.structured_output if hasattr(history, "structured_output") else None
        )
        print("\n📋 STRUCTURED OUTPUT:")
        print("-" * 80)
        if structured_output is not None:
            output_str = (
                str(structured_output)[:200] + "..."
                if len(str(structured_output)) > 200
                else str(structured_output)
            )
            print(f"  {output_str}")
        else:
            print("  (no structured output)")
    except Exception as e:
        print(f"  Structured Output Error: {e}")

    print("\n" + "=" * 80 + "\n")


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
