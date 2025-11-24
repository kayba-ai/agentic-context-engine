#!/usr/bin/env python3
"""
Baseline Domain Checker (WITHOUT ACE)

Simple domain checker using browser automation without any learning.
Compare this with ace_domain_checker.py to see ACE's value.
"""

import asyncio
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

from browser_use import Agent, Browser, ChatBrowserUse

# Import common utilities from parent directory
import sys


# Utility function for timeout calculation
def calculate_timeout_steps(timeout_seconds: float) -> int:
    """Calculate steps for timeout based on 1 step per 12 seconds."""
    return int(timeout_seconds // 12)


# Import domain-specific utilities from local module
from domain_utils import (
    get_test_domains,
    parse_domain_checker_output,
    DOMAIN_CHECKER_TEMPLATE,
)


async def check_domain(domain: str, headless: bool = True):
    """Check domain availability without any learning, with retry logic."""
    max_retries = 3
    last_error = None
    total_steps = 0
    attempt_details = []

    # Track browser-use tokens across all attempts
    total_browseruse_tokens = 0

    for attempt in range(max_retries):
        browser = None
        agent = None  # Initialize agent for this attempt
        history = None  # Initialize history for this attempt
        steps = 0  # Initialize steps for this attempt
        try:
            # Start browser
            browser = Browser(headless=headless)
            await browser.start()

            # Create agent with basic task (no learning, no strategy optimization)
            llm = ChatBrowserUse()

            # Use common template
            task = DOMAIN_CHECKER_TEMPLATE.format(domain=domain)

            agent = Agent(
                task=task,
                llm=llm,
                browser=browser,
                max_actions_per_step=5,
                max_steps=25,
                calculate_cost=True,  # Enable cost tracking
            )

            # Run with timeout
            history = await asyncio.wait_for(agent.run(), timeout=180.0)

            # Parse result (back to original working logic)
            output = history.final_result() if hasattr(history, "final_result") else ""
            steps = (
                len(history.action_names())
                if hasattr(history, "action_names") and history.action_names()
                else 0
            )

            # Add steps to total and track attempt
            total_steps += steps
            attempt_details.append(f"attempt {attempt + 1}: {steps} steps")

            # Determine status
            status = "ERROR"
            output_upper = output.upper()
            domain_upper = domain.upper()

            if f"AVAILABLE: {domain_upper}" in output_upper:
                status = "AVAILABLE"
            elif f"TAKEN: {domain_upper}" in output_upper:
                status = "TAKEN"

            # If we got a valid status, collect tokens before returning
            if status != "ERROR":
                # For testing purposes, we need to determine what's actually correct
                # Since these are test domains, we'll assume they're AVAILABLE unless we can verify otherwise
                # In a real scenario, you'd check against a known ground truth
                expected_status = "AVAILABLE"  # Test domains should be available
                correct = status == expected_status

                # Collect tokens from this successful attempt
                attempt_tokens = 0

                # Method 1: Try to get tokens from history (works after successful completion)
                if "history" in locals() and history and hasattr(history, "usage"):
                    try:
                        usage = history.usage
                        if usage:
                            # Try different ways to extract total tokens
                            if hasattr(usage, "total_tokens"):
                                attempt_tokens = usage.total_tokens
                            elif isinstance(usage, dict) and "total_tokens" in usage:
                                attempt_tokens = usage["total_tokens"]
                            elif hasattr(usage, "input_tokens") and hasattr(
                                usage, "output_tokens"
                            ):
                                attempt_tokens = (
                                    usage.input_tokens + usage.output_tokens
                                )
                            elif (
                                isinstance(usage, dict)
                                and "input_tokens" in usage
                                and "output_tokens" in usage
                            ):
                                attempt_tokens = (
                                    usage["input_tokens"] + usage["output_tokens"]
                                )
                    except Exception as e:
                        print(f"   ⚠️ Could not get tokens from history: {e}")

                # Method 2: Try agent.token_cost_service (works even during partial execution)
                if attempt_tokens == 0 and "agent" in locals() and agent:
                    try:
                        if hasattr(agent, "token_cost_service"):
                            usage_summary = (
                                await agent.token_cost_service.get_usage_summary()
                            )
                            if usage_summary:
                                if (
                                    isinstance(usage_summary, dict)
                                    and "total_tokens" in usage_summary
                                ):
                                    attempt_tokens = usage_summary["total_tokens"]
                                elif hasattr(usage_summary, "total_tokens"):
                                    attempt_tokens = usage_summary.total_tokens
                    except Exception as e:
                        print(f"   ⚠️ Could not get tokens from agent service: {e}")

                total_browseruse_tokens += attempt_tokens
                print(
                    f"   🤖 Attempt {attempt + 1} tokens: {attempt_tokens} (total: {total_browseruse_tokens})"
                )

                return {
                    "domain": domain,
                    "status": status,
                    "steps": steps,  # Steps from final attempt
                    "total_steps": total_steps,  # Cumulative steps
                    "output": output,
                    "success": True,  # Successfully got a result
                    "correct": correct,  # Whether the result was accurate
                    "expected": expected_status,
                    "attempt": attempt + 1,
                    "attempt_details": attempt_details,
                    "browseruse_tokens": total_browseruse_tokens,
                }

            # Store error for potential retry
            last_error = f"Failed to get valid result: {output}"

        except asyncio.TimeoutError:
            # Calculate additional steps for timeout duration
            timeout_duration = 180.0  # The timeout value used in wait_for()
            timeout_steps = calculate_timeout_steps(timeout_duration)

            # Get actual steps completed before timeout
            try:
                if history and hasattr(history, "number_of_steps"):
                    actual_steps = history.number_of_steps()
                elif (
                    history
                    and hasattr(history, "action_names")
                    and history.action_names()
                ):
                    actual_steps = len(history.action_names())
                else:
                    actual_steps = 0  # Unknown - don't make up numbers
            except:
                actual_steps = 0  # Can't determine actual steps

            # Add timeout steps to actual steps
            steps = actual_steps + timeout_steps
            total_steps += steps
            attempt_details.append(
                f"attempt {attempt + 1}: {steps} steps (timeout, +{timeout_steps} for duration)"
            )
            last_error = f"Timeout on attempt {attempt + 1}"

        except Exception as e:

            # Get actual steps even on error
            try:
                if history and hasattr(history, "number_of_steps"):
                    steps = history.number_of_steps()
                elif (
                    history
                    and hasattr(history, "action_names")
                    and history.action_names()
                ):
                    steps = len(history.action_names())
                else:
                    steps = 0
            except:
                steps = 0

            total_steps += steps
            attempt_details.append(f"attempt {attempt + 1}: {steps} steps (error)")
            last_error = f"Error on attempt {attempt + 1}: {str(e)}"
            print(f"   💥 Error ({steps} steps): {str(e)}")

        finally:
            # Capture tokens from this attempt using browser-use's cost tracking
            attempt_tokens = 0

            # Method 1: Try to get tokens from history (works after successful completion)
            if "history" in locals() and history and hasattr(history, "usage"):
                try:
                    usage = history.usage
                    if usage:
                        # Try different ways to extract total tokens
                        if hasattr(usage, "total_tokens"):
                            attempt_tokens = usage.total_tokens
                        elif isinstance(usage, dict) and "total_tokens" in usage:
                            attempt_tokens = usage["total_tokens"]
                        elif hasattr(usage, "input_tokens") and hasattr(
                            usage, "output_tokens"
                        ):
                            attempt_tokens = usage.input_tokens + usage.output_tokens
                        elif (
                            isinstance(usage, dict)
                            and "input_tokens" in usage
                            and "output_tokens" in usage
                        ):
                            attempt_tokens = (
                                usage["input_tokens"] + usage["output_tokens"]
                            )
                except Exception as e:
                    print(f"   ⚠️ Could not get tokens from history: {e}")

            # Method 2: Try agent.token_cost_service (works even during partial execution)
            if attempt_tokens == 0 and "agent" in locals() and agent:
                try:
                    if hasattr(agent, "token_cost_service"):
                        usage_summary = (
                            await agent.token_cost_service.get_usage_summary()
                        )
                        if usage_summary:
                            if (
                                isinstance(usage_summary, dict)
                                and "total_tokens" in usage_summary
                            ):
                                attempt_tokens = usage_summary["total_tokens"]
                            elif hasattr(usage_summary, "total_tokens"):
                                attempt_tokens = usage_summary.total_tokens
                except Exception as e:
                    print(f"   ⚠️ Could not get tokens from agent service: {e}")

            total_browseruse_tokens += attempt_tokens
            print(
                f"   🤖 Attempt {attempt + 1} tokens: {attempt_tokens} (total: {total_browseruse_tokens})"
            )

            if browser:
                try:
                    await browser.stop()
                except:
                    pass

    # All retries failed - use accumulated tokens from all attempts
    return {
        "domain": domain,
        "status": "ERROR",
        "steps": steps if "steps" in locals() else 0,
        "total_steps": total_steps,
        "error": f"Failed after {max_retries} attempts. Last error: {last_error}",
        "success": False,
        "correct": False,
        "expected": "AVAILABLE",
        "attempt": max_retries,
        "attempt_details": attempt_details,
        "browseruse_tokens": total_browseruse_tokens,
    }


def main():
    """Main function - basic domain checking without learning."""

    print("\n🤖 Baseline Domain Checker (WITHOUT ACE)")
    print("🚫 No learning - same approach every time")
    print("=" * 50)

    # Get test domains
    domains = get_test_domains()
    print(f"📋 Testing {len(domains)} domains:")
    for i, domain in enumerate(domains, 1):
        print(f"  {i}. {domain}")

    print("\n🔄 Starting domain checks (no learning)...\n")

    results = []

    # Check each domain without any learning
    for i, domain in enumerate(domains, 1):
        print(f"🔍 [{i}/{len(domains)}] Checking domain: {domain}")

        # Run check
        result = asyncio.run(check_domain(domain, headless=False))
        results.append(result)

        # Show what happened
        status = result["status"]
        steps = result["steps"]
        total_steps = result.get("total_steps", steps)
        success = result["success"]
        attempt = result.get("attempt", 1)
        attempt_details = result.get("attempt_details", [])

        # Show detailed step breakdown for multiple attempts
        step_info = f"{total_steps} steps"
        if attempt > 1:
            step_info += f" total ({', '.join(attempt_details)})"
        else:
            step_info += f" (1 attempt)"

        correct = result.get("correct", False)
        accuracy_indicator = "✓" if correct else "✗"
        expected = result.get("expected", "UNKNOWN")
        print(f"   📊 Result: {status} ({accuracy_indicator}) - {step_info}")
        if not correct and success:
            print(f"       Expected: {expected}, Got: {status}")
        print()

    # Show final results
    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print(
        f"{'#':<3} {'Domain':<25} {'Status':<10} {'Acc':<4} {'Steps':<8} {'Browser-Tokens':<13} {'Details'}"
    )
    print("-" * 93)

    for i, result in enumerate(results, 1):
        domain = result["domain"]
        status = result["status"]
        steps = result["steps"]
        total_steps = result.get("total_steps", steps)
        success = result["success"]
        attempt = result.get("attempt", 1)
        attempt_details = result.get("attempt_details", [])

        # Show detailed step breakdown for multiple attempts
        if attempt > 1:
            step_details = f"({', '.join(attempt_details)})"
        else:
            step_details = "(1 attempt)"

        correct = result.get("correct", False)
        accuracy_indicator = "✓" if correct else "✗"
        browseruse_tokens = result.get("browseruse_tokens", 0)

        print(
            f"{i:<3} {domain:<25} {status:<10} {accuracy_indicator:<4} {total_steps:<8} {browseruse_tokens:<12} {step_details}"
        )

    # Enhanced Summary
    successful = sum(1 for r in results if r["success"])
    correct = sum(1 for r in results if r.get("correct", False))
    total_steps = sum(r.get("total_steps", r["steps"]) for r in results)
    domains_with_retries = sum(1 for r in results if r.get("attempt", 1) > 1)
    total_attempts = sum(r.get("attempt", 1) for r in results)

    avg_steps_per_domain = total_steps / len(results) if results else 0
    avg_steps_per_success = total_steps / successful if successful > 0 else 0

    # Calculate actual browser-use token usage
    total_browseruse_tokens = sum(r.get("browseruse_tokens", 0) for r in results)
    avg_browseruse_tokens_per_domain = (
        total_browseruse_tokens / len(results) if results else 0.0
    )

    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    print(
        f"✅ Success rate:          {successful:>2}/{len(results)} ({100*successful/len(results):>5.1f}%)"
    )
    print(
        f"🎯 Accuracy rate:         {correct:>2}/{len(results)} ({100*correct/len(results):>5.1f}%)"
    )
    print(f"🔄 Domains w/ retries:    {domains_with_retries:>2}/{len(results)}")
    print(f"🔢 Total attempts:        {total_attempts:>6}")
    print()
    print(
        f"{'📊 Steps:':<25} {total_steps:>6} total     {avg_steps_per_domain:>6.1f} per domain"
    )
    print(
        f"{'🤖 Browser-Use Tokens:':<25} {total_browseruse_tokens:>6} total     {avg_browseruse_tokens_per_domain:>6.1f} per domain"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
