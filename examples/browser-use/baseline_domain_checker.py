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

from browser_use import Agent, Browser, ChatOpenAI

load_dotenv()


def get_test_domains() -> List[str]:
    """Get list of test domains to check."""
    return [
        "testdomain123456.com",
        "myuniquedomain789.net",
        "brandnewstartup2024.io",
        "innovativetech555.org",
        "creativesolutions999.co",
        "digitalagency2024.biz",
        "techstartup123.app",
        "newcompany456.info",
        "uniquebusiness789.online",
        "moderntech2024.dev"
    ]


async def check_domain(domain: str, model: str = "gpt-4o-mini", headless: bool = True):
    """Check domain availability without any learning, with retry logic."""
    max_retries = 3
    last_error = None
    total_steps = 0
    attempt_details = []

    for attempt in range(max_retries):
        browser = None
        steps = 0  # Initialize steps for this attempt
        history = None  # Initialize history for this attempt
        try:
            # Start browser
            browser = Browser(headless=headless)
            await browser.start()

            # Create agent with basic task (no learning, no strategy optimization)
            llm = ChatOpenAI(model=model, temperature=0.0)

            task = f"""
You are a domain availability checking agent. Check if the domain "{domain}" is available.

  IMPORTANT: Do NOT navigate to {domain} directly. Instead:
  1. Go to a domain checking website
  2. In the search bar type "{domain}" on that website
  3. Read the availability status from the results

Output format (exactly one of these):
AVAILABLE: {domain}
TAKEN: {domain}
ERROR: <reason>"""

            agent = Agent(
                task=task,
                llm=llm,
                browser=browser,
                max_actions_per_step=5,
                max_steps=20,
            )

            # Run with timeout
            history = await asyncio.wait_for(agent.run(), timeout=180.0)

            # Parse result
            output = history.final_result() if hasattr(history, "final_result") else ""
            steps = len(history.action_names()) if hasattr(history, "action_names") and history.action_names() else 0

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

            # If we got a valid status, return with success based on correctness
            if status != "ERROR":
                # For testing purposes, we need to determine what's actually correct
                # Since these are test domains, we'll assume they're AVAILABLE unless we can verify otherwise
                # In a real scenario, you'd check against a known ground truth
                expected_status = "AVAILABLE"  # Test domains should be available
                correct = (status == expected_status)

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
                    "attempt_details": attempt_details
                }

            # Store error for potential retry
            total_steps += steps
            attempt_details.append(f"attempt {attempt + 1}: {steps} steps")
            last_error = f"Failed to get valid result: {output}"

        except asyncio.TimeoutError:
            # Get actual steps completed before timeout
            try:
                if history and hasattr(history, "number_of_steps"):
                    steps = history.number_of_steps()
                elif history and hasattr(history, "action_names") and history.action_names():
                    steps = len(history.action_names())
                else:
                    steps = 0  # Unknown - don't make up numbers
            except:
                steps = 0  # Can't determine actual steps

            total_steps += steps
            attempt_details.append(f"attempt {attempt + 1}: {steps} steps (timeout)")
            last_error = f"Timeout on attempt {attempt + 1}"

        except Exception as e:
            # Get actual steps even on error
            try:
                if history and hasattr(history, "number_of_steps"):
                    steps = history.number_of_steps()
                elif history and hasattr(history, "action_names") and history.action_names():
                    steps = len(history.action_names())
                else:
                    steps = 0  # Unknown steps for real errors
            except:
                steps = 0

            total_steps += steps
            attempt_details.append(f"attempt {attempt + 1}: {steps} steps (error)")
            last_error = f"Error on attempt {attempt + 1}: {str(e)}"

        finally:
            if browser:
                try:
                    await browser.stop()
                except:
                    pass

    # All retries failed
    return {
        "domain": domain,
        "status": "ERROR",
        "steps": steps if 'steps' in locals() else 0,
        "total_steps": total_steps,
        "error": f"Failed after {max_retries} attempts. Last error: {last_error}",
        "success": False,
        "correct": False,
        "expected": "AVAILABLE",
        "attempt": max_retries,
        "attempt_details": attempt_details
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
        status = result['status']
        steps = result['steps']
        total_steps = result.get('total_steps', steps)
        success = result['success']
        attempt = result.get('attempt', 1)
        attempt_details = result.get('attempt_details', [])

        # Show detailed step breakdown for multiple attempts
        step_info = f"{total_steps} steps"
        if attempt > 1:
            step_info += f" total ({', '.join(attempt_details)})"
        else:
            step_info += f" (1 attempt)"

        correct = result.get('correct', False)
        accuracy_indicator = '✓' if correct else '✗'
        expected = result.get('expected', 'UNKNOWN')
        print(f"   📊 Result: {status} ({accuracy_indicator}) - {step_info}")
        if not correct and success:
            print(f"       Expected: {expected}, Got: {status}")
        print()

    # Show final results
    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print(f"{'#':<3} {'Domain':<25} {'Status':<10} {'Acc':<4} {'Steps':<8} {'Details'}")
    print("-" * 80)

    for i, result in enumerate(results, 1):
        domain = result['domain']
        status = result['status']
        steps = result['steps']
        total_steps = result.get('total_steps', steps)
        success = result['success']
        attempt = result.get('attempt', 1)
        attempt_details = result.get('attempt_details', [])

        # Show detailed step breakdown for multiple attempts
        if attempt > 1:
            step_details = f"({', '.join(attempt_details)})"
        else:
            step_details = "(1 attempt)"

        correct = result.get('correct', False)
        accuracy_indicator = '✓' if correct else '✗'

        print(f"{i:<3} {domain:<25} {status:<10} {accuracy_indicator:<4} {total_steps:<8} {step_details}")

    # Enhanced Summary
    successful = sum(1 for r in results if r['success'])
    correct = sum(1 for r in results if r.get('correct', False))
    total_steps = sum(r.get('total_steps', r['steps']) for r in results)
    domains_with_retries = sum(1 for r in results if r.get('attempt', 1) > 1)
    total_attempts = sum(r.get('attempt', 1) for r in results)

    avg_steps_per_domain = total_steps / len(results) if results else 0
    avg_steps_per_success = total_steps / successful if successful > 0 else 0

    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    print(f"✅ Success rate:         {successful:>2}/{len(results)} ({100*successful/len(results):>5.1f}%)")
    print(f"🎯 Accuracy rate:        {correct:>2}/{len(results)} ({100*correct/len(results):>5.1f}%)")
    print(f"📊 Total steps:          {total_steps:>6} across all attempts")
    print(f"📈 Avg steps/domain:     {avg_steps_per_domain:>6.1f}")
    print(f"🔄 Domains w/ retries:   {domains_with_retries:>2}/{len(results)}")
    print(f"🔢 Total attempts:       {total_attempts:>6}")
    print("=" * 80)


if __name__ == "__main__":
    main()