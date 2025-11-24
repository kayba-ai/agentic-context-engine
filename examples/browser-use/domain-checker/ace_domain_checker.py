#!/usr/bin/env python3
"""
ACE + Browser-Use Domain Checker Demo

Simple demo showing ACE learning to improve at checking domain availability.
Uses the new ACEAgent integration for clean, automatic learning.
"""

import asyncio
import datetime
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Import Opik for token tracking
try:
    import opik
except ImportError:
    opik = None

# Add parent directories to path for imports
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import ACE framework with new integration
from ace import ACEAgent
from ace.observability import configure_opik
from browser_use import ChatBrowserUse


# Utility function for timeout calculation
def calculate_timeout_steps(timeout_seconds: float) -> int:
    """Calculate steps for timeout based on 1 step per 12 seconds."""
    return int(timeout_seconds // 12)


# Import domain-specific utilities
from domain_utils import get_test_domains

# Domain checking task definition
DOMAIN_CHECK_TASK_TEMPLATE = """
You are a browser agent. For every step, first think, then act.
Use exactly this format:
Thought: describe what you want to do next
Action: <browser-use-tool with JSON args>
I will reply with Observation: … after each action.
Repeat Thought → Action → Observation until you can answer.
When you are done, write Final: with the result.

Task: Check if the domain "{domain}" is available.

IMPORTANT: Do NOT navigate to {domain} directly. Instead:
1. Go to a domain checking website (like whois.net, namecheap.com, or godaddy.com)
2. In the search bar type "{domain}" on that website
3. Read the availability status from the results

Output format (exactly one of these):
AVAILABLE: {domain}
TAKEN: {domain}
ERROR: <reason>

Remember: Focus on accuracy and efficiency. Use learned strategies to improve your approach.
"""


def get_ace_token_usage(
    run_start_time: datetime.datetime = None,
) -> tuple[int, int, int, int]:
    """Query Opik for ACE token usage only.

    Returns:
        tuple: (ace_tokens, generator_tokens, reflector_tokens, curator_tokens)
    """
    try:
        if not opik:
            print("   ⚠️ Opik not available for token tracking")
            return 0, 0, 0, 0

        # Create client and flush to ensure data is sent
        client = opik.Opik()
        client.flush()

        print(f"   📋 Querying Opik for ACE token usage...")

        # Use run start time if available, otherwise fall back to last 10 minutes
        if run_start_time:
            recent_time = run_start_time.isoformat().replace("+00:00", "Z")
            print(f"   🕐 Searching for traces since run start: {recent_time}")
        else:
            now = datetime.datetime.now(datetime.timezone.utc)
            recent_time = (
                (now - datetime.timedelta(minutes=10))
                .isoformat()
                .replace("+00:00", "Z")
            )
            print(
                f"   🕐 Searching for traces since: {recent_time} (fallback: last 10 minutes)"
            )

        all_traces = []

        # Only search ACE project for role breakdown
        for project in ["ace-roles"]:
            try:
                traces = client.search_traces(
                    project_name=project,
                    filter_string=f'start_time >= "{recent_time}"',
                    max_results=50,
                )
                print(f"   📊 Found {len(traces)} recent traces in '{project}' project")
                all_traces.extend(traces)
            except Exception as e:
                print(f"   ⚠️ Failed to search '{project}' project: {e}")

        # Track individual ACE role tokens
        generator_tokens = 0
        reflector_tokens = 0
        curator_tokens = 0

        print(f"   🔍 Processing {len(all_traces)} total traces...")

        # First pass: identify and process ACE role traces
        for trace in all_traces:
            trace_name = getattr(trace, "name", "unknown")
            trace_name_lower = trace_name.lower()

            if any(
                role in trace_name_lower
                for role in ["generator", "reflector", "curator"]
            ):
                print(f"      📋 ACE Trace: '{trace_name}'")

                # Get usage from trace or spans
                total_tokens = 0

                if trace.usage:
                    total_tokens = trace.usage.get("total_tokens", 0)
                    print(f"         💰 Tokens: {total_tokens}")
                else:
                    # Check spans for this trace
                    try:
                        spans = client.search_spans(trace_id=trace.id)
                        for span in spans:
                            if hasattr(span, "usage") and span.usage:
                                span_tokens = span.usage.get("total_tokens", 0)
                                total_tokens += span_tokens

                        if total_tokens > 0:
                            print(f"         💰 Tokens (from spans): {total_tokens}")
                    except Exception as e:
                        print(f"         ⚠️ Failed to get spans: {e}")

                # Classify by role
                if "generator" in trace_name_lower:
                    generator_tokens += total_tokens
                    print(f"         🎯 Added to Generator")
                elif "reflector" in trace_name_lower:
                    reflector_tokens += total_tokens
                    print(f"         🔍 Added to Reflector")
                elif "curator" in trace_name_lower:
                    curator_tokens += total_tokens
                    print(f"         📝 Added to Curator")

        # Calculate total ACE tokens
        ace_tokens = generator_tokens + reflector_tokens + curator_tokens

        print(f"   📊 ACE Role breakdown:")
        print(f"      🎯 Generator: {generator_tokens} tokens")
        print(f"      🔍 Reflector: {reflector_tokens} tokens")
        print(f"      📝 Curator: {curator_tokens} tokens")

        return (ace_tokens, generator_tokens, reflector_tokens, curator_tokens)

    except Exception as e:
        print(f"   Warning: Could not retrieve token usage from Opik: {e}")
        return 0, 0, 0, 0


def parse_domain_result(output: str, domain: str) -> dict:
    """Parse domain check result from agent output."""
    if not output:
        return {"status": "ERROR", "reason": "No output"}

    output_upper = output.upper()
    domain_upper = domain.upper()

    # Check for exact format first
    if f"AVAILABLE: {domain_upper}" in output_upper:
        return {"status": "AVAILABLE"}
    elif f"TAKEN: {domain_upper}" in output_upper:
        return {"status": "TAKEN"}

    # Check for natural language indicators of availability
    elif (
        ("AVAILABLE" in output_upper and domain_upper in output_upper)
        or ("ADD TO CART" in output_upper and domain_upper in output_upper)
        or ("PRICE:" in output_upper and domain_upper in output_upper)
        or (
            "REGISTRATION" in output_upper
            and "AVAILABLE" in output_upper
            and domain_upper in output_upper
        )
    ):
        return {"status": "AVAILABLE"}

    # Check for natural language indicators of taken/unavailable
    elif (
        ("TAKEN" in output_upper and domain_upper in output_upper)
        or ("REGISTERED" in output_upper and domain_upper in output_upper)
        or ("NOT AVAILABLE" in output_upper and domain_upper in output_upper)
        or ("UNAVAILABLE" in output_upper and domain_upper in output_upper)
    ):
        return {"status": "TAKEN"}

    else:
        return {
            "status": "ERROR",
            "reason": f"Could not parse result: {output[:100]}...",
        }


async def check_single_domain(agent: ACEAgent, domain: str) -> dict:
    """Check a single domain and return results with metrics, with retry logic."""
    print(f"🔍 Checking domain: {domain}")

    max_retries = 3
    last_error = None
    total_steps = 0
    attempt_details = []

    # Track browser-use tokens across all attempts
    total_browseruse_tokens = 0

    for attempt in range(max_retries):
        print(f"   🔄 Attempt {attempt + 1}/{max_retries}")

        try:
            # Create task for this specific domain
            task = DOMAIN_CHECK_TASK_TEMPLATE.format(domain=domain)

            # Run domain check with ACE learning (with timeout like baseline)
            history = await asyncio.wait_for(
                agent.run(task=task, max_steps=25), timeout=180.0
            )

            # Extract results
            output = history.final_result() if hasattr(history, "final_result") else ""
            steps = (
                len(history.action_names())
                if hasattr(history, "action_names") and history.action_names()
                else 0
            )

            # Add steps to total and track attempt
            total_steps += steps
            attempt_details.append(f"attempt {attempt + 1}: {steps} steps")

            # Parse domain check result
            result = parse_domain_result(output, domain)

            # If we got a valid status, collect tokens and return success
            if result["status"] != "ERROR":
                # For testing purposes, we need to determine what's actually correct
                # Since these are test domains, we'll assume they're AVAILABLE unless we can verify otherwise
                # In a real scenario, you'd check against a known ground truth
                expected_status = "AVAILABLE"  # Test domains should be available
                correct = result["status"] == expected_status

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

                # Method 2: Try agent internal token tracking (ACEAgent specific)
                if attempt_tokens == 0 and hasattr(agent, "browser_llm"):
                    try:
                        # ACEAgent uses browser_use Agent internally, check if it has token tracking
                        if hasattr(agent, "_last_agent") and hasattr(
                            agent._last_agent, "token_cost_service"
                        ):
                            usage_summary = (
                                await agent._last_agent.token_cost_service.get_usage_summary()
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
                    f"   🤖 Attempt {attempt + 1} browser tokens: {attempt_tokens} (total: {total_browseruse_tokens})"
                )

                return {
                    "domain": domain,
                    "status": result["status"],
                    "success": True,
                    "correct": correct,  # Whether the result was accurate
                    "expected": expected_status,
                    "steps": steps,  # Steps from final attempt
                    "total_steps": total_steps,  # Cumulative steps
                    "output": output,
                    "error": result.get("reason"),
                    "attempt": attempt + 1,
                    "attempt_details": attempt_details,
                    "browseruse_tokens": total_browseruse_tokens,
                }

            # Store error for potential retry
            last_error = f"Failed to get valid result: {output}"

        except asyncio.TimeoutError:
            # Calculate timeout steps (same as baseline logic)
            timeout_duration = 180.0
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

            steps = actual_steps + timeout_steps
            total_steps += steps
            attempt_details.append(
                f"attempt {attempt + 1}: {steps} steps (timeout, +{timeout_steps} for duration)"
            )
            last_error = f"Timeout on attempt {attempt + 1}"
            print(
                f"   ⏰ Timeout after {actual_steps} steps (+{timeout_steps} timeout penalty)"
            )

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

            # Method 2: Try agent internal token tracking (ACEAgent specific)
            if attempt_tokens == 0 and hasattr(agent, "browser_llm"):
                try:
                    # ACEAgent uses browser_use Agent internally, check if it has token tracking
                    if hasattr(agent, "_last_agent") and hasattr(
                        agent._last_agent, "token_cost_service"
                    ):
                        usage_summary = (
                            await agent._last_agent.token_cost_service.get_usage_summary()
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

    # All retries failed - use accumulated tokens from all attempts
    print(f"   ❌ All {max_retries} attempts failed")
    return {
        "domain": domain,
        "status": "ERROR",
        "success": False,
        "correct": False,
        "expected": "AVAILABLE",
        "steps": steps if "steps" in locals() else 0,
        "total_steps": total_steps,
        "error": f"Failed after {max_retries} attempts. Last error: {last_error}",
        "attempt": max_retries,
        "attempt_details": attempt_details,
        "browseruse_tokens": total_browseruse_tokens,
    }


async def main():
    """Run domain checking with ACE learning."""

    # Capture start time for trace filtering
    run_start_time = datetime.datetime.now(datetime.timezone.utc)

    # Configure observability
    try:
        configure_opik(project_name="ace-domain-checker")
        print("📊 Opik observability enabled")
    except:
        print("📊 Opik not available, continuing without observability")

    print("\n🔍 ACE + Browser-Use Domain Checker")
    print("🧠 Automated domain checking with learning!")
    print("=" * 60)

    # Setup playbook persistence
    playbook_path = Path(__file__).parent / "ace_domain_checker_playbook.json"

    # Create ACE agent - handles everything automatically!
    agent = ACEAgent(
        llm=ChatBrowserUse(),  # Browser automation LLM
        ace_model="claude-haiku-4-5-20251001",  # ACE learning LLM
        ace_max_tokens=4096,  # Enough for domain check analysis
        playbook_path=str(playbook_path) if playbook_path.exists() else None,
        max_steps=25,  # Browser automation steps
        calculate_cost=True,  # Track usage
    )

    # Show current knowledge
    if playbook_path.exists():
        print(f"📚 Loaded {len(agent.playbook.bullets())} learned strategies")
    else:
        print("🆕 Starting with empty playbook - learning from scratch")

    # Get test domains
    domains = get_test_domains()
    print(f"\n📋 Testing {len(domains)} domains:")
    for i, domain in enumerate(domains, 1):
        print(f"  {i}. {domain}")

    print(f"\n🎯 Each domain check will teach ACE new strategies")
    print("💡 ACE learns automatically after each execution\n")

    # Run domain checks with learning
    results = []
    for i, domain in enumerate(domains, 1):
        print(f"\n{'='*20} DOMAIN CHECK {i}/{len(domains)} {'='*20}")

        result = await check_single_domain(agent, domain)
        results.append(result)

        # Show immediate results
        status_icon = "✅" if result["success"] else "❌"
        steps = result["steps"]
        total_steps = result.get("total_steps", steps)
        attempt = result.get("attempt", 1)
        attempt_details = result.get("attempt_details", [])

        # Show detailed step breakdown for multiple attempts
        step_info = f"{total_steps} steps"
        if attempt > 1:
            step_info += f" total ({', '.join(attempt_details)})"
        else:
            step_info += f" (1 attempt)"

        print(f"{status_icon} {domain}: {result['status']} ({step_info})")

        if result["error"]:
            print(f"   Error: {result['error']}")

        # Small delay between checks to avoid rate limits
        if i < len(domains):
            print(f"⏳ Waiting 2 seconds before next check...")
            await asyncio.sleep(2)

    # Save learned strategies
    agent.save_playbook(str(playbook_path))

    # Query ACE tokens after all roles have completed
    print(f"\n💰 Querying ACE token usage after all domains processed...")
    time.sleep(5)  # Wait for Opik to index final traces
    (
        total_ace_tokens,
        total_generator_tokens,
        total_reflector_tokens,
        total_curator_tokens,
    ) = get_ace_token_usage(run_start_time)

    # Show final results
    print(f"\n{'='*60}")
    print("📊 DOMAIN CHECK RESULTS")
    print("=" * 60)
    print(
        f"{'#':<3} {'Domain':<25} {'Status':<10} {'Acc':<4} {'Steps':<8} {'Browser-Tokens':<13} {'Details'}"
    )
    print("-" * 93)

    total_steps = 0
    successful_checks = 0
    total_attempts = 0

    for i, result in enumerate(results, 1):
        status_icon = "✅" if result["success"] else "❌"
        steps = result["steps"]
        total_steps_domain = result.get("total_steps", steps)
        attempt = result.get("attempt", 1)
        attempt_details = result.get("attempt_details", [])

        total_steps += total_steps_domain
        total_attempts += attempt
        if result["success"]:
            successful_checks += 1

        # Show detailed step breakdown for multiple attempts
        if attempt > 1:
            step_details = f"({', '.join(attempt_details)})"
        else:
            step_details = "(1 attempt)"

        correct = result.get("correct", False)
        accuracy_indicator = "✓" if correct else "✗"
        browseruse_tokens = result.get("browseruse_tokens", 0)

        print(
            f"{i:<3} {result['domain']:<25} {result['status']:<10} {accuracy_indicator:<4} {total_steps_domain:<8} {browseruse_tokens:<12} {step_details}"
        )

        if not correct and result["success"]:
            expected = result.get("expected", "UNKNOWN")
            print(f"       Expected: {expected}, Got: {result['status']}")

    # Enhanced Summary
    successful = sum(1 for r in results if r["success"])
    correct = sum(1 for r in results if r.get("correct", False))
    domains_with_retries = sum(1 for r in results if r.get("attempt", 1) > 1)

    avg_steps_per_domain = total_steps / len(results) if results else 0

    # Calculate actual browser-use token usage
    total_browseruse_tokens = sum(r.get("browseruse_tokens", 0) for r in results)
    avg_browseruse_tokens_per_domain = (
        total_browseruse_tokens / len(results) if results else 0.0
    )

    # Calculate ACE token averages
    avg_ace_tokens_per_domain = total_ace_tokens / len(results) if results else 0.0

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
    print(
        f"{'🧠 ACE Tokens:':<25} {total_ace_tokens:>6} total     {avg_ace_tokens_per_domain:>6.1f} per domain"
    )
    print("=" * 80)

    # Show learned strategies
    strategies = agent.playbook.bullets()
    print(f"\n🎯 LEARNED STRATEGIES: {len(strategies)} total")
    print("-" * 60)

    if strategies:
        # Show recent strategies (last 5)
        recent_strategies = strategies[-5:] if len(strategies) > 5 else strategies

        for i, bullet in enumerate(recent_strategies, 1):
            helpful = bullet.helpful
            harmful = bullet.harmful
            effectiveness = (
                "✅" if helpful > harmful else "⚠️" if helpful == harmful else "❌"
            )
            print(f"{i}. {effectiveness} {bullet.content}")
            print(f"   (+{helpful}/-{harmful})")

        if len(strategies) > 5:
            print(f"   ... and {len(strategies) - 5} other strategies")

        print(f"\n💾 Strategies saved to: {playbook_path}")
        print("🔄 Next run will use these learned strategies automatically!")
    else:
        print("No new strategies learned (tasks may have failed)")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
