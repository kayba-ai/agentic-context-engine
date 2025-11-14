#!/usr/bin/env python3
"""
ACE + Browser-Use Grocery Shopping Demo

Advanced demo showing ACE learning to improve at grocery shopping automation.
Uses OnlineAdapter for incremental learning after each shopping scenario.
"""

import asyncio
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dotenv import load_dotenv

load_dotenv()

# Import browser-use for actual shopping execution
from browser_use import Agent, Browser, ChatBrowserUse

# Import ACE framework for learning
from ace import (
    LiteLLMClient,
    Generator,
    Reflector,
    Curator,
    OnlineAdapter,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    Playbook,
)
from ace.prompts_v2_1 import PromptManager
from ace.observability import configure_opik

# Import common utilities from parent directory
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import (
    calculate_timeout_steps,
    format_result_output,
    MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
)
from debug import print_history_details

try:
    import opik

    client = opik.Opik()
except:
    client = None

# Base grocery shopping task template - will be enhanced by ACE Generator
BASE_GROCERY_TASK = """
### Migros Grocery Shopping Test - Essential 5 Items

**Objective:**
Shop for 5 essential items at Migros online store to test the shopping automation.
Find the CHEAPEST available option for each item and add them to the basket.

**Essential Items to Find:**
1. **1L full-fat milk** - Most frequently purchased, stable pricing
2. **10 eggs (large)** - Price-sensitive staple, easy to compare
3. **1kg bananas** - Fresh produce benchmark
4. **500g butter** - Higher-value dairy item
5. **1 loaf white bread (500g)** - Daily staple

---

### Shopping Instructions:
- Visit https://www.migros.ch/en
- Search for each of the 5 items above
- Find the CHEAPEST option that meets specifications
- Add ONE of each to basket (don't checkout)
- Note: item name, brand, price, price per unit if shown
- Record total basket price

### Important Instructions:
- For each item, find the CHEAPEST available option that meets specifications
- If exact match unavailable, choose closest alternative and note the difference
- If multiple cheap options exist, note the price range (min-max)
- If website requires login, continue as guest with address: 9000 St. Gallen, Kesslerstrasse 2
- DO NOT complete checkout - only add to basket for price comparison

### Final Output Required:
Create a clear summary showing:

**MIGROS BASKET:**
- 1L milk: [brand] - CHF [price] ([price per L])
- 10 eggs: [brand] - CHF [price] ([price per egg])
- 1kg bananas: [brand] - CHF [price] ([price per kg])
- 500g butter: [brand] - CHF [price] ([price per 100g])
- White bread: [brand] - CHF [price] ([price per 100g])
- **TOTAL: CHF [total]**

Note any substitutions or unavailable items clearly.
"""


class GroceryShoppingEnvironment(TaskEnvironment):
    """Environment that evaluates grocery shopping performance using browser automation."""

    def __init__(
        self,
        headless: bool = True,
        run_start_time=None,
    ):
        self.headless = headless
        self.run_start_time = run_start_time

    def evaluate(self, sample: Sample, generator_output):
        """Run browser automation for grocery shopping and evaluate the result."""

        # Extract shopping scenario from sample
        scenario = sample.question
        strategy = generator_output.final_answer

        print(f"🛒 Shopping scenario: {scenario[:50]}...")
        print(f"🧠 Using strategy: {strategy[:100]}...")

        # Capture current trace ID for token tracking
        trace_id = None
        try:
            if client:
                from opik import opik_context

                trace_data = opik_context.get_current_trace_data()
                trace_id = trace_data.id if trace_data else None
                print(
                    f"   🆔 Captured trace ID: {trace_id[:8] if trace_id else 'None'}..."
                )
        except Exception as e:
            print(f"   ⚠️ Failed to get trace ID: {e}")

        # Run browser automation with strategy
        result = asyncio.run(self._execute_shopping(scenario, strategy))

        # Get browser-use tokens from result
        browseruse_tokens = result.get("browseruse_tokens", 0)
        print(f"   📊 Browser tokens: {browseruse_tokens}")

        # Evaluate shopping performance
        success = result["success"]
        items_found = result.get("items_found", 0)
        target_items = 5  # Our target is 5 essential items

        # Calculate shopping metrics
        item_completion = items_found / target_items if target_items > 0 else 0
        total_price = result.get("total_price", 0)
        steps_taken = result["steps"]

        # Pass ONLY raw browser-use execution logs to the reflector
        # No interpretation, analysis, or additional context - just the raw logs
        execution_logs = result.get("execution_logs", [])
        if execution_logs:
            feedback = "\n".join(execution_logs)
        else:
            # Fallback: minimal execution info if logs unavailable
            feedback = f"Browser execution completed with {steps_taken} steps. Success: {success}."

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=None,  # No strict ground truth for grocery shopping
            metrics={
                "success": success,
                "items_found": items_found,
                "item_completion": item_completion,
                "total_price": total_price,
                "steps": steps_taken,
                "browseruse_tokens": browseruse_tokens,
                "efficiency_score": max(
                    0, 1 - (steps_taken - 15) / 25
                ),  # Efficiency based on steps
            },
        )

    def _get_token_usage(self, trace_id: str = None) -> tuple[int, int, int, int]:
        """Query Opik for ACE token usage only.

        Returns:
            tuple: (ace_tokens, generator_tokens, reflector_tokens, curator_tokens)
        """
        if not client:
            return 0, 0, 0, 0

        try:
            import datetime

            # Create client and flush to ensure data is sent
            client.flush()

            # Use run start time if available, otherwise fall back to last 10 minutes
            if self.run_start_time:
                recent_time = self.run_start_time.isoformat().replace("+00:00", "Z")
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

            # Search ACE project for role breakdown
            for project in ["ace-roles"]:
                try:
                    traces = client.search_traces(
                        project_name=project,
                        filter_string=f'start_time >= "{recent_time}"',
                        max_results=50,
                    )
                    print(
                        f"   📊 Found {len(traces)} recent traces in '{project}' project"
                    )
                    all_traces.extend(traces)
                except Exception as e:
                    print(f"   ⚠️ Failed to search '{project}' project: {e}")

            # Track individual ACE role tokens
            generator_tokens = 0
            reflector_tokens = 0
            curator_tokens = 0

            # Process ACE role traces
            for trace in all_traces:
                trace_name = getattr(trace, "name", "unknown")
                trace_name_lower = trace_name.lower()

                if any(
                    role in trace_name_lower
                    for role in ["generator", "reflector", "curator"]
                ):
                    total_tokens = 0

                    if trace.usage:
                        total_tokens = trace.usage.get("total_tokens", 0)
                    else:
                        try:
                            spans = client.search_spans(trace_id=trace.id)
                            for span in spans:
                                if hasattr(span, "usage") and span.usage:
                                    span_tokens = span.usage.get("total_tokens", 0)
                                    total_tokens += span_tokens
                        except Exception as e:
                            print(f"         ⚠️ Failed to get spans: {e}")

                    # Classify by role
                    if "generator" in trace_name_lower:
                        generator_tokens += total_tokens
                    elif "reflector" in trace_name_lower:
                        reflector_tokens += total_tokens
                    elif "curator" in trace_name_lower:
                        curator_tokens += total_tokens

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

    async def _execute_shopping(self, scenario: str, strategy: str):
        """Execute browser automation for grocery shopping."""

        # Track metrics
        steps = 0
        browseruse_tokens = 0

        try:
            # Create enhanced task with strategy
            enhanced_task = f"""
{BASE_GROCERY_TASK}

=== STRATEGY GUIDANCE ===
{strategy}

=== SPECIFIC SCENARIO ===
{scenario}

Remember: Use the strategy above to guide your approach, but adapt as needed for this specific scenario.
"""

            # Create agent with ChatBrowserUse (same as baseline)
            llm = ChatBrowserUse()
            agent = Agent(
                task=enhanced_task,
                llm=llm,
                max_actions_per_step=5,
                max_steps=20,
                calculate_cost=True,
            )

            # Run the shopping task (no timeout, same as baseline)
            history = await agent.run()

            # Extract step count
            if history and hasattr(history, "action_names") and history.action_names():
                steps = len(history.action_names())
            else:
                steps = 0

            # Get the final result text
            output = (
                history.final_result()
                if hasattr(history, "final_result")
                else "No output captured"
            )

            # Extract browser-use token usage (same as baseline)
            # Method 1: Try to get tokens from history
            if history and hasattr(history, "usage"):
                try:
                    usage = history.usage
                    if usage:
                        if hasattr(usage, "total_tokens"):
                            browseruse_tokens = usage.total_tokens
                        elif isinstance(usage, dict) and "total_tokens" in usage:
                            browseruse_tokens = usage["total_tokens"]
                        elif hasattr(usage, "input_tokens") and hasattr(
                            usage, "output_tokens"
                        ):
                            browseruse_tokens = usage.input_tokens + usage.output_tokens
                        elif (
                            isinstance(usage, dict)
                            and "input_tokens" in usage
                            and "output_tokens" in usage
                        ):
                            browseruse_tokens = (
                                usage["input_tokens"] + usage["output_tokens"]
                            )
                except Exception as e:
                    print(f"⚠️ Could not get tokens from history: {e}")

            # Method 2: Try agent.token_cost_service
            if browseruse_tokens == 0 and agent:
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
                                browseruse_tokens = usage_summary["total_tokens"]
                            elif hasattr(usage_summary, "total_tokens"):
                                browseruse_tokens = usage_summary.total_tokens
                except Exception as e:
                    print(f"⚠️ Could not get tokens from agent service: {e}")

            # Parse shopping results
            shopping_results = self._parse_shopping_results(output)

            # Extract RAW execution logs from browser-use agent ONLY
            # No analysis, no interpretation, no metrics - just the raw logs
            execution_logs = []
            try:
                # Final output from agent
                execution_logs.append(f"FINAL OUTPUT:\n{output}")

                # Action history - step by step actions taken
                if hasattr(history, "action_history") and history.action_history():
                    execution_logs.append(f"\nACTION HISTORY:")
                    for i, action in enumerate(history.action_history(), 1):
                        execution_logs.append(f"{i}. {action}")

                # Action results - outcomes of each action
                if hasattr(history, "action_results") and history.action_results():
                    execution_logs.append(f"\nACTION RESULTS:")
                    for i, result in enumerate(history.action_results(), 1):
                        execution_logs.append(f"{i}. {result}")

                # URLs visited during execution
                if hasattr(history, "urls") and history.urls():
                    execution_logs.append(f"\nURLs VISITED:")
                    execution_logs.append(f"{history.urls()}")

                # Error details
                if hasattr(history, "errors") and history.errors():
                    execution_logs.append(f"\nERRORS:")
                    execution_logs.append(f"{history.errors()}")

                # Model thoughts/reasoning (if available)
                if hasattr(history, "model_thoughts") and history.model_thoughts():
                    execution_logs.append(f"\nAGENT THOUGHTS:")
                    for i, thought in enumerate(history.model_thoughts(), 1):
                        execution_logs.append(f"{i}. {thought}")

                # Action names (high-level action types)
                if hasattr(history, "action_names") and history.action_names():
                    execution_logs.append(f"\nACTION NAMES:")
                    action_names = history.action_names()
                    execution_logs.append(f"{action_names}")

            except Exception as e:
                execution_logs.append(f"Error extracting logs: {e}")

            return {
                "success": True,
                "steps": steps,
                "items_found": shopping_results["items_found"],
                "total_price": shopping_results["total_price"],
                "output": output,
                "browseruse_tokens": browseruse_tokens,
                "execution_logs": execution_logs,
            }

        except Exception as e:
            print(f"❌ Error during shopping: {str(e)}")
            return {
                "success": False,
                "steps": steps,
                "items_found": 0,
                "total_price": 0,
                "error": f"Shopping failed: {str(e)}",
                "browseruse_tokens": browseruse_tokens,
                "execution_logs": [],
            }

    def _parse_shopping_results(self, output: str) -> dict:
        """Parse shopping results from agent output to extract items and total price."""
        items_found = 0
        total_price = 0.0

        # Count items found (look for numbered items or bullet points)
        import re

        # Look for item patterns like "1. milk", "- eggs", etc.
        item_patterns = [
            r"\d+\.\s+\*\*[^:]+:\*\*",  # Numbered items with bold formatting
            r"-\s*\*\*[^:]+:\*\*",  # Bullet points with bold formatting
            r"\d+\.\s+[A-Za-z]",  # Simple numbered items
            r"-\s*[A-Za-z]",  # Simple bullet points
        ]

        for pattern in item_patterns:
            matches = re.findall(pattern, output)
            items_found = max(items_found, len(matches))

        # Look for total price patterns
        total_patterns = [
            r"\*\*TOTAL:\s*CHF\s*(\d+(?:\.\d{2})?)\*\*",
            r"TOTAL:\s*CHF\s*(\d+(?:\.\d{2})?)",
            r"CHF\s*(\d+(?:\.\d{2})?)\s*total",
        ]

        for pattern in total_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                try:
                    total_price = float(match.group(1))
                    break
                except ValueError:
                    continue

        return {
            "items_found": items_found,
            "total_price": total_price,
        }


def parse_basket_data_legacy(output_text):
    """Parse basket data from agent output to extract exact items and prices."""
    import re

    stores_data = {}

    # Look for the final basket summary - the agent uses "# MIGROS BASKET - SHOPPING COMPLETE"
    # Then look for the "**MIGROS BASKET:**" section within that
    basket_pattern = r"(?i)\*?\*?MIGROS\s+BASKET:\*?\*?\s*(.*?)(?=---|\*?\*?TOTAL|$)"
    match = re.search(basket_pattern, output_text, re.DOTALL)

    if match:
        basket_section = match.group(1).strip()
        items = []
        total = "Not found"
        total_value = None

        # Parse numbered items like "1. **1L milk:** Valflora IP-SUISSE Whole milk HOCH PAST 3.5% Fat - **CHF 1.40**"
        item_pattern = r"(\d+)\.\s+\*\*([^:]+):\*\*\s+([^-]+)\s+-\s+\*\*CHF\s+(\d+(?:\.\d{2})?)\*\*\s*(?:\([^)]+\))?"
        item_matches = re.findall(item_pattern, basket_section)

        for item_match in item_matches:
            number, item_type, product_name, price = item_match
            item_str = f"{number}. {item_type}: {product_name.strip()} - CHF {price}"
            items.append(item_str)

        # Look for total - pattern like "## **TOTAL: CHF 15.75**"
        total_pattern = r"(?i)\*?\*?TOTAL:\s*CHF\s*(\d+(?:\.\d{2})?)\*?\*?"
        total_match = re.search(total_pattern, output_text)
        if total_match:
            total_value = float(total_match.group(1))
            total = f"TOTAL: CHF {total_value}"

        stores_data["MIGROS"] = {
            "items": items,
            "total": total,
            "total_value": total_value,
        }

    return stores_data


def parse_basket_data(output_text):
    """Parse basket data from agent output to extract exact items and prices."""
    return parse_basket_data_legacy(output_text)


def print_results_summary(output_text):
    """Print a formatted summary showing exact basket items and prices per store."""
    print("\n" + "=" * 80)
    print("🛒 GROCERY PRICE COMPARISON RESULTS")
    print("=" * 80)
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏪 Store: Migros (Test Run)")
    print(f"📦 Items: 5 essential grocery items")

    # Parse basket data from agent output
    stores_data = parse_basket_data(output_text)

    print("\n📋 BASKET ITEMS & PRICES:")
    print("=" * 80)

    if stores_data:
        for store_name, basket_info in stores_data.items():
            print(f"\n🏪 {store_name} BASKET:")
            print("-" * 50)

            if basket_info["items"]:
                for i, item in enumerate(basket_info["items"], 1):
                    print(f"  {i}. {item}")
                print(f"\n  💰 {basket_info['total']}")
            else:
                print("  ⚠️ No items found or basket data incomplete")
                print("  📝 Check detailed output below for manual review")

        # Price comparison if we have totals
        totals_available = [
            (store, info["total_value"])
            for store, info in stores_data.items()
            if info["total_value"] is not None
        ]

        if len(totals_available) >= 2:
            print(f"\n🏆 PRICE COMPARISON:")
            print("-" * 50)
            totals_available.sort(key=lambda x: x[1])
            winner = totals_available[0]
            most_expensive = totals_available[-1]
            savings = most_expensive[1] - winner[1]

            print(f"  🥇 Cheapest: {winner[0]} - CHF {winner[1]:.2f}")
            print(
                f"  🥉 Most expensive: {most_expensive[0]} - CHF {most_expensive[1]:.2f}"
            )
            print(f"  💸 You save: CHF {savings:.2f} by choosing {winner[0]}")
        elif len(totals_available) == 1:
            print(
                f"\n⚠️ Only one store total found: {totals_available[0][0]} - CHF {totals_available[0][1]:.2f}"
            )
        else:
            print(f"\n⚠️ Could not extract store totals for comparison")
    else:
        print("\n⚠️ Could not parse basket data from output")

    print("=" * 80)


def print_single_run_results(result, total_ace_tokens, role_breakdown):
    """Print results summary for single ACE run."""
    print("\n" + "=" * 80)
    print("🛒 ACE GROCERY SHOPPING RESULTS")
    print("=" * 80)
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏪 Store: Migros (ACE Enhanced)")
    print(f"📦 Mode: Single run with learning")

    print("\n📋 SHOPPING RESULTS:")
    print("=" * 80)

    metrics = result.environment_result.metrics
    success = "✓" if metrics.get("success", False) else "✗"
    items = metrics.get("items_found", 0)
    price = metrics.get("total_price", 0)
    steps = metrics.get("steps", 0)
    efficiency = metrics.get("efficiency_score", 0)
    browseruse_tokens = metrics.get("browseruse_tokens", 0)

    print(f"Success: {success}")
    print(f"Items found: {items}/5")
    print(f"Total price: CHF {price:.2f}")
    print(f"Steps taken: {steps}")
    print(f"Efficiency score: {efficiency:.2f}")

    generator_tokens, reflector_tokens, curator_tokens = role_breakdown

    print("\n" + "=" * 80)
    print("📊 PERFORMANCE METRICS")
    print("=" * 80)
    print(f"{'🤖 Browser-Use Tokens:':<25} {browseruse_tokens:>6}")
    print(f"{'🧠 ACE Learning Tokens:':<25} {total_ace_tokens:>6}")
    print()
    print("🧠 ACE Learning Breakdown:")
    print(f"   🎯 Generator:      {generator_tokens:>6} tokens  (strategy planning)")
    print(f"   🔍 Reflector:      {reflector_tokens:>6} tokens  (performance analysis)")
    print(f"   📝 Curator:        {curator_tokens:>6} tokens  (strategy updates)")
    print(f"   {'─' * 40}")
    print(f"   🧠 Total ACE:      {total_ace_tokens:>6} tokens")
    print("=" * 80)


async def run_grocery_shopping_legacy():
    """Run grocery shopping and collect metrics."""
    print("🛒 Starting Migros shopping test...")
    print("Testing automation with 5 essential items at Migros only")
    print("-" * 50)

    # Track metrics
    steps = 0
    browseruse_tokens = 0

    try:
        # Run the shopping task
        history = await agent.run()

        # Extract step count
        if history and hasattr(history, "action_names") and history.action_names():
            steps = len(history.action_names())
        else:
            steps = 0

        # Get the final result text
        result_text = (
            history.final_result()
            if hasattr(history, "final_result")
            else "No output captured"
        )

        # Extract browser-use token usage
        # Method 1: Try to get tokens from history
        if history and hasattr(history, "usage"):
            try:
                usage = history.usage
                if usage:
                    if hasattr(usage, "total_tokens"):
                        browseruse_tokens = usage.total_tokens
                    elif isinstance(usage, dict) and "total_tokens" in usage:
                        browseruse_tokens = usage["total_tokens"]
                    elif hasattr(usage, "input_tokens") and hasattr(
                        usage, "output_tokens"
                    ):
                        browseruse_tokens = usage.input_tokens + usage.output_tokens
                    elif (
                        isinstance(usage, dict)
                        and "input_tokens" in usage
                        and "output_tokens" in usage
                    ):
                        browseruse_tokens = (
                            usage["input_tokens"] + usage["output_tokens"]
                        )
            except Exception as e:
                print(f"⚠️ Could not get tokens from history: {e}")

        # Method 2: Try agent.token_cost_service
        if browseruse_tokens == 0 and agent:
            try:
                if hasattr(agent, "token_cost_service"):
                    usage_summary = await agent.token_cost_service.get_usage_summary()
                    if usage_summary:
                        if (
                            isinstance(usage_summary, dict)
                            and "total_tokens" in usage_summary
                        ):
                            browseruse_tokens = usage_summary["total_tokens"]
                        elif hasattr(usage_summary, "total_tokens"):
                            browseruse_tokens = usage_summary.total_tokens
            except Exception as e:
                print(f"⚠️ Could not get tokens from agent service: {e}")

        return {
            "result_text": str(result_text),
            "steps": steps,
            "browseruse_tokens": browseruse_tokens,
            "success": True,
        }

    except Exception as e:
        print(f"❌ Error during shopping: {str(e)}")
        return {
            "result_text": f"Shopping failed: {str(e)}",
            "steps": steps,
            "browseruse_tokens": browseruse_tokens,
            "success": False,
        }


def main():
    """Main function - ACE online learning for grocery shopping."""

    # Capture start time for trace filtering
    import datetime

    run_start_time = datetime.datetime.now(datetime.timezone.utc)

    # Configure Opik if available
    try:
        configure_opik(project_name="ace-browser-grocery-shopping")
        print("📊 Opik observability enabled")
    except:
        print("📊 Opik not available, continuing without observability")

    print("\n🚀 ACE + Browser-Use Grocery Shopping")
    print("🧠 Learns from each shopping experience!")
    print("=" * 50)

    # Load existing playbook or create new one
    playbook_path = Path(__file__).parent / "ace_grocery_shopping_playbook.json"

    if playbook_path.exists():
        print(f"📚 Loading existing playbook from {playbook_path}")
        playbook = Playbook.load_from_file(str(playbook_path))
        print(f"🧠 Loaded {len(playbook.bullets())} existing strategies")

        # Show existing strategies
        if playbook.bullets():
            print("📋 Existing strategies:")
            for i, bullet in enumerate(playbook.bullets(), 1):
                helpful_score = bullet.helpful_count
                harmful_score = bullet.harmful_count
                net_score = helpful_score - harmful_score
                effectiveness = (
                    "+++"
                    if net_score >= 2
                    else "++" if net_score >= 1 else "+" if net_score >= 0 else "-"
                )
                print(f"  {i}. [{effectiveness:>3}] {bullet.content}")
    else:
        print(f"🆕 Creating new playbook (will save to {playbook_path})")
        playbook = Playbook()

    # Create ACE components with OnlineAdapter (using LiteLLM for ACE roles)
    llm = LiteLLMClient(model="claude-haiku-4-5-20251001", temperature=0.2)

    # Create separate LLM client for Reflector with higher max_tokens to handle large raw logs
    reflector_llm = LiteLLMClient(
        model="claude-haiku-4-5-20251001",
        temperature=0.2,
        max_tokens=8192,  # Increased from default 512 to handle verbose browser-use logs
    )

    # Create prompt manager for enhanced prompts
    manager = PromptManager()

    adapter = OnlineAdapter(
        playbook=playbook,  # Use loaded playbook instead of empty one
        generator=Generator(llm, prompt_template=manager.get_generator_prompt()),
        reflector=Reflector(
            reflector_llm, prompt_template=manager.get_reflector_prompt()
        ),
        curator=Curator(llm, prompt_template=manager.get_curator_prompt()),
        max_refinement_rounds=2,
    )

    # Create environment
    environment = GroceryShoppingEnvironment(
        headless=True,  # Set False to see browser (slower but easier to debug)
        run_start_time=run_start_time,  # Pass start time for trace filtering
    )

    print("\n🔄 Starting single ACE-enhanced shopping run...\n")

    # Single shopping scenario - same task as baseline but with ACE learning
    sample = Sample(
        question="Shop for 5 essential grocery items at Migros, focusing on finding the cheapest options available",
        ground_truth="Successful shopping with all 5 items found and accurate pricing",
        context="Find the cheapest available option for each item: 1L full-fat milk, 10 eggs (large), 1kg bananas, 500g butter, 1 loaf white bread (500g). Optimize for both efficiency and effectiveness.",
    )

    # Run single execution with learning
    print(f"📋 Processing single shopping scenario with ACE learning...")
    results = adapter.run([sample], environment)  # Pass single sample in a list
    result = results[0]  # Extract the single result

    # Query ACE tokens after learning completed
    print(f"\n💰 Querying ACE token usage after shopping and learning...")
    import time

    time.sleep(5)  # Wait for Opik to index final traces
    (
        total_ace_tokens,
        total_generator_tokens,
        total_reflector_tokens,
        total_curator_tokens,
    ) = environment._get_token_usage()

    # Show single run results
    print_single_run_results(
        result,
        total_ace_tokens,
        (total_generator_tokens, total_reflector_tokens, total_curator_tokens),
    )

    # Show learned strategies
    if adapter.playbook.bullets():
        print(f"\n🎯 LEARNED SHOPPING STRATEGIES:")
        print("-" * 50)
        for i, bullet in enumerate(adapter.playbook.bullets(), 1):
            helpful_score = bullet.helpful_count
            harmful_score = bullet.harmful_count
            net_score = helpful_score - harmful_score
            effectiveness = (
                "+++"
                if net_score >= 2
                else "++" if net_score >= 1 else "+" if net_score >= 0 else "-"
            )
            print(f"  {i}. [{effectiveness:>3}] {bullet.content}")
        print()
        print(
            "Legend: +++ Very Effective, ++ Effective, + Helpful, - Needs Improvement"
        )
    else:
        print(f"\n💡 No new strategies learned in this run")
        print(
            "   Strategies are learned from execution feedback and saved for future runs"
        )

    # Save playbook for future use in the same directory
    adapter.playbook.save_to_file(str(playbook_path))
    print(f"\n💾 Playbook saved to {playbook_path}")
    print("💡 Next run will automatically load these strategies!")


async def main_legacy():
    # Legacy main function for backward compatibility
    result = await run_grocery_shopping_legacy()
    print_results_summary(result["result_text"])
    print(f"\n📊 PERFORMANCE METRICS:")
    print("=" * 50)
    print(f"🔄 Steps taken: {result['steps']}")
    print(f"🤖 Browser-use tokens: {result['browseruse_tokens']}")
    print(f"✅ Shopping success: {'Yes' if result['success'] else 'No'}")
    input("\n📱 Press Enter to close the browser...")


if __name__ == "__main__":
    # Run ACE learning version (comment out for legacy version)
    main()

    # Uncomment for legacy version:
    # asyncio.run(main_legacy())
