#!/usr/bin/env python3
"""
ACE + Browser-Use Grocery Shopping Demo

Simple demo showing ACE learning to improve at grocery shopping automation.
Uses the new ACEAgent integration for clean, automatic learning.
"""

import asyncio
import datetime
import os
import sys
import time
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dotenv import load_dotenv

load_dotenv()

# Import Opik for token tracking
try:
    import opik
except ImportError:
    opik = None

# Import ACE framework with new integration
from ace import ACEAgent
from ace.observability import configure_opik
from browser_use import ChatBrowserUse


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


# Shopping task definition
GROCERY_SHOPPING_TASK = """
### Migros Grocery Shopping - Essential 5 Items

**Objective:**
Shop for 5 essential items at Migros online store. Find the CHEAPEST available option for each item and add them to the basket.

**Essential Items:**
1. **1L full-fat milk**
2. **10 eggs (large)**
3. **1kg bananas**
4. **500g butter**
5. **1 loaf white bread (500g)**

**Instructions:**
- Visit https://www.migros.ch/en
- Search for each item and find the CHEAPEST option
- Add ONE of each to basket (don't checkout)
- Record item details: name, brand, price
- Provide final basket summary with total price

**Final Output Format:**
```
MIGROS BASKET:
- 1L milk: [brand] - CHF [price]
- 10 eggs: [brand] - CHF [price]
- 1kg bananas: [brand] - CHF [price]
- 500g butter: [brand] - CHF [price]
- White bread: [brand] - CHF [price]
TOTAL: CHF [total]
```

**Important:**
- Find CHEAPEST options only
- If exact match unavailable, choose closest alternative
- DO NOT complete purchase - basket only
"""


async def main():
    """Run grocery shopping with ACE learning."""

    # Capture start time for trace filtering
    run_start_time = datetime.datetime.now(datetime.timezone.utc)

    # Configure observability
    try:
        configure_opik(project_name="ace-grocery-shopping")
        print("📊 Opik observability enabled")
    except:
        print("📊 Opik not available, continuing without observability")

    print("\n🛒 ACE + Browser-Use Grocery Shopping")
    print("🧠 Automated shopping with learning!")
    print("=" * 60)

    # Setup playbook persistence
    playbook_path = Path(__file__).parent / "ace_grocery_playbook.json"

    # Create ACE agent - handles everything automatically!
    agent = ACEAgent(
        llm=ChatBrowserUse(),  # Browser automation LLM
        ace_model="claude-haiku-4-5-20251001",  # ACE learning LLM
        ace_max_tokens=4096,  # Enough for shopping analysis
        playbook_path=str(playbook_path) if playbook_path.exists() else None,
        max_steps=30,  # Browser automation steps
        calculate_cost=True,  # Track usage
    )

    # Show current knowledge
    if playbook_path.exists():
        print(f"📚 Loaded {len(agent.playbook.bullets())} learned strategies")
    else:
        print("🆕 Starting with empty playbook - learning from scratch")

    print(f"\n🎯 Task: Shop for 5 essential items at Migros")
    print("💡 ACE will learn from this shopping experience\n")

    try:
        # Run shopping with automatic ACE learning
        history = await agent.run(task=GROCERY_SHOPPING_TASK)

        print("\n" + "=" * 60)
        print("✅ SHOPPING COMPLETED")
        print("=" * 60)

        # Show results
        if hasattr(history, "final_result"):
            print("📋 Shopping Results:")
            print(history.final_result())

        # Show performance metrics
        steps = 0
        browseruse_tokens = 0
        ace_tokens = 0

        if hasattr(history, "number_of_steps"):
            steps = history.number_of_steps()

        # Extract browser-use tokens from history
        if hasattr(history, "usage"):
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
                print(f"⚠️ Could not extract browser-use tokens: {e}")

        # Query ACE tokens after shopping completed
        print(f"\n💰 Querying ACE token usage after shopping...")
        time.sleep(5)  # Wait for Opik to index final traces
        (
            ace_tokens,
            generator_tokens,
            reflector_tokens,
            curator_tokens,
        ) = get_ace_token_usage(run_start_time)

        print(f"\n📊 PERFORMANCE METRICS:")
        print("-" * 60)
        print(f"🔄 Steps taken: {steps}")
        print(f"🤖 Browser-use tokens: {browseruse_tokens:,}")
        print(f"🧠 ACE learning tokens: {ace_tokens:,}")
        print(f"💰 Total tokens: {browseruse_tokens + ace_tokens:,}")

    except Exception as e:
        print(f"\n❌ Shopping failed: {e}")
        print("🧠 ACE will still learn from this failure")

    # Save learned strategies
    agent.save_playbook(str(playbook_path))

    # Show what ACE learned
    strategies = agent.playbook.bullets()
    print(f"\n🎯 LEARNED STRATEGIES: {len(strategies)} total")
    print("-" * 60)

    if strategies:
        # Show recent strategies (last 3)
        recent_strategies = strategies[-3:] if len(strategies) > 3 else strategies

        for i, bullet in enumerate(recent_strategies, 1):
            helpful = bullet.helpful
            harmful = bullet.harmful
            effectiveness = (
                "✅" if helpful > harmful else "⚠️" if helpful == harmful else "❌"
            )
            print(f"{i}. {effectiveness} {bullet.content}")
            print(f"   (+{helpful}/-{harmful})")

        if len(strategies) > 3:
            print(f"   ... and {len(strategies) - 3} older strategies")

        print(f"\n💾 Strategies saved to: {playbook_path}")
        print("🔄 Next run will use these learned strategies automatically!")
    else:
        print("No new strategies learned (task may have failed)")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
