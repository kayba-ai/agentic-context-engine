#!/usr/bin/env python3
import asyncio
import os
import sys
import json
from datetime import datetime

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dotenv import load_dotenv

load_dotenv()

from browser_use import Agent, ChatBrowserUse

task = """
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


def parse_basket_data(output_text):
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


agent = Agent(task=task, llm=ChatBrowserUse())


async def run_grocery_shopping():
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


async def main():
    # Run grocery shopping and collect metrics
    result = await run_grocery_shopping()

    # Print results summary with metrics
    print_results_summary(result["result_text"])

    # Print metrics summary
    print(f"\n📊 PERFORMANCE METRICS:")
    print("=" * 50)
    print(f"🔄 Steps taken: {result['steps']}")
    print(f"🤖 Browser-use tokens: {result['browseruse_tokens']}")
    print(f"✅ Shopping success: {'Yes' if result['success'] else 'No'}")

    input("\n📱 Press Enter to close the browser...")


if __name__ == "__main__":
    asyncio.run(main())
