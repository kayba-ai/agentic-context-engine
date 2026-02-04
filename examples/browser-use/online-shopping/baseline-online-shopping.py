#!/usr/bin/env python3
"""
Baseline Browser-Use Grocery Shopping Demo

Run grocery shopping automation multiple times and aggregate results.

Usage:
    uv run python examples/browser-use/online-shopping/baseline-online-shopping.py
    uv run python examples/browser-use/online-shopping/baseline-online-shopping.py --runs 3
    uv run python examples/browser-use/online-shopping/baseline-online-shopping.py --runs 5 --no-pause
"""
import argparse
import asyncio
import os
import sys
import json
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dotenv import load_dotenv

load_dotenv()

from browser_use import Agent, ChatBrowserUse

TASK = """
### Migros Grocery Shopping - Essential 5 Items

**Objective:**
Shop for 5 essential items at Migros online store. Find the CHEAPEST available option for each item and add them to the basket.

**Essential Items:**
1. **1L full-fat milk**
2. **10 eggs (large)**
3. **1kg bananas**
4. **500g butter**
5. **1 loaf fresh white bread (500g)**

**Instructions:**
- Visit https://www.migros.ch/en
- Search for each item and find the CHEAPEST option
- Add each item to basket (don't checkout)
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

# Expected correct items for accuracy calculation
EXPECTED_ITEMS = {
    "milk": {"brand": "Valflora", "price": "1.40"},
    "eggs": {"brand": "M-Budget", "price": "4.25"},
    "bananas": {"brand": "Fresca", "price": "2.55"},
    "butter": {"brand": "M-Budget", "price": "7."},
    "bread": {"brand": "Fleur de Pains", "price": "3.6"},
}


@dataclass
class RunResult:
    """Results from a single shopping run."""

    run_number: int
    success: bool
    result_text: str
    steps: int
    tokens: int
    correct_items: int
    accuracy_pct: float
    duration_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def parse_basket_data(output_text):
    """Parse basket data from agent output to extract exact items and prices."""
    import re

    stores_data = {}

    basket_pattern = r"(?i)\*?\*?MIGROS\s+BASKET:\*?\*?\s*(.*?)(?=---|\*?\*?TOTAL|$)"
    match = re.search(basket_pattern, output_text, re.DOTALL)

    if match:
        basket_section = match.group(1).strip()
        items = []
        total = "Not found"
        total_value = None

        item_pattern = r"(\d+)\.\s+\*\*([^:]+):\*\*\s+([^-]+)\s+-\s+\*\*CHF\s+(\d+(?:\.\d{2})?)\*\*\s*(?:\([^)]+\))?"
        item_matches = re.findall(item_pattern, basket_section)

        for item_match in item_matches:
            number, item_type, product_name, price = item_match
            item_str = f"{number}. {item_type}: {product_name.strip()} - CHF {price}"
            items.append(item_str)

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


def calculate_accuracy(result_text: str) -> tuple[int, float]:
    """Calculate accuracy based on expected items."""
    if not result_text:
        return 0, 0.0

    correct_items = 0

    # Milk: Valflora @ 1.40
    if "Valflora" in result_text and "1.40" in result_text:
        correct_items += 1

    # Eggs: M-Budget @ 4.25
    if (
        "M-Budget" in result_text
        and "eggs" in result_text.lower()
        and "4.25" in result_text
    ):
        correct_items += 1

    # Bananas: Fresca @ 2.55
    if (
        "Fresca" in result_text
        and ("bananas" in result_text.lower())
        and "2.55" in result_text
    ):
        correct_items += 1

    # Butter: M-Budget @ 7.00
    if (
        "M-Budget" in result_text
        and "butter" in result_text.lower()
        and "7." in result_text
    ):
        correct_items += 1

    # Bread: Fleur de Pains @ 3.60
    if "Fleur de Pains" in result_text and "3.6" in result_text:
        correct_items += 1

    accuracy_pct = (correct_items / 5) * 100
    return correct_items, accuracy_pct


def print_run_summary(run_result: RunResult):
    """Print summary for a single run."""
    status = "SUCCESS" if run_result.success else "FAILED"
    print(f"\n{'=' * 70}")
    print(f"RUN {run_result.run_number} - {status}")
    print(f"{'=' * 70}")
    print(f"  Duration: {run_result.duration_seconds:.1f}s")
    print(f"  Steps: {run_result.steps}")
    print(f"  Tokens: {run_result.tokens:,}")
    print(f"  Accuracy: {run_result.correct_items}/5 ({run_result.accuracy_pct:.0f}%)")

    # Parse and show basket items
    stores_data = parse_basket_data(run_result.result_text)
    if stores_data and "MIGROS" in stores_data:
        basket = stores_data["MIGROS"]
        if basket["items"]:
            print(f"\n  Basket Items:")
            for item in basket["items"]:
                print(f"    {item}")
            if basket["total_value"]:
                print(f"    Total: CHF {basket['total_value']:.2f}")


def print_aggregate_summary(results: List[RunResult]):
    """Print aggregated summary of all runs."""
    if not results:
        print("\nNo results to summarize.")
        return

    total_runs = len(results)
    successful_runs = sum(1 for r in results if r.success)
    failed_runs = total_runs - successful_runs

    # Calculate aggregates
    total_steps = sum(r.steps for r in results)
    total_tokens = sum(r.tokens for r in results)
    total_duration = sum(r.duration_seconds for r in results)
    total_correct = sum(r.correct_items for r in results)
    max_possible_correct = total_runs * 5

    avg_steps = total_steps / total_runs if total_runs > 0 else 0
    avg_tokens = total_tokens / total_runs if total_runs > 0 else 0
    avg_duration = total_duration / total_runs if total_runs > 0 else 0
    avg_accuracy = (
        (total_correct / max_possible_correct) * 100 if max_possible_correct > 0 else 0
    )

    # Per-run accuracy stats
    accuracies = [r.accuracy_pct for r in results]
    min_accuracy = min(accuracies) if accuracies else 0
    max_accuracy = max(accuracies) if accuracies else 0

    print(f"\n{'=' * 70}")
    print(f"AGGREGATE SUMMARY - {total_runs} RUNS")
    print(f"{'=' * 70}")

    print(f"\n  Run Statistics:")
    print(
        f"    Successful: {successful_runs}/{total_runs} ({(successful_runs/total_runs)*100:.0f}%)"
    )
    print(f"    Failed: {failed_runs}/{total_runs}")

    print(f"\n  Performance Metrics:")
    print(f"    Total Duration: {total_duration:.1f}s")
    print(f"    Avg Duration: {avg_duration:.1f}s per run")
    print(f"    Total Steps: {total_steps}")
    print(f"    Avg Steps: {avg_steps:.1f} per run")
    print(f"    Total Tokens: {total_tokens:,}")
    print(f"    Avg Tokens: {avg_tokens:,.0f} per run")

    print(f"\n  Accuracy:")
    print(f"    Total Correct Items: {total_correct}/{max_possible_correct}")
    print(f"    Overall Accuracy: {avg_accuracy:.1f}%")
    print(f"    Min Run Accuracy: {min_accuracy:.0f}%")
    print(f"    Max Run Accuracy: {max_accuracy:.0f}%")

    # Per-run breakdown table
    print(f"\n  Per-Run Breakdown:")
    print(
        f"    {'Run':<5} {'Status':<8} {'Duration':<10} {'Steps':<7} {'Tokens':<10} {'Accuracy':<10}"
    )
    print(f"    {'-'*5} {'-'*8} {'-'*10} {'-'*7} {'-'*10} {'-'*10}")
    for r in results:
        status = "OK" if r.success else "FAIL"
        print(
            f"    {r.run_number:<5} {status:<8} {r.duration_seconds:<10.1f} {r.steps:<7} {r.tokens:<10,} {r.correct_items}/5 ({r.accuracy_pct:.0f}%)"
        )

    print(f"\n{'=' * 70}")


async def run_grocery_shopping(run_number: int) -> RunResult:
    """Run grocery shopping and collect metrics."""
    print(f"\n{'#' * 70}")
    print(f"# STARTING RUN {run_number}")
    print(f"{'#' * 70}")
    print(f"Testing automation with 5 essential items at Migros")

    start_time = time.time()
    steps = 0
    browseruse_tokens = 0
    result_text = ""
    success = False

    # Create fresh agent for each run
    agent = Agent(task=TASK, llm=ChatBrowserUse())

    try:
        history = await agent.run()

        if history and hasattr(history, "action_names") and history.action_names():
            steps = len(history.action_names())

        result_text = (
            history.final_result()
            if hasattr(history, "final_result")
            else "No output captured"
        )

        # Extract token usage
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
                print(f"  Warning: Could not get tokens from history: {e}")

        if browseruse_tokens == 0:
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
                print(f"  Warning: Could not get tokens from agent service: {e}")

        success = True

    except Exception as e:
        print(f"  Error during shopping: {str(e)}")
        result_text = f"Shopping failed: {str(e)}"

    duration = time.time() - start_time
    correct_items, accuracy_pct = (
        calculate_accuracy(result_text) if success else (0, 0.0)
    )

    return RunResult(
        run_number=run_number,
        success=success,
        result_text=str(result_text),
        steps=steps,
        tokens=browseruse_tokens,
        correct_items=correct_items,
        accuracy_pct=accuracy_pct,
        duration_seconds=duration,
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Run baseline grocery shopping automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                  # Single run
  %(prog)s --runs 3         # Run 3 times
  %(prog)s --runs 5 --no-pause  # Run 5 times without pausing between runs
        """,
    )
    parser.add_argument(
        "--runs",
        "-n",
        type=int,
        default=1,
        help="Number of runs to execute (default: 1)",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Don't pause between runs or at the end",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Save results to JSON file",
    )

    args = parser.parse_args()

    print(f"Baseline Grocery Shopping - {args.runs} run(s)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results: List[RunResult] = []

    for i in range(1, args.runs + 1):
        result = await run_grocery_shopping(i)
        results.append(result)

        # Print individual run summary
        print_run_summary(result)

        # Pause between runs (unless last run or --no-pause)
        if i < args.runs and not args.no_pause:
            print(f"\n  Pausing before next run... (Press Ctrl+C to stop)")
            try:
                await asyncio.sleep(3)
            except KeyboardInterrupt:
                print("\n  Stopping early...")
                break

    # Print aggregate summary
    print_aggregate_summary(results)

    # Save results to JSON if requested
    if args.output:
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_runs": len(results),
                "successful_runs": sum(1 for r in results if r.success),
            },
            "runs": [
                {
                    "run_number": r.run_number,
                    "success": r.success,
                    "steps": r.steps,
                    "tokens": r.tokens,
                    "correct_items": r.correct_items,
                    "accuracy_pct": r.accuracy_pct,
                    "duration_seconds": r.duration_seconds,
                    "timestamp": r.timestamp,
                }
                for r in results
            ],
            "aggregate": {
                "total_steps": sum(r.steps for r in results),
                "total_tokens": sum(r.tokens for r in results),
                "total_duration": sum(r.duration_seconds for r in results),
                "avg_accuracy": (
                    sum(r.accuracy_pct for r in results) / len(results)
                    if results
                    else 0
                ),
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    if not args.no_pause:
        input("\nPress Enter to close the browser...")


if __name__ == "__main__":
    asyncio.run(main())
