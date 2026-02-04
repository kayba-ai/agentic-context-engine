#!/usr/bin/env python3
"""
ACE + Browser-Use Grocery Shopping Demo

Run grocery shopping automation with ACE learning, multiple times with aggregated results.

Usage:
    uv run python examples/browser-use/online-shopping/ace-online-shopping.py
    uv run python examples/browser-use/online-shopping/ace-online-shopping.py --runs 3
    uv run python examples/browser-use/online-shopping/ace-online-shopping.py --runs 5 --no-pause -o results.json
"""

import argparse
import asyncio
import datetime
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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


# Shopping task definition
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
    browseruse_tokens: int
    ace_tokens: int
    correct_items: int
    accuracy_pct: float
    duration_seconds: float
    skills_count: int
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())


def get_ace_token_usage(
    run_start_time: datetime.datetime = None,
    verbose: bool = False,
) -> tuple[int, int, int, int]:
    """Query Opik for ACE token usage only.

    Returns:
        tuple: (ace_tokens, agent_tokens, reflector_tokens, skill_manager_tokens)
    """
    try:
        if not opik:
            if verbose:
                print("   Opik not available for token tracking")
            return 0, 0, 0, 0

        # Create client and flush to ensure data is sent
        client = opik.Opik()
        client.flush()

        if verbose:
            print(f"   Querying Opik for ACE token usage...")

        # Use run start time if available, otherwise fall back to last 10 minutes
        if run_start_time:
            recent_time = run_start_time.isoformat().replace("+00:00", "Z")
        else:
            now = datetime.datetime.now(datetime.timezone.utc)
            recent_time = (
                (now - datetime.timedelta(minutes=10))
                .isoformat()
                .replace("+00:00", "Z")
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
                all_traces.extend(traces)
            except Exception:
                pass

        # Track individual ACE role tokens
        agent_tokens = 0
        reflector_tokens = 0
        skill_manager_tokens = 0

        for trace in all_traces:
            trace_name = getattr(trace, "name", "unknown")
            trace_name_lower = trace_name.lower()

            if any(
                role in trace_name_lower
                for role in ["agent", "reflector", "skill_manager"]
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
                    except Exception:
                        pass

                if "agent" in trace_name_lower:
                    agent_tokens += total_tokens
                elif "reflector" in trace_name_lower:
                    reflector_tokens += total_tokens
                elif "skill_manager" in trace_name_lower:
                    skill_manager_tokens += total_tokens

        ace_tokens = agent_tokens + reflector_tokens + skill_manager_tokens
        return (ace_tokens, agent_tokens, reflector_tokens, skill_manager_tokens)

    except Exception:
        return 0, 0, 0, 0


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
    if "M-Budget" in result_text and "eggs" in result_text.lower() and "4.25" in result_text:
        correct_items += 1

    # Bananas: Fresca @ 2.55
    if "Fresca" in result_text and ("bananas" in result_text.lower()) and "2.55" in result_text:
        correct_items += 1

    # Butter: M-Budget @ 7.00
    if "M-Budget" in result_text and "butter" in result_text.lower() and "7." in result_text:
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
    print(f"  Browser-use tokens: {run_result.browseruse_tokens:,}")
    print(f"  ACE tokens: {run_result.ace_tokens:,}")
    print(f"  Total tokens: {run_result.browseruse_tokens + run_result.ace_tokens:,}")
    print(f"  Accuracy: {run_result.correct_items}/5 ({run_result.accuracy_pct:.0f}%)")
    print(f"  Skills in skillbook: {run_result.skills_count}")

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
    total_browseruse_tokens = sum(r.browseruse_tokens for r in results)
    total_ace_tokens = sum(r.ace_tokens for r in results)
    total_tokens = total_browseruse_tokens + total_ace_tokens
    total_duration = sum(r.duration_seconds for r in results)
    total_correct = sum(r.correct_items for r in results)
    max_possible_correct = total_runs * 5

    avg_steps = total_steps / total_runs if total_runs > 0 else 0
    avg_browseruse_tokens = total_browseruse_tokens / total_runs if total_runs > 0 else 0
    avg_ace_tokens = total_ace_tokens / total_runs if total_runs > 0 else 0
    avg_duration = total_duration / total_runs if total_runs > 0 else 0
    avg_accuracy = (total_correct / max_possible_correct) * 100 if max_possible_correct > 0 else 0

    # Per-run accuracy stats
    accuracies = [r.accuracy_pct for r in results]
    min_accuracy = min(accuracies) if accuracies else 0
    max_accuracy = max(accuracies) if accuracies else 0

    # Skills growth
    first_skills = results[0].skills_count if results else 0
    final_skills = results[-1].skills_count if results else 0

    print(f"\n{'=' * 70}")
    print(f"AGGREGATE SUMMARY - {total_runs} RUNS (ACE)")
    print(f"{'=' * 70}")

    print(f"\n  Run Statistics:")
    print(f"    Successful: {successful_runs}/{total_runs} ({(successful_runs/total_runs)*100:.0f}%)")
    print(f"    Failed: {failed_runs}/{total_runs}")

    print(f"\n  Performance Metrics:")
    print(f"    Total Duration: {total_duration:.1f}s")
    print(f"    Avg Duration: {avg_duration:.1f}s per run")
    print(f"    Total Steps: {total_steps}")
    print(f"    Avg Steps: {avg_steps:.1f} per run")

    print(f"\n  Token Usage:")
    print(f"    Browser-use: {total_browseruse_tokens:,} total ({avg_browseruse_tokens:,.0f} avg)")
    print(f"    ACE Learning: {total_ace_tokens:,} total ({avg_ace_tokens:,.0f} avg)")
    print(f"    Combined: {total_tokens:,} total")

    print(f"\n  Accuracy:")
    print(f"    Total Correct Items: {total_correct}/{max_possible_correct}")
    print(f"    Overall Accuracy: {avg_accuracy:.1f}%")
    print(f"    Min Run Accuracy: {min_accuracy:.0f}%")
    print(f"    Max Run Accuracy: {max_accuracy:.0f}%")

    print(f"\n  ACE Learning:")
    print(f"    Starting Skills: {first_skills}")
    print(f"    Final Skills: {final_skills}")
    print(f"    Skills Learned: {final_skills - first_skills}")

    # Per-run breakdown table
    print(f"\n  Per-Run Breakdown:")
    print(f"    {'Run':<5} {'Status':<8} {'Duration':<10} {'Steps':<7} {'BU Tok':<10} {'ACE Tok':<10} {'Accuracy':<12} {'Skills':<6}")
    print(f"    {'-'*5} {'-'*8} {'-'*10} {'-'*7} {'-'*10} {'-'*10} {'-'*12} {'-'*6}")
    for r in results:
        status = "OK" if r.success else "FAIL"
        print(f"    {r.run_number:<5} {status:<8} {r.duration_seconds:<10.1f} {r.steps:<7} {r.browseruse_tokens:<10,} {r.ace_tokens:<10,} {r.correct_items}/5 ({r.accuracy_pct:.0f}%)    {r.skills_count:<6}")

    print(f"\n{'=' * 70}")


async def run_grocery_shopping(
    run_number: int,
    skillbook_path: Path,
    ace_model: str,
) -> RunResult:
    """Run grocery shopping with ACE and collect metrics."""
    print(f"\n{'#' * 70}")
    print(f"# STARTING RUN {run_number} (ACE)")
    print(f"{'#' * 70}")
    print(f"Testing automation with 5 essential items at Migros")

    run_start_time = datetime.datetime.now(datetime.timezone.utc)
    start_time = time.time()
    steps = 0
    browseruse_tokens = 0
    ace_tokens = 0
    result_text = ""
    success = False
    skills_count = 0

    # Create ACE agent - loads existing skillbook if available
    agent = ACEAgent(
        llm=ChatBrowserUse(),
        ace_model=ace_model,
        ace_max_tokens=4096,
        skillbook_path=str(skillbook_path) if skillbook_path.exists() else None,
        max_steps=30,
        calculate_cost=True,
    )

    skills_count = len(agent.skillbook.skills())
    print(f"  Starting with {skills_count} skills in skillbook")

    try:
        history = await agent.run(task=TASK)

        if history and hasattr(history, "number_of_steps"):
            steps = history.number_of_steps()
        elif history and hasattr(history, "action_names") and history.action_names():
            steps = len(history.action_names())

        result_text = (
            history.final_result()
            if hasattr(history, "final_result")
            else "No output captured"
        )

        # Extract browser-use token usage
        if history and hasattr(history, "usage"):
            try:
                usage = history.usage
                if usage:
                    if hasattr(usage, "total_tokens"):
                        browseruse_tokens = usage.total_tokens
                    elif isinstance(usage, dict) and "total_tokens" in usage:
                        browseruse_tokens = usage["total_tokens"]
                    elif hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
                        browseruse_tokens = usage.input_tokens + usage.output_tokens
                    elif isinstance(usage, dict) and "input_tokens" in usage and "output_tokens" in usage:
                        browseruse_tokens = usage["input_tokens"] + usage["output_tokens"]
            except Exception as e:
                print(f"  Warning: Could not get tokens from history: {e}")

        # Query ACE tokens
        time.sleep(3)  # Wait for Opik to index
        ace_tokens, _, _, _ = get_ace_token_usage(run_start_time)

        success = True

        # Save skillbook after each run
        agent.save_skillbook(str(skillbook_path))
        skills_count = len(agent.skillbook.skills())

    except Exception as e:
        print(f"  Error during shopping: {str(e)}")
        result_text = f"Shopping failed: {str(e)}"
        # Still try to save skillbook
        try:
            agent.save_skillbook(str(skillbook_path))
            skills_count = len(agent.skillbook.skills())
        except Exception:
            pass

    duration = time.time() - start_time
    correct_items, accuracy_pct = calculate_accuracy(result_text) if success else (0, 0.0)

    return RunResult(
        run_number=run_number,
        success=success,
        result_text=str(result_text),
        steps=steps,
        browseruse_tokens=browseruse_tokens,
        ace_tokens=ace_tokens,
        correct_items=correct_items,
        accuracy_pct=accuracy_pct,
        duration_seconds=duration,
        skills_count=skills_count,
    )


async def main():
    parser = argparse.ArgumentParser(
        description="Run ACE grocery shopping automation with learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                  # Single run
  %(prog)s --runs 3         # Run 3 times (skillbook persists across runs)
  %(prog)s --runs 5 --no-pause  # Run 5 times without pausing
  %(prog)s --runs 3 --fresh     # Start with empty skillbook
        """,
    )
    parser.add_argument(
        "--runs", "-n",
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
        "--output", "-o",
        type=str,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start with empty skillbook (ignore existing)",
    )
    parser.add_argument(
        "--ace-model",
        type=str,
        default="claude-haiku-4-5-20251001",
        help="Model for ACE learning (default: claude-haiku-4-5-20251001)",
    )

    args = parser.parse_args()

    # Configure observability
    try:
        configure_opik(project_name="ace-grocery-shopping")
        print("Opik observability enabled")
    except Exception:
        print("Opik not available, continuing without observability")

    print(f"\nACE Grocery Shopping - {args.runs} run(s)")
    print(f"ACE Model: {args.ace_model}")
    print(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup skillbook persistence
    skillbook_path = Path(__file__).parent / "ace_grocery_skillbook.json"

    if args.fresh and skillbook_path.exists():
        print(f"  --fresh flag: removing existing skillbook")
        skillbook_path.unlink()

    if skillbook_path.exists():
        print(f"  Loading existing skillbook from: {skillbook_path}")
    else:
        print(f"  Starting with empty skillbook")

    results: List[RunResult] = []

    for i in range(1, args.runs + 1):
        result = await run_grocery_shopping(i, skillbook_path, args.ace_model)
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

    # Show final skillbook state
    if skillbook_path.exists():
        print(f"\nSkillbook saved to: {skillbook_path}")

    # Save results to JSON if requested
    if args.output:
        output_data = {
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_runs": len(results),
                "successful_runs": sum(1 for r in results if r.success),
                "ace_model": args.ace_model,
            },
            "runs": [
                {
                    "run_number": r.run_number,
                    "success": r.success,
                    "steps": r.steps,
                    "browseruse_tokens": r.browseruse_tokens,
                    "ace_tokens": r.ace_tokens,
                    "correct_items": r.correct_items,
                    "accuracy_pct": r.accuracy_pct,
                    "duration_seconds": r.duration_seconds,
                    "skills_count": r.skills_count,
                    "timestamp": r.timestamp,
                }
                for r in results
            ],
            "aggregate": {
                "total_steps": sum(r.steps for r in results),
                "total_browseruse_tokens": sum(r.browseruse_tokens for r in results),
                "total_ace_tokens": sum(r.ace_tokens for r in results),
                "total_duration": sum(r.duration_seconds for r in results),
                "avg_accuracy": sum(r.accuracy_pct for r in results) / len(results) if results else 0,
                "skills_learned": results[-1].skills_count - results[0].skills_count if results else 0,
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to: {args.output}")

    if not args.no_pause:
        input("\nPress Enter to close the browser...")


if __name__ == "__main__":
    asyncio.run(main())
