#!/usr/bin/env python3
"""
Run SWE-bench benchmarks using ACEMiniSWE (mini-swe-agent + ACE learning).

This script evaluates mini-swe-agent on real SWE-bench Lite tasks from HuggingFace,
comparing baseline performance vs ACE-enhanced performance.

Usage:
    # Baseline (no ACE learning)
    uv run python benchmarks/run_mini_swe_benchmark.py --limit 10 --skip-adaptation

    # ACE with deduplication
    uv run python benchmarks/run_mini_swe_benchmark.py --limit 10 --dedup

    # Compare baseline vs ACE
    uv run python benchmarks/run_mini_swe_benchmark.py --limit 10 --compare

    # Full benchmark with custom model
    uv run python benchmarks/run_mini_swe_benchmark.py \
        --model claude-opus-4-20250514 \
        --limit 10 \
        --dedup \
        --save-skillbook benchmark_results/swe_bench_opus.json

Requirements:
    - mini-swe-agent: uv add mini-swe-agent --group demos
    - API key: ANTHROPIC_API_KEY or OPENAI_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from ace.integrations import ACEMiniSWE, MINI_SWE_AVAILABLE
from ace.deduplication import DeduplicationConfig
from ace.skillbook import Skillbook

# Suppress verbose logging
import litellm

litellm.suppress_debug_info = True


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model for mini-swe-agent execution (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--ace-model",
        default="gpt-4o-mini",
        help="Model for ACE learning (Reflector/SkillManager) (default: gpt-4o-mini)",
    )

    # Data configuration
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of SWE-bench tasks to run (default: 10)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip first N tasks (default: 0)",
    )

    # ACE configuration
    parser.add_argument(
        "--skip-adaptation",
        action="store_true",
        help="Run baseline without ACE learning",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Enable skill deduplication",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.85,
        help="Deduplication similarity threshold (default: 0.85)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both baseline and ACE, then compare results",
    )

    # Agent configuration
    parser.add_argument(
        "--step-limit",
        type=int,
        default=30,
        help="Maximum agent steps per task (default: 30)",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=3.0,
        help="Maximum cost per task in USD (default: 3.0)",
    )

    # Output configuration
    parser.add_argument(
        "--output",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)",
    )
    parser.add_argument(
        "--save-skillbook",
        type=str,
        help="Path to save learned skillbook after benchmark",
    )
    parser.add_argument(
        "--load-skillbook",
        type=str,
        help="Path to load existing skillbook before benchmark",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="Save detailed per-task results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    return parser.parse_args()


def load_swe_bench_tasks(limit: int, offset: int = 0) -> List[Dict[str, Any]]:
    """Load SWE-bench Lite tasks from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed.")
        print("Install with: uv add datasets")
        sys.exit(1)

    print(f"Loading SWE-bench Lite dataset from HuggingFace...")
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")

    # Apply offset and limit
    tasks = []
    for i, item in enumerate(dataset):
        if i < offset:
            continue
        if len(tasks) >= limit:
            break
        tasks.append(dict(item))

    print(f"Loaded {len(tasks)} tasks (offset={offset}, limit={limit})")
    return tasks


def format_swe_task(task: Dict[str, Any]) -> str:
    """Format SWE-bench task as a task description for mini-swe-agent."""
    return f"""Repository: {task['repo']}

Issue: {task['problem_statement']}

Base Commit: {task['base_commit']}

Please investigate and fix this issue. The task is complete when you have:
1. Understood the issue from the problem statement
2. Located the relevant code
3. Implemented a fix
4. Verified the fix addresses the issue"""


def run_single_task(
    agent: ACEMiniSWE,
    task: Dict[str, Any],
    task_idx: int,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Run a single SWE-bench task and return results."""
    task_id = task.get("instance_id", f"task_{task_idx}")
    repo = task.get("repo", "unknown")

    if not quiet:
        print(f"\n{'='*60}")
        print(f"Task {task_idx + 1}: {task_id}")
        print(f"Repository: {repo}")
        print("=" * 60)

    task_description = format_swe_task(task)

    start_time = time.time()
    try:
        # Create a temp directory for this task
        # In a real SWE-bench setup, you'd clone the repo at base_commit
        # For now, we simulate with a temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            result = agent.run(task=task_description, working_dir=tmpdir)

        elapsed = time.time() - start_time

        result_data = {
            "task_id": task_id,
            "repo": repo,
            "status": result.status,
            "success": result.success,
            "steps": result.steps,
            "elapsed_seconds": round(elapsed, 2),
            "message": result.message[:500] if result.message else "",
            "error": None,
        }

        if not quiet:
            status_emoji = "✅" if result.success else "❌"
            print(f"\nResult: {status_emoji} {result.status}")
            print(f"Steps: {result.steps}, Time: {elapsed:.1f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        result_data = {
            "task_id": task_id,
            "repo": repo,
            "status": "Error",
            "success": False,
            "steps": 0,
            "elapsed_seconds": round(elapsed, 2),
            "message": "",
            "error": str(e),
        }

        if not quiet:
            print(f"\n❌ Error: {e}")

    return result_data


def run_benchmark(
    args: argparse.Namespace,
    tasks: List[Dict[str, Any]],
    is_learning: bool = True,
    skillbook: Optional[Skillbook] = None,
) -> Dict[str, Any]:
    """Run benchmark on all tasks and return results."""
    # Create dedup config if requested
    dedup_config = None
    if args.dedup and is_learning:
        dedup_config = DeduplicationConfig(
            similarity_threshold=args.dedup_threshold,
        )

    # Create agent
    agent = ACEMiniSWE(
        model=args.model,
        ace_model=args.ace_model,
        skillbook=skillbook,
        is_learning=is_learning,
        dedup_config=dedup_config,
        step_limit=args.step_limit,
        cost_limit=args.cost_limit,
    )

    mode = "ACE" if is_learning else "Baseline"
    if not args.quiet:
        print(f"\n{'#'*60}")
        print(f"Running {mode} benchmark on {len(tasks)} tasks")
        print(f"Model: {args.model}")
        print(f"ACE Model: {args.ace_model}")
        print(f"Learning: {is_learning}")
        print(f"Dedup: {args.dedup and is_learning}")
        print("#" * 60)

    results = []
    start_time = time.time()

    for idx, task in enumerate(tasks):
        if not args.quiet:
            print(f"\nProgress: {idx + 1}/{len(tasks)}")

        result = run_single_task(agent, task, idx, args.quiet)
        results.append(result)

        # Show cumulative stats
        successes = sum(1 for r in results if r["success"])
        if not args.quiet:
            print(
                f"Cumulative: {successes}/{len(results)} successful ({100*successes/len(results):.1f}%)"
            )

    total_elapsed = time.time() - start_time

    # Compute summary statistics
    successes = sum(1 for r in results if r["success"])
    errors = sum(1 for r in results if r["error"])
    avg_steps = (
        mean([r["steps"] for r in results if r["steps"] > 0])
        if any(r["steps"] > 0 for r in results)
        else 0
    )

    summary = {
        "mode": mode,
        "model": args.model,
        "ace_model": args.ace_model,
        "is_learning": is_learning,
        "dedup": args.dedup and is_learning,
        "total_tasks": len(tasks),
        "successes": successes,
        "failures": len(tasks) - successes - errors,
        "errors": errors,
        "success_rate": successes / len(tasks) if tasks else 0,
        "avg_steps": round(avg_steps, 1),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "avg_elapsed_seconds": round(total_elapsed / len(tasks), 2) if tasks else 0,
        "skills_learned": len(agent.skillbook.skills()) if agent.skillbook else 0,
    }

    return {
        "summary": summary,
        "results": results,
        "skillbook": agent.skillbook,
    }


def run_comparison(
    args: argparse.Namespace,
    tasks: List[Dict[str, Any]],
) -> None:
    """Run baseline vs ACE comparison."""
    print("\n" + "=" * 80)
    print("RUNNING COMPARISON: Baseline vs ACE")
    print("=" * 80)

    # Run baseline
    print("\n1️⃣  Running BASELINE (no learning)...")
    baseline = run_benchmark(args, tasks, is_learning=False)

    # Run ACE
    print("\n2️⃣  Running ACE (with learning)...")
    ace = run_benchmark(args, tasks, is_learning=True)

    # Save skillbook if requested
    if args.save_skillbook:
        skillbook = ace["skillbook"]
        skillbook_path = Path(args.save_skillbook)
        skillbook_path.parent.mkdir(parents=True, exist_ok=True)
        skillbook.save_to_file(str(skillbook_path))
        print(f"\n📚 Skillbook saved to: {skillbook_path}")

    # Compare results
    print("\n" + "=" * 80)
    print("📊 COMPARISON RESULTS")
    print("=" * 80)

    baseline_summary = baseline["summary"]
    ace_summary = ace["summary"]

    print(f"\n🔬 BASELINE:")
    print(f"  Success Rate: {baseline_summary['success_rate']:.1%}")
    print(f"  Avg Steps: {baseline_summary['avg_steps']}")
    print(f"  Total Time: {baseline_summary['total_elapsed_seconds']:.1f}s")

    print(f"\n🧠 ACE:")
    print(f"  Success Rate: {ace_summary['success_rate']:.1%}")
    print(f"  Avg Steps: {ace_summary['avg_steps']}")
    print(f"  Total Time: {ace_summary['total_elapsed_seconds']:.1f}s")
    print(f"  Skills Learned: {ace_summary['skills_learned']}")

    # Improvement
    rate_diff = ace_summary["success_rate"] - baseline_summary["success_rate"]
    if rate_diff > 0:
        print(f"\n✅ ACE improved success rate by +{rate_diff:.1%}")
    elif rate_diff < 0:
        print(f"\n⚠️  ACE decreased success rate by {rate_diff:.1%}")
    else:
        print(f"\n➡️  No change in success rate")

    # Save comparison results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    comparison_file = output_dir / f"mini_swe_comparison_{args.model}_{timestamp}.json"

    # Remove non-serializable skillbook before saving
    baseline.pop("skillbook", None)
    ace.pop("skillbook", None)

    comparison_data = {
        "benchmark": "swe_bench_lite",
        "model": args.model,
        "ace_model": args.ace_model,
        "timestamp": timestamp,
        "tasks_count": len(tasks),
        "configuration": {
            "step_limit": args.step_limit,
            "cost_limit": args.cost_limit,
            "dedup": args.dedup,
            "dedup_threshold": args.dedup_threshold,
        },
        "baseline": baseline,
        "ace": ace,
        "improvement": {
            "success_rate_diff": rate_diff,
            "time_ratio": (
                ace_summary["total_elapsed_seconds"]
                / baseline_summary["total_elapsed_seconds"]
                if baseline_summary["total_elapsed_seconds"] > 0
                else 0
            ),
        },
    }

    with open(comparison_file, "w") as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {comparison_file}")


def main():
    """Main entry point."""
    args = parse_args()

    # Check dependencies
    if not MINI_SWE_AVAILABLE:
        print("Error: mini-swe-agent not installed.")
        print("Install with: uv add mini-swe-agent --group demos")
        sys.exit(1)

    # Check API keys
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Error: No API key found.")
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable.")
        sys.exit(1)

    # Load tasks
    tasks = load_swe_bench_tasks(args.limit, args.offset)

    if not tasks:
        print("Error: No tasks loaded.")
        sys.exit(1)

    # Run benchmark
    if args.compare:
        run_comparison(args, tasks)
    else:
        # Single mode run
        is_learning = not args.skip_adaptation

        # Load existing skillbook if provided
        skillbook = None
        if args.load_skillbook:
            skillbook = Skillbook.load_from_file(args.load_skillbook)
            print(f"Loaded skillbook from: {args.load_skillbook}")
            print(f"Starting with {len(skillbook.skills())} strategies")

        result = run_benchmark(
            args, tasks, is_learning=is_learning, skillbook=skillbook
        )

        # Save skillbook
        if args.save_skillbook and is_learning:
            skillbook = result["skillbook"]
            skillbook_path = Path(args.save_skillbook)
            skillbook_path.parent.mkdir(parents=True, exist_ok=True)
            skillbook.save_to_file(str(skillbook_path))
            print(f"\n📚 Skillbook saved to: {skillbook_path}")

        # Print summary
        summary = result["summary"]
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Mode: {summary['mode']}")
        print(f"Model: {summary['model']}")
        print(f"Tasks: {summary['total_tasks']}")
        print(f"Successes: {summary['successes']}")
        print(f"Failures: {summary['failures']}")
        print(f"Errors: {summary['errors']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Avg Steps: {summary['avg_steps']}")
        print(f"Total Time: {summary['total_elapsed_seconds']:.1f}s")
        if is_learning:
            print(f"Skills Learned: {summary['skills_learned']}")

        # Save detailed results
        if args.save_detailed:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            mode_str = "ace" if is_learning else "baseline"
            results_file = (
                output_dir / f"mini_swe_{mode_str}_{args.model}_{timestamp}.json"
            )

            # Remove non-serializable skillbook
            result.pop("skillbook", None)

            with open(results_file, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            print(f"\n💾 Detailed results saved to: {results_file}")

    print("\n✅ Benchmark completed!")


if __name__ == "__main__":
    main()
