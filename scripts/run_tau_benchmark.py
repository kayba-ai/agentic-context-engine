#!/usr/bin/env python3
"""
Run TAU-bench (τ²-bench) evaluation with ACE framework.

TAU-bench evaluates tool-calling agents in customer service domains
(airline, retail, telecom) using multi-turn conversations and database
state assertions.

Key metrics:
- pass^k: Run each task k times, pass only if ALL k succeed (consistency)
- ACE epochs: Train skillbook on subset before evaluation

Workflow: Train on subset (ACE epochs) → Evaluate on test set with pass^k (frozen skillbook)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
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

from ace import (
    Reflector,
    SkillManager,
    Skillbook,
    AgentOutput,
)
from ace.llm_providers import LiteLLMClient

# Suppress LiteLLM debug messages
import litellm

litellm.suppress_debug_info = True

# TAU2 imports
from tau2.agent.llm_agent import LLMAgent
from tau2.metrics.agent_metrics import pass_hat_k
from tau2.registry import registry
from tau2.run import run_task


class ACELLMAgent(LLMAgent):
    """LLMAgent with ACE skillbook injection into the system prompt."""

    # Class-level skillbook (set before each run)
    _skillbook: Optional[Skillbook] = None

    @classmethod
    def set_skillbook(cls, skillbook: Optional[Skillbook]):
        """Set the skillbook to inject into the system prompt."""
        cls._skillbook = skillbook

    @property
    def system_prompt(self) -> str:
        """Return system prompt with skillbook strategies appended."""
        base_prompt = super().system_prompt

        if self._skillbook and len(self._skillbook.skills()) > 0:
            skillbook_section = f"""

<learned_strategies>
## Learned Strategies (from previous tasks)
{self._skillbook.as_prompt()}

Use these strategies when applicable. Cite skill IDs in your reasoning.
</learned_strategies>
"""
            return base_prompt + skillbook_section

        return base_prompt


# Register the custom agent with tau2's registry
try:
    registry.register_agent(ACELLMAgent, "ace_llm_agent")
except ValueError:
    # Already registered (e.g., when running multiple times in same process)
    pass


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Domain configuration
    parser.add_argument(
        "--domain",
        choices=["airline", "retail", "telecom", "all"],
        default="airline",
        help="Domain to evaluate (default: airline)",
    )
    parser.add_argument(
        "--task-split",
        choices=["base", "human", "gpt4o"],
        default="base",
        help="Task split to use (default: base)",
    )

    # Data configuration
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of tasks to evaluate (default: all)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train/test split ratio (default: 0.8 = 80%% train, 20%% test)",
    )

    # Pass^k configuration
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=1,
        help="K value for pass^k metric (default: 1)",
    )

    # ACE configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of ACE training epochs (default: 1)",
    )
    parser.add_argument(
        "--max-refinement-rounds",
        type=int,
        default=3,
        help="Maximum refinement rounds per sample (default: 3)",
    )
    parser.add_argument(
        "--skip-ace",
        action="store_true",
        help="Skip ACE training, run baseline only",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both baseline and ACE, then compare results",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Agent model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--user-llm",
        default="gpt-4o-mini",
        help="User simulator model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)",
    )

    # Output configuration
    parser.add_argument(
        "--output",
        default="tau_benchmark_results",
        help="Output directory for results (default: tau_benchmark_results)",
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


def create_llm_client(
    args: argparse.Namespace, model: Optional[str] = None
) -> LiteLLMClient:
    """Create LLM client with specified configuration."""
    return LiteLLMClient(
        model=model or args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=120,
    )


def load_tau_tasks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Load TAU-bench tasks for the specified domain."""
    try:
        from benchmarks.loaders.tau2 import Tau2Loader
    except ImportError:
        print("Error: tau2-bench is not installed.")
        print(
            "Install with: pip install tau2-bench or pip install ace-framework[tau-bench]"
        )
        sys.exit(1)

    loader = Tau2Loader()
    domains = (
        ["airline", "retail", "telecom"] if args.domain == "all" else [args.domain]
    )

    all_tasks = []
    for domain in domains:
        if not args.quiet:
            print(f"Loading {domain} tasks (split: {args.task_split})...")

        tasks = list(
            loader.load(
                domain=domain,
                task_split=args.task_split,
                limit=args.limit,
                user_llm=args.user_llm,
            )
        )
        all_tasks.extend(tasks)

        if not args.quiet:
            print(f"  Loaded {len(tasks)} tasks from {domain}")

    return all_tasks


def split_data(tasks: List[Dict[str, Any]], split_ratio: float) -> tuple:
    """Split tasks into train and test sets."""
    if split_ratio >= 1.0:
        return tasks, []

    split_idx = int(len(tasks) * split_ratio)
    return tasks[:split_idx], tasks[split_idx:]


def run_single_task(
    task: Dict[str, Any],
    skillbook: Skillbook,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """
    Run a single TAU task using tau2's run_task with skillbook injection.

    This uses tau2's proper tool-calling LLMAgent (via ACELLMAgent subclass)
    instead of ACE's simple text-based Agent.
    """
    # Set skillbook on the custom agent class before running
    ACELLMAgent.set_skillbook(skillbook)

    try:
        # Run with our custom agent that has skillbook injected
        simulation = run_task(
            domain=task["domain"],
            task=task["task"],  # The actual tau2 Task object
            agent="ace_llm_agent",  # Our registered custom agent
            user="user_simulator",
            llm_agent=args.model,
            llm_args_agent={"temperature": args.temperature},
            llm_user=args.user_llm,
            llm_args_user={"temperature": 0.0},
            max_steps=30,
        )

        reward = simulation.reward_info.reward if simulation.reward_info else 0.0
        return {
            "task_id": task["task_id"],
            "domain": task["domain"],
            "reward": reward,
            "success": reward >= 1.0,
            "steps": len(simulation.messages) if simulation.messages else 0,
            "cost": getattr(simulation, "agent_cost", None),
        }
    except Exception as e:
        return {
            "task_id": task["task_id"],
            "domain": task["domain"],
            "reward": 0.0,
            "success": False,
            "steps": 0,
            "cost": None,
            "error": str(e),
        }


def evaluate_pass_k(
    tasks: List[Dict[str, Any]],
    skillbook: Skillbook,
    args: argparse.Namespace,
    k: int = 1,
    quiet: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate tasks using pass^k metric per TAU-bench paper (arXiv:2406.12045).

    pass^k = average of C(successes, k) / C(trials, k) across all tasks.

    This is a combinatorial probability: the chance that all k randomly
    selected trials from the pool would be successes.
    """
    results = []
    pass_sums = {i: 0.0 for i in range(1, k + 1)}  # Accumulate pass^1, ..., pass^k

    for i, task in enumerate(tasks):
        if not quiet:
            print(
                f"  Task {i + 1}/{len(tasks)}: {task['task_id']}", end=" ", flush=True
            )

        # Run k trials
        trial_results = []
        for _ in range(k):
            trial_result = run_single_task(task, skillbook, args)
            trial_results.append(trial_result)

        # Record final pass^k for this task
        task_passed_all = all(tr["success"] for tr in trial_results)

        # Compute pass^j for each j using combinatorial formula
        num_successes = sum(1 for tr in trial_results if tr["success"])
        task_pass_k = {}
        for j in range(1, k + 1):
            task_pass_k[j] = pass_hat_k(k, num_successes, j)

        results.append(
            {
                "task_id": task["task_id"],
                "domain": task["domain"],
                "trials": trial_results,
                "passed_all": task_passed_all,
                "pass_k_values": task_pass_k,
            }
        )

        # Accumulate for averaging
        for j in range(1, k + 1):
            pass_sums[j] += task_pass_k[j]

        if not quiet:
            status = "✓" if task_passed_all else "✗"
            reward = trial_results[0]["reward"] if trial_results else 0.0
            print(f"{status} (reward={reward:.2f}, pass^k={task_pass_k})")

    # Average pass^k across all tasks
    n_tasks = len(tasks)
    metrics = {}
    for j in range(1, k + 1):
        metrics[f"pass_{j}"] = pass_sums[j] / n_tasks if n_tasks > 0 else 0.0

    return {
        "tasks_evaluated": n_tasks,
        "k": k,
        "pass_sums": pass_sums,
        "metrics": metrics,
        "results": results,
    }


def run_ace_training(
    train_tasks: List[Dict[str, Any]],
    args: argparse.Namespace,
    quiet: bool = False,
) -> Skillbook:
    """
    Run ACE training on train tasks.

    Uses tau2's run_task with our ACELLMAgent to execute tasks,
    then learns from the results using ACE's Reflector and SkillManager.
    """
    if not quiet:
        print(
            f"\n📚 ACE Training Phase ({len(train_tasks)} tasks × {args.epochs} epochs)"
        )

    client = create_llm_client(args)
    reflector = Reflector(client)
    skill_manager = SkillManager(client)
    skillbook = Skillbook()

    # Run adaptation with conversation-based learning
    for epoch in range(1, args.epochs + 1):
        if not quiet:
            print(f"  Epoch {epoch}/{args.epochs}")

        for i, task in enumerate(train_tasks):
            try:
                # Run task with current skillbook using tau2's proper tool-calling agent
                result = run_single_task(task, skillbook, args)

                # Create feedback for Reflector based on task outcome
                if result["success"]:
                    feedback = f"Task SUCCEEDED. Reward: {result['reward']:.2f}, Steps: {result['steps']}"
                else:
                    feedback = f"Task FAILED. Reward: {result['reward']:.2f}, Steps: {result['steps']}"
                    if "error" in result:
                        feedback += f". Error: {result['error']}"

                # Create agent output representation for ACE learning
                agent_output = AgentOutput(
                    final_answer=f"reward={result['reward']:.2f}",
                    reasoning="Multi-turn tool-calling conversation",
                    skill_ids=[],
                )

                # Learn from result
                reflection = reflector.reflect(
                    question=task["instruction"],
                    agent_output=agent_output,
                    skillbook=skillbook,
                    ground_truth=None,
                    feedback=feedback,
                )

                skill_manager_output = skill_manager.update_skills(
                    reflection=reflection,
                    skillbook=skillbook,
                    question_context=task["instruction"],
                    progress=f"epoch {epoch}/{args.epochs} · task {i + 1}/{len(train_tasks)}",
                )

                skillbook.apply_update(skill_manager_output.update)

                if not quiet:
                    status = "✓" if result["success"] else "✗"
                    print(
                        f"    [{i + 1}/{len(train_tasks)}] {task['task_id']} {status} (reward={result['reward']:.2f})"
                    )

            except Exception as e:
                if not quiet:
                    print(
                        f"    [{i + 1}/{len(train_tasks)}] {task['task_id']} ERROR: {e}"
                    )
                continue

    if not quiet:
        print(f"  Training complete. Skillbook has {len(skillbook.skills())} skills")

    return skillbook


def run_evaluation(
    args: argparse.Namespace,
    tasks: List[Dict[str, Any]],
    skillbook: Skillbook,
    phase_name: str = "Evaluation",
) -> Dict[str, Any]:
    """Run pass^k evaluation on tasks."""
    if not args.quiet:
        print(f"\n🧪 {phase_name} Phase (k={args.k})")

    # Run pass^k evaluation using tau2's run_task
    eval_results = evaluate_pass_k(
        tasks=tasks,
        skillbook=skillbook,
        args=args,
        k=args.k,
        quiet=args.quiet,
    )

    return eval_results


def print_results(
    results: Dict[str, Any],
    title: str,
    args: argparse.Namespace,
) -> None:
    """Print evaluation results summary."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"Domain: {args.domain}")
    print(f"Tasks evaluated: {results['tasks_evaluated']}")
    print(f"K value: {results['k']}")
    print()
    print("Pass^k Metrics (TAU-bench formula: C(successes,k)/C(trials,k)):")
    for j in range(1, results["k"] + 1):
        metric = results["metrics"][f"pass_{j}"]
        print(f"  pass^{j}: {metric:.2%}")
    print("=" * 60)


def save_results(
    args: argparse.Namespace,
    results: Dict[str, Any],
    skillbook: Skillbook,
    phase: str,
) -> None:
    """Save evaluation results to files."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = f"tau_{args.domain}_{args.model}_{phase}_{timestamp}"

    # Save summary
    summary_file = output_dir / f"{base_name}_summary.json"
    summary = {
        "benchmark": "tau_bench",
        "domain": args.domain,
        "task_split": args.task_split,
        "model": args.model,
        "user_llm": args.user_llm,
        "phase": phase,
        "timestamp": timestamp,
        "configuration": {
            "k": args.k,
            "epochs": args.epochs,
            "split_ratio": args.split_ratio,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        },
        "results": {
            "tasks_evaluated": results["tasks_evaluated"],
            "pass_sums": results["pass_sums"],
            "metrics": results["metrics"],
        },
        "skillbook_stats": skillbook.stats() if skillbook else {},
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    if not args.quiet:
        print(f"\n💾 Results saved to: {summary_file}")

    # Save detailed results if requested
    if args.save_detailed:
        detailed_file = output_dir / f"{base_name}_detailed.json"
        with open(detailed_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        if not args.quiet:
            print(f"   Detailed results: {detailed_file}")

    # Save skillbook
    if skillbook and len(skillbook.skills()) > 0:
        skillbook_file = output_dir / f"{base_name}_skillbook.json"
        skillbook.save_to_file(str(skillbook_file))

        if not args.quiet:
            print(f"   Skillbook: {skillbook_file}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    if not args.quiet:
        print("🚀 TAU-bench Evaluation")
        print(f"   Domain: {args.domain}")
        print(f"   Model: {args.model}")
        print(f"   K: {args.k}")
        if not args.skip_ace:
            print(f"   ACE epochs: {args.epochs}")
            print(f"   Split ratio: {args.split_ratio}")

    # Load tasks
    tasks = load_tau_tasks(args)
    if not tasks:
        print("Error: No tasks loaded")
        sys.exit(1)

    if not args.quiet:
        print(f"\n📊 Loaded {len(tasks)} tasks")

    if args.compare:
        # Comparison mode: baseline vs ACE
        train_tasks, test_tasks = split_data(tasks, args.split_ratio)

        if not args.quiet:
            print(f"   Train: {len(train_tasks)}, Test: {len(test_tasks)}")

        # Run baseline (empty skillbook)
        print("\n" + "=" * 60)
        print("  1️⃣  BASELINE (no ACE)")
        print("=" * 60)
        baseline_skillbook = Skillbook()
        baseline_results = run_evaluation(
            args, test_tasks, baseline_skillbook, "Baseline"
        )
        print_results(baseline_results, "BASELINE Results", args)

        # Run ACE training + evaluation
        print("\n" + "=" * 60)
        print("  2️⃣  ACE (with training)")
        print("=" * 60)
        ace_skillbook = run_ace_training(train_tasks, args, args.quiet)
        ace_results = run_evaluation(args, test_tasks, ace_skillbook, "ACE Test")
        print_results(ace_results, "ACE Results", args)

        # Compare
        print("\n" + "=" * 60)
        print("  📊 COMPARISON")
        print("=" * 60)
        for j in range(1, args.k + 1):
            baseline_metric = baseline_results["metrics"][f"pass_{j}"]
            ace_metric = ace_results["metrics"][f"pass_{j}"]
            diff = ace_metric - baseline_metric
            indicator = "✅" if diff > 0 else ("⚠️" if diff < 0 else "➖")
            print(
                f"  pass^{j}: {baseline_metric:.2%} → {ace_metric:.2%} ({diff:+.2%}) {indicator}"
            )

        print(f"\n  Skills learned: {len(ace_skillbook.skills())}")
        print("=" * 60)

        # Save both results
        save_results(args, baseline_results, baseline_skillbook, "baseline")
        save_results(args, ace_results, ace_skillbook, "ace")

    elif args.skip_ace:
        # Baseline only
        baseline_skillbook = Skillbook()
        results = run_evaluation(args, tasks, baseline_skillbook, "Baseline")
        print_results(results, "BASELINE Results", args)
        save_results(args, results, baseline_skillbook, "baseline")

    else:
        # ACE training + evaluation
        train_tasks, test_tasks = split_data(tasks, args.split_ratio)

        if not args.quiet:
            print(f"   Train: {len(train_tasks)}, Test: {len(test_tasks)}")

        # Train
        skillbook = run_ace_training(train_tasks, args, args.quiet)

        # Evaluate on test set (frozen skillbook)
        results = run_evaluation(args, test_tasks, skillbook, "Test")
        print_results(results, "TAU-bench Results (ACE)", args)
        save_results(args, results, skillbook, "ace")

    if not args.quiet:
        print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
