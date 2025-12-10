#!/usr/bin/env python3
"""
Run ACE benchmarks with comprehensive evaluation and reporting.

This script provides a command-line interface for running benchmarks
with the ACE framework, supporting multiple benchmark types and
configuration options.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Dict, List, Any

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
    Agent,
    Reflector,
    SkillManager,
    OfflineACE,
    OnlineACE,
    Skillbook,
)
from ace.llm_providers import LiteLLMClient
from ace import Sample
from benchmarks import BenchmarkTaskManager

# Suppress LiteLLM debug messages
import litellm

litellm.suppress_debug_info = True


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Benchmark selection
    parser.add_argument(
        "benchmark",
        help="Benchmark name to run (finer_ord, xbrl_math, appworld, or 'list' to show available)",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model to use for evaluation (default: gpt-4o-mini)",
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

    # Data configuration
    parser.add_argument(
        "--split", default="test", help="Dataset split to evaluate (default: test)"
    )
    parser.add_argument(
        "--limit", type=int, help="Limit number of samples to evaluate (default: all)"
    )

    # ACE configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of offline adaptation epochs (default: 1)",
    )
    parser.add_argument(
        "--max-refinement-rounds",
        type=int,
        default=3,
        help="Maximum refinement rounds per sample (default: 3)",
    )
    parser.add_argument(
        "--skip-adaptation",
        action="store_true",
        help="Skip ACE adaptation and run direct evaluation",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train/test split ratio (default: 0.8 = 80%% train, 20%% test)",
    )
    parser.add_argument(
        "--online-mode",
        action="store_true",
        help="Use online learning (OnlineACE) instead of offline adaptation",
    )
    parser.add_argument(
        "--prompt-version",
        choices=["v1", "v2"],
        default="v1",
        help="Prompt version to use (default: v1)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both baseline and ACE, then compare results",
    )

    # Output configuration
    parser.add_argument(
        "--output",
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)",
    )
    parser.add_argument(
        "--save-detailed", action="store_true", help="Save detailed per-sample results"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")

    # Cache configuration
    parser.add_argument(
        "--cache-dir", help="Override cache directory for benchmark data"
    )

    return parser.parse_args()


def list_available_benchmarks() -> None:
    """List all available benchmarks."""
    manager = BenchmarkTaskManager()
    benchmarks = manager.list_benchmarks()

    print("Available benchmarks:")
    for name in benchmarks:
        try:
            config = manager.get_config(name)
            print(f"  {name} - {config.metadata.get('description', 'No description')}")
        except Exception as e:
            print(f"  {name} - (Error loading config: {e})")


def create_llm_client(args: argparse.Namespace) -> LiteLLMClient:
    """Create LLM client with specified configuration."""
    return LiteLLMClient(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=120,
    )


def load_benchmark_data(
    args: argparse.Namespace, manager: BenchmarkTaskManager
) -> List[Sample]:
    """Load and convert benchmark data to Sample format."""
    if not args.quiet:
        print(f"Loading {args.benchmark} data (split: {args.split})...")

    # Load raw data
    try:
        raw_data = list(manager.load_benchmark_data(args.benchmark))
    except Exception as e:
        print(f"Error loading benchmark data: {e}")
        sys.exit(1)

    # Apply limit if specified
    if args.limit:
        raw_data = raw_data[: args.limit]

    if not args.quiet:
        print(f"Loaded {len(raw_data)} samples")

    # Convert to Sample format
    samples = []

    for i, data in enumerate(raw_data):
        if args.benchmark == "appworld":
            # AppWorld has special handling
            sample = Sample(
                question=data["instruction"],
                context=f"Available APIs: {data['api_docs']}",
                ground_truth="Task completion successful",
            )
        elif args.benchmark == "finer_ord":
            # FiNER now comes pre-processed from the loader
            sample = Sample(
                question=data["question"],
                ground_truth=data["ground_truth"],
                context=data.get("context", ""),
            )
        elif args.benchmark == "xbrl_math":
            # XBRL-Math handling
            sample = Sample(
                question=data.get("question", ""),
                context=data.get("context", ""),
                ground_truth=str(data.get("answer", "")),
            )
        elif args.benchmark == "simple_qa":
            # Squad/SQuAD handling - answers is a dict with text list
            answers = data.get("answers", {})
            if isinstance(answers, dict) and "text" in answers:
                ground_truth = answers["text"][0] if answers["text"] else ""
            else:
                ground_truth = str(answers) if answers else ""

            sample = Sample(
                question=data["question"],
                ground_truth=ground_truth,
                context=data.get("context", ""),
            )
        elif args.benchmark == "hellaswag":
            # HellaSwag handling - format multiple choice and convert label
            choices = data["endings"]
            question = f"""Context: {data['ctx']}

Which ending makes the most sense?

A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Answer with just the letter (A, B, C, or D)."""

            # Convert numeric label to letter
            label_map = {"0": "A", "1": "B", "2": "C", "3": "D"}
            ground_truth = label_map.get(str(data["label"]), "A")

            sample = Sample(question=question, ground_truth=ground_truth)
        elif args.benchmark in ["arc_easy", "arc_challenge"]:
            # ARC handling - format multiple choice
            choices = data["choices"]["text"]
            question = f"""Question: {data['question']}

A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Answer with just the letter (A, B, C, or D)."""

            sample = Sample(question=question, ground_truth=data["answerKey"])
        elif args.benchmark == "mmlu":
            # MMLU handling - format multiple choice
            choices = data["choices"]
            question = f"""Question: {data['question']}

A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Answer with just the letter (A, B, C, or D)."""

            # Convert numeric answer to letter
            answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
            ground_truth = answer_map.get(data["answer"], "A")

            sample = Sample(question=question, ground_truth=ground_truth)
        else:
            # Generic handling - check if already processed
            if "question" in data:
                # Already processed by a processor
                sample = Sample(
                    question=data["question"],
                    ground_truth=data.get("ground_truth", ""),
                    context=data.get("context", ""),
                )
            else:
                # Raw data - use generic handling
                sample = Sample(
                    question=str(data.get("question", data.get("input", ""))),
                    ground_truth=str(data.get("answer", data.get("output", ""))),
                    context=str(data.get("context", "")),
                )

        samples.append(sample)

    return samples


def run_comparison_mode(
    args: argparse.Namespace, samples: List[Sample], manager: BenchmarkTaskManager
) -> None:
    """Run both baseline and ACE evaluations, then compare results."""
    print(f"🚀 Running COMPARISON MODE for {args.benchmark}")
    print(
        f"Model: {args.model}, Samples: {len(samples)}, Prompt: {args.prompt_version}"
    )
    print("=" * 60)

    # Run baseline evaluation
    print("\n1️⃣ Running BASELINE evaluation...")
    baseline_args = argparse.Namespace(**vars(args))
    baseline_args.skip_adaptation = True
    baseline_results = run_evaluation(baseline_args, samples, manager)

    # Run ACE evaluation
    print("\n2️⃣ Running ACE evaluation...")
    ace_args = argparse.Namespace(**vars(args))
    ace_args.skip_adaptation = False
    ace_results = run_evaluation(ace_args, samples, manager)

    # Compare and display results
    print("\n" + "=" * 80)
    print("📊 BASELINE vs ACE COMPARISON")
    print("=" * 80)

    # Get metrics from both runs
    baseline_summary = baseline_results.get("summary", {})

    # For ACE, use test performance (true generalization)
    ace_summary = ace_results.get("test_summary", ace_results.get("summary", {}))

    print(f"\n🔬 BASELINE Performance:")
    for metric, value in baseline_summary.items():
        if metric.endswith("_mean"):
            base_metric = metric[:-5]
            print(f"  {base_metric.replace('_', ' ').title()}: {value:.2%}")

    print(f"\n🧠 ACE Performance (Test - True Generalization):")
    for metric, value in ace_summary.items():
        if metric.endswith("_mean"):
            base_metric = metric[:-5]
            improvement = ""
            if metric in baseline_summary:
                diff = value - baseline_summary[metric]
                if diff > 0:
                    improvement = f" (+{diff:.2%} ✅)"
                elif diff < 0:
                    improvement = f" ({diff:.2%} ⚠️)"
                else:
                    improvement = " (no change)"
            print(
                f"  {base_metric.replace('_', ' ').title()}: {value:.2%}{improvement}"
            )

    # Show overfitting analysis if available
    if "overfitting_gap" in ace_results and ace_results["overfitting_gap"]:
        print(f"\n📈 ACE Overfitting Analysis:")
        overfitting_gap = ace_results["overfitting_gap"]
        for metric, gap in overfitting_gap.items():
            if metric.endswith("_mean"):
                base_metric = metric[:-5]
                if gap > 0.05:
                    print(
                        f"  {base_metric.replace('_', ' ').title()}: {gap:.2%} ⚠️  (significant overfitting)"
                    )
                elif gap > 0.02:
                    print(
                        f"  {base_metric.replace('_', ' ').title()}: {gap:.2%} ⚡ (minor overfitting)"
                    )
                else:
                    print(
                        f"  {base_metric.replace('_', ' ').title()}: {gap:.2%} ✅ (good generalization)"
                    )

    print("\n" + "=" * 80)

    # Save comparison results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    comparison_file = (
        output_dir / f"comparison_{args.benchmark}_{args.model}_{timestamp}.json"
    )

    comparison_data = {
        "benchmark": args.benchmark,
        "model": args.model,
        "prompt_version": args.prompt_version,
        "timestamp": timestamp,
        "evaluation_mode": "comparison",
        "configuration": {
            "split": args.split,
            "epochs": args.epochs,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "split_ratio": args.split_ratio,
            "online_mode": args.online_mode,
            "prompt_version": args.prompt_version,
        },
        "baseline_results": baseline_results,
        "ace_results": ace_results,
        "summary": {
            "baseline_summary": baseline_summary,
            "ace_test_summary": ace_summary,
            "ace_train_summary": ace_results.get("train_summary", {}),
            "overfitting_gap": ace_results.get("overfitting_gap", {}),
        },
    }

    with open(comparison_file, "w") as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)

    print(f"💾 Comparison results saved to: {comparison_file}")
    print(f"✅ Comparison completed successfully!")


def create_ace_components(client: LiteLLMClient, prompt_version: str):
    """Create ACE components with specified prompt version."""
    if prompt_version == "v2":
        try:
            from ace.prompts_v2 import PromptManager

            manager = PromptManager()
            agent = Agent(client, prompt_template=manager.get_agent_prompt())
            reflector = Reflector(
                client, prompt_template=manager.get_reflector_prompt()
            )
            skill_manager = SkillManager(
                client, prompt_template=manager.get_skill_manager_prompt()
            )
        except ImportError:
            print("Warning: v2 prompts not available, falling back to v1")
            agent = Agent(client)
            reflector = Reflector(client)
            skill_manager = SkillManager(client)
    else:
        # Use default v1 prompts
        agent = Agent(client)
        reflector = Reflector(client)
        skill_manager = SkillManager(client)

    return agent, reflector, skill_manager


def split_samples(samples: List[Sample], split_ratio: float):
    """Split samples into train and test sets."""
    if split_ratio >= 1.0:
        return samples, []  # All training, no test

    split_idx = int(len(samples) * split_ratio)
    train_samples = samples[:split_idx]
    test_samples = samples[split_idx:]

    return train_samples, test_samples


def run_evaluation(
    args: argparse.Namespace, samples: List[Sample], manager: BenchmarkTaskManager
) -> Dict[str, Any]:
    """Run benchmark evaluation with ACE using proper train/test split."""
    if not args.quiet:
        print(
            f"Starting evaluation with {args.model} (prompt: {args.prompt_version})..."
        )

    # Create LLM client and ACE components with appropriate prompts
    client = create_llm_client(args)
    agent, reflector, skill_manager = create_ace_components(client, args.prompt_version)
    environment = manager.get_benchmark(args.benchmark)

    results = []
    train_results = []
    test_results = []

    if args.skip_adaptation:
        # Direct evaluation without ACE adaptation - use all samples as test
        if not args.quiet:
            print("🔬 Running BASELINE evaluation (no adaptation)")

        skillbook = Skillbook()

        for i, sample in enumerate(samples):
            if not args.quiet and i % 10 == 0:
                print(f"Progress: {i}/{len(samples)} samples processed")

            # Generate response
            output = agent.generate(
                question=sample.question, context=sample.context, skillbook=skillbook
            )

            # Evaluate
            env_result = environment.evaluate(sample, output)

            results.append(
                {
                    "sample_id": f"{args.benchmark}_{i:04d}",
                    "question": sample.question,
                    "prediction": output.final_answer,
                    "ground_truth": sample.ground_truth,
                    "metrics": env_result.metrics,
                    "feedback": env_result.feedback,
                    "split": "baseline",
                }
            )

        result_dict = {
            "benchmark": args.benchmark,
            "model": args.model,
            "prompt_version": args.prompt_version,
            "evaluation_mode": "baseline",
            "samples_evaluated": len(results),
            "results": results,
            "summary": compute_summary_metrics(results),
        }

    else:
        # ACE adaptation with train/test split
        train_samples, test_samples = split_samples(samples, args.split_ratio)

        if not args.quiet:
            print(
                f"📊 Train/test split: {len(train_samples)} train, {len(test_samples)} test (ratio: {args.split_ratio:.2f})"
            )

        if args.online_mode:
            # Online learning mode - learn from each sample sequentially
            if not args.quiet:
                print("🔄 Running ONLINE LEARNING evaluation")

            adapter = OnlineACE(
                skillbook=Skillbook(),
                agent=agent,
                reflector=reflector,
                skill_manager=skill_manager,
                max_refinement_rounds=args.max_refinement_rounds,
                enable_observability=True,
            )

            # Process all samples sequentially (each is learned from then tested)
            adaptation_results = adapter.run(samples, environment)

            # Convert to results format
            for step_idx, step in enumerate(adaptation_results):
                results.append(
                    {
                        "sample_id": f"{args.benchmark}_{step_idx:04d}",
                        "question": step.sample.question,
                        "prediction": step.agent_output.final_answer,
                        "ground_truth": step.sample.ground_truth,
                        "metrics": step.environment_result.metrics,
                        "feedback": step.environment_result.feedback,
                        "split": "online",
                        "step": step_idx,
                    }
                )

            result_dict = {
                "benchmark": args.benchmark,
                "model": args.model,
                "prompt_version": args.prompt_version,
                "evaluation_mode": "online",
                "samples_evaluated": len(results),
                "results": results,
                "summary": compute_summary_metrics(results),
            }

        else:
            # Offline learning with proper train/test split
            if not args.quiet:
                print(f"🧠 Running OFFLINE LEARNING evaluation ({args.epochs} epochs)")

            adapter = OfflineACE(
                skillbook=Skillbook(),
                agent=agent,
                reflector=reflector,
                skill_manager=skill_manager,
                max_refinement_rounds=args.max_refinement_rounds,
                enable_observability=True,
            )

            # Train on training samples
            if len(train_samples) > 0:
                if not args.quiet:
                    print(f"📚 Training on {len(train_samples)} samples...")
                adaptation_results = adapter.run(
                    train_samples, environment, epochs=args.epochs
                )

                # Store training results
                for step_idx, step in enumerate(adaptation_results):
                    train_results.append(
                        {
                            "sample_id": f"{args.benchmark}_train_{step_idx:04d}",
                            "question": step.sample.question,
                            "prediction": step.agent_output.final_answer,
                            "ground_truth": step.sample.ground_truth,
                            "metrics": step.environment_result.metrics,
                            "feedback": step.environment_result.feedback,
                            "split": "train",
                        }
                    )

            # Test on unseen test samples using learned skillbook
            if len(test_samples) > 0:
                if not args.quiet:
                    print(f"🧪 Testing on {len(test_samples)} unseen samples...")

                for i, sample in enumerate(test_samples):
                    # Generate response with learned skillbook
                    output = agent.generate(
                        question=sample.question,
                        context=sample.context,
                        skillbook=adapter.skillbook,
                    )

                    # Evaluate
                    env_result = environment.evaluate(sample, output)

                    test_results.append(
                        {
                            "sample_id": f"{args.benchmark}_test_{i:04d}",
                            "question": sample.question,
                            "prediction": output.final_answer,
                            "ground_truth": sample.ground_truth,
                            "metrics": env_result.metrics,
                            "feedback": env_result.feedback,
                            "split": "test",
                        }
                    )

            # Combine results
            results = train_results + test_results

            # Calculate overfitting gap
            train_summary = (
                compute_summary_metrics(train_results) if train_results else {}
            )
            test_summary = compute_summary_metrics(test_results) if test_results else {}

            overfitting_gap = {}
            for metric in train_summary:
                if metric in test_summary:
                    overfitting_gap[metric] = (
                        train_summary[metric] - test_summary[metric]
                    )

            result_dict = {
                "benchmark": args.benchmark,
                "model": args.model,
                "prompt_version": args.prompt_version,
                "evaluation_mode": "offline_train_test_split",
                "split_ratio": args.split_ratio,
                "train_samples": len(train_samples),
                "test_samples": len(test_samples),
                "epochs": args.epochs,
                "samples_evaluated": len(results),
                "results": results,
                "train_summary": train_summary,
                "test_summary": test_summary,
                "overfitting_gap": overfitting_gap,
                "summary": test_summary,  # Overall summary uses test performance (TRUE performance)
            }

        # Export observability data if available
        observability_data = None
        if hasattr(adapter, "observability_data"):
            observability_data = adapter.observability_data

        if observability_data:
            result_dict["observability"] = observability_data

    return result_dict


def compute_summary_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute summary metrics across all results."""
    if not results:
        return {}

    # Collect all metric values
    all_metrics = {}
    for result in results:
        for metric_name, value in result["metrics"].items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)

    # Compute averages
    summary = {}
    for metric_name, values in all_metrics.items():
        summary[f"{metric_name}_mean"] = mean(values)
        summary[f"{metric_name}_min"] = min(values)
        summary[f"{metric_name}_max"] = max(values)

    return summary


def save_results(args: argparse.Namespace, evaluation_results: Dict[str, Any]) -> None:
    """Save evaluation results to files."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base_name = f"{args.benchmark}_{args.model}_{timestamp}"

    # Save summary results
    summary_file = output_dir / f"{base_name}_summary.json"
    summary_data = {
        "benchmark": evaluation_results["benchmark"],
        "model": evaluation_results["model"],
        "timestamp": timestamp,
        "samples_evaluated": evaluation_results["samples_evaluated"],
        "summary_metrics": evaluation_results["summary"],
        "configuration": {
            "split": args.split,
            "epochs": args.epochs,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "skip_adaptation": args.skip_adaptation,
            "split_ratio": args.split_ratio,
            "online_mode": args.online_mode,
            "prompt_version": args.prompt_version,
            "evaluation_mode": evaluation_results.get("evaluation_mode", "unknown"),
        },
    }

    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    if not args.quiet:
        print(f"Summary saved to: {summary_file}")

    # Save detailed results if requested
    if args.save_detailed:
        detailed_file = output_dir / f"{base_name}_detailed.json"
        with open(detailed_file, "w") as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

        if not args.quiet:
            print(f"Detailed results saved to: {detailed_file}")

    # Print summary to console
    print("\n" + "=" * 60)
    print(f"Benchmark: {evaluation_results['benchmark']}")
    print(f"Model: {evaluation_results['model']}")
    print(f"Prompt Version: {evaluation_results.get('prompt_version', 'v1')}")
    print(f"Evaluation Mode: {evaluation_results.get('evaluation_mode', 'unknown')}")

    if "train_samples" in evaluation_results and "test_samples" in evaluation_results:
        print(
            f"Train/Test Split: {evaluation_results['train_samples']}/{evaluation_results['test_samples']} (ratio: {evaluation_results.get('split_ratio', 0.8):.2f})"
        )
    else:
        print(f"Samples: {evaluation_results['samples_evaluated']}")
    print("=" * 60)

    # Show test metrics (true performance) for train/test split
    if "test_summary" in evaluation_results and evaluation_results["test_summary"]:
        print("🧪 TEST PERFORMANCE (True Generalization):")
        for metric, value in evaluation_results["test_summary"].items():
            if metric.endswith("_mean"):
                base_metric = metric[:-5]
                print(f"  {base_metric.replace('_', ' ').title()}: {value:.2%}")

        # Show overfitting gap if available
        if (
            "overfitting_gap" in evaluation_results
            and evaluation_results["overfitting_gap"]
        ):
            print("\n📈 OVERFITTING ANALYSIS:")
            for metric, gap in evaluation_results["overfitting_gap"].items():
                if metric.endswith("_mean"):
                    base_metric = metric[:-5]
                    if gap > 0.05:  # Significant overfitting
                        print(
                            f"  {base_metric.replace('_', ' ').title()} Gap: {gap:.2%} ⚠️  (overfitting)"
                        )
                    else:
                        print(
                            f"  {base_metric.replace('_', ' ').title()} Gap: {gap:.2%} ✅"
                        )

        # Show training performance for reference
        if (
            "train_summary" in evaluation_results
            and evaluation_results["train_summary"]
        ):
            print("\n📚 TRAIN PERFORMANCE (Reference):")
            for metric, value in evaluation_results["train_summary"].items():
                if metric.endswith("_mean"):
                    base_metric = metric[:-5]
                    print(f"  {base_metric.replace('_', ' ').title()}: {value:.2%}")
    else:
        # Fallback for baseline or online mode
        for metric, value in evaluation_results["summary"].items():
            if metric.endswith("_mean"):
                base_metric = metric[:-5]
                print(f"{base_metric.replace('_', ' ').title()}: {value:.2%}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Handle special commands
    if args.benchmark == "list":
        list_available_benchmarks()
        return

    # Set cache directory if specified
    if args.cache_dir:
        os.environ["BENCHMARK_CACHE_DIR"] = args.cache_dir

    # Initialize benchmark manager
    try:
        manager = BenchmarkTaskManager()
    except Exception as e:
        print(f"Error initializing benchmark manager: {e}")
        sys.exit(1)

    # Validate benchmark exists
    if args.benchmark not in manager.list_benchmarks():
        print(f"Error: Unknown benchmark '{args.benchmark}'")
        print("Use 'list' to see available benchmarks")
        sys.exit(1)

    # Validate benchmark configuration
    validation_errors = manager.validate_config(args.benchmark)
    if validation_errors:
        print(f"Benchmark validation errors:")
        for error in validation_errors:
            print(f"  - {error}")
        sys.exit(1)

    # Load benchmark data
    samples = load_benchmark_data(args, manager)

    # Check if running in comparison mode
    if args.compare:
        # Run comparison mode (baseline vs ACE)
        run_comparison_mode(args, samples, manager)
    else:
        # Run normal evaluation
        evaluation_results = run_evaluation(args, samples, manager)

        # Save and display results
        save_results(args, evaluation_results)

    if not args.quiet:
        print(f"\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
