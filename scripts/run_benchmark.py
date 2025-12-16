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

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
import threading
import atexit

litellm.suppress_debug_info = True


class IncrementalResultWriter:
    """
    Writes benchmark results incrementally to prevent data loss on crash.

    Creates a JSONL file where each line is a complete result.
    Also maintains a summary file that's updated periodically.
    """

    def __init__(self, output_dir: Path, benchmark: str, model: str, config: dict):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.base_name = f"{benchmark}_{model}_{self.timestamp}"

        # JSONL file for incremental results (one JSON object per line)
        self.results_file = self.output_dir / f"{self.base_name}_incremental.jsonl"
        self.summary_file = self.output_dir / f"{self.base_name}_live_summary.json"

        self.config = config
        self.results = []
        self.train_results = []
        self.test_results = []
        self._lock = threading.Lock()

        # Write initial metadata
        self._write_metadata()

        # Register cleanup on exit
        atexit.register(self._finalize)

    def _write_metadata(self):
        """Write initial metadata to the results file."""
        metadata = {
            "_type": "metadata",
            "timestamp": self.timestamp,
            "config": self.config,
        }
        with open(self.results_file, 'w') as f:
            f.write(json.dumps(metadata) + '\n')

    def add_result(self, result: dict, split: str = "baseline"):
        """Add a single result and write it immediately."""
        with self._lock:
            result["_type"] = "result"
            result["_index"] = len(self.results)

            # Append to JSONL file immediately
            with open(self.results_file, 'a') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

            self.results.append(result)

            # Track by split
            if split == "train":
                self.train_results.append(result)
            elif split == "test":
                self.test_results.append(result)

            # Update summary every 5 results
            if len(self.results) % 5 == 0:
                self._update_summary()

    def _update_summary(self):
        """Update the live summary file."""
        summary = {
            "timestamp": self.timestamp,
            "samples_processed": len(self.results),
            "train_samples": len(self.train_results),
            "test_samples": len(self.test_results),
            "config": self.config,
            "status": "in_progress",
        }

        if self.results:
            summary["current_metrics"] = self._compute_metrics(self.results)
        if self.train_results:
            summary["train_metrics"] = self._compute_metrics(self.train_results)
        if self.test_results:
            summary["test_metrics"] = self._compute_metrics(self.test_results)

        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

    def _compute_metrics(self, results: list) -> dict:
        """Compute summary metrics from results."""
        if not results:
            return {}

        all_metrics = {}
        for result in results:
            if "metrics" not in result:
                continue
            for metric_name, value in result["metrics"].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        summary = {}
        for metric_name, values in all_metrics.items():
            if values:
                summary[f"{metric_name}_mean"] = mean(values)

        return summary

    def _finalize(self):
        """Called on exit to write final summary."""
        with self._lock:
            summary = {
                "timestamp": self.timestamp,
                "samples_processed": len(self.results),
                "train_samples": len(self.train_results),
                "test_samples": len(self.test_results),
                "config": self.config,
                "status": "completed" if len(self.results) > 0 else "no_results",
            }

            if self.results:
                summary["final_metrics"] = self._compute_metrics(self.results)
            if self.train_results:
                summary["train_metrics"] = self._compute_metrics(self.train_results)
            if self.test_results:
                summary["test_metrics"] = self._compute_metrics(self.test_results)

            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

    def get_results(self) -> list:
        """Get all collected results."""
        with self._lock:
            return list(self.results)

    def get_train_results(self) -> list:
        """Get training results."""
        with self._lock:
            return list(self.train_results)

    def get_test_results(self) -> list:
        """Get test results."""
        with self._lock:
            return list(self.test_results)


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
            # AppWorld has special handling - may come pre-processed
            if "question" in data and "ground_truth" in data:
                sample = Sample(
                    question=data["question"],
                    ground_truth=data["ground_truth"],
                    context=data.get("context", ""),
                    metadata=data.get("metadata", {}),
                )
            else:
                sample = Sample(
                    question=data.get("instruction", ""),
                    context=f"Available APIs: {data.get('api_docs', '')}",
                    ground_truth="Task completion successful",
                    metadata=data.get("metadata", {}),
                )
        elif args.benchmark == "finer_ord":
            # FiNER now comes pre-processed from the loader
            sample = Sample(
                question=data["question"],
                ground_truth=data["ground_truth"],
                context=data.get("context", ""),
            )
        elif args.benchmark == "xbrl_math":
            # XBRL-Math handling - may come pre-processed
            if "question" in data and "ground_truth" in data:
                sample = Sample(
                    question=data["question"],
                    ground_truth=data["ground_truth"],
                    context=data.get("context", ""),
                )
            else:
                sample = Sample(
                    question=data.get("question", ""),
                    context=data.get("context", ""),
                    ground_truth=str(data.get("answer", "")),
                )
        elif args.benchmark == "simple_qa":
            # Squad/SQuAD handling - may come pre-processed
            if "question" in data and "ground_truth" in data:
                sample = Sample(
                    question=data["question"],
                    ground_truth=data["ground_truth"],
                    context=data.get("context", ""),
                    metadata=data.get("metadata", {}),
                )
            else:
                # Raw data - answers is a dict with text list
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
            # HellaSwag handling - may come pre-processed
            if "question" in data and "ground_truth" in data:
                sample = Sample(
                    question=data["question"],
                    ground_truth=data["ground_truth"],
                    context=data.get("context", ""),
                    metadata=data.get("metadata", {}),
                )
            else:
                # Raw data - format multiple choice
                choices = data.get("endings", [])
                question = f"""Context: {data.get('ctx', '')}

Which ending makes the most sense?

A) {choices[0] if len(choices) > 0 else ''}
B) {choices[1] if len(choices) > 1 else ''}
C) {choices[2] if len(choices) > 2 else ''}
D) {choices[3] if len(choices) > 3 else ''}

Answer with just the letter (A, B, C, or D)."""

                # Convert numeric label to letter
                label_map = {"0": "A", "1": "B", "2": "C", "3": "D"}
                ground_truth = label_map.get(str(data.get("label", "0")), "A")

                sample = Sample(question=question, ground_truth=ground_truth)
        elif args.benchmark in ["arc_easy", "arc_challenge"]:
            # ARC handling - may come pre-processed
            if "question" in data and "ground_truth" in data:
                sample = Sample(
                    question=data["question"],
                    ground_truth=data["ground_truth"],
                    context=data.get("context", ""),
                    metadata=data.get("metadata", {}),
                )
            else:
                # Raw data - format multiple choice
                choices_data = data.get("choices", {})
                choices = choices_data.get("text", []) if isinstance(choices_data, dict) else []
                question = f"""Question: {data.get('question', '')}

A) {choices[0] if len(choices) > 0 else ''}
B) {choices[1] if len(choices) > 1 else ''}
C) {choices[2] if len(choices) > 2 else ''}
D) {choices[3] if len(choices) > 3 else ''}

Answer with just the letter (A, B, C, or D)."""

                sample = Sample(question=question, ground_truth=data.get("answerKey", "A"))
        elif args.benchmark == "mmlu":
            # MMLU handling - may come pre-processed from MultipleChoiceProcessor
            if "question" in data and "ground_truth" in data:
                # Already processed by processor
                sample = Sample(
                    question=data["question"],
                    ground_truth=data["ground_truth"],
                    context=data.get("context", ""),
                    metadata=data.get("metadata", {}),
                )
            else:
                # Raw data - format multiple choice
                choices = data.get("choices", [])
                question = f"""Question: {data.get('question', '')}

A) {choices[0] if len(choices) > 0 else ''}
B) {choices[1] if len(choices) > 1 else ''}
C) {choices[2] if len(choices) > 2 else ''}
D) {choices[3] if len(choices) > 3 else ''}

Answer with just the letter (A, B, C, or D)."""

                # Convert numeric answer to letter (handle both int and string)
                answer_idx = data.get("answer", 0)
                if isinstance(answer_idx, str):
                    answer_idx = int(answer_idx) if answer_idx.isdigit() else 0
                answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
                ground_truth = answer_map.get(answer_idx, "A")

                sample = Sample(question=question, ground_truth=ground_truth)
        elif args.benchmark == "swe_bench":
            # SWE-bench handling - software engineering bug fixes
            hints = data.get("hints_text", "")
            hints_section = f"\n\nHints: {hints}" if hints else ""

            question = f"""Repository: {data.get('repo', 'unknown')}

Issue: {data.get('problem_statement', '')}

Base Commit: {data.get('base_commit', '')}{hints_section}

Please analyze this issue and provide a patch (in unified diff format) that resolves it.
Include your reasoning before the patch."""

            sample = Sample(
                question=question,
                ground_truth=data.get("patch", ""),
                context=data.get("test_patch", ""),
                metadata=data.get("metadata", {}),
            )
        elif args.benchmark == "gsm8k":
            # GSM8K math problems - may come pre-processed from GSM8KProcessor
            if "question" in data and "ground_truth" in data and data.get("ground_truth"):
                # Already processed by GSM8KProcessor
                sample = Sample(
                    question=data["question"],
                    ground_truth=data["ground_truth"],
                    context=data.get("context", ""),
                    metadata=data.get("metadata", {}),
                )
            else:
                # Raw data - format it ourselves
                question = f"""Solve this math problem step by step:

{data.get('question', '')}

Provide your final numerical answer after ####."""

                # Extract final answer from GSM8K format (#### NUMBER)
                import re
                answer = data.get("answer", "")
                match = re.search(r"####\s*(.+)", answer)
                ground_truth = match.group(1).strip().replace(",", "") if match else answer

                sample = Sample(
                    question=question,
                    ground_truth=ground_truth,
                    context=answer,  # Full solution as context
                )
        elif args.benchmark == "letta_bench":
            # Letta benchmark handling
            history = data.get("conversation_history", "")
            query = data.get("query", "")

            question = f"""{history}

Current query: {query}

Respond based on the conversation history and any relevant memories."""

            sample = Sample(
                question=question,
                ground_truth=data.get("expected_response", ""),
                metadata=data.get("metadata", {}),
            )
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

    # Create incremental result writer for crash recovery
    writer_config = {
        "benchmark": args.benchmark,
        "model": args.model,
        "prompt_version": args.prompt_version,
        "split": args.split,
        "epochs": args.epochs,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "skip_adaptation": args.skip_adaptation,
        "split_ratio": args.split_ratio,
        "online_mode": args.online_mode,
        "total_samples": len(samples),
    }
    writer = IncrementalResultWriter(
        output_dir=Path(args.output),
        benchmark=args.benchmark,
        model=args.model,
        config=writer_config,
    )

    if not args.quiet:
        print(f"📝 Incremental results: {writer.results_file}")
        print(f"📊 Live summary: {writer.summary_file}")

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

            # Build the prompt for logging (same as Agent._generate_impl)
            prompt_used = agent.prompt_template.format(
                skillbook=skillbook.as_prompt() or "(empty skillbook)",
                reflection="(none)",
                question=sample.question,
                context=sample.context or "(none)",
            )

            result = {
                "sample_id": f"{args.benchmark}_{i:04d}",
                "question": sample.question,
                "context": sample.context or "",
                "prompt": prompt_used,
                "reasoning": output.reasoning,
                "prediction": output.final_answer,
                "ground_truth": sample.ground_truth,
                "skill_ids_cited": output.skill_ids,
                "metrics": env_result.metrics,
                "feedback": env_result.feedback,
                "split": "baseline",
            }

            # Save incrementally
            writer.add_result(result, split="baseline")
            results.append(result)

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

            # Process samples one at a time with incremental saving
            failed_count = 0
            for step_idx, sample in enumerate(samples, start=1):
                if not args.quiet and step_idx % 10 == 0:
                    print(f"  Online progress: {step_idx}/{len(samples)}")

                try:
                    # Process single sample through ACE pipeline
                    step = adapter._process_sample(
                        sample,
                        environment,
                        epoch=1,
                        total_epochs=1,
                        step_index=step_idx,
                        total_steps=len(samples),
                    )

                    # Build the prompt for logging
                    prompt_used = agent.prompt_template.format(
                        skillbook=adapter.skillbook.as_prompt() or "(empty skillbook)",
                        reflection="(none)",
                        question=step.sample.question,
                        context=step.sample.context or "(none)",
                    )

                    result = {
                        "sample_id": f"{args.benchmark}_{step_idx:04d}",
                        "question": step.sample.question,
                        "context": step.sample.context or "",
                        "prompt": prompt_used,
                        "reasoning": step.agent_output.reasoning,
                        "prediction": step.agent_output.final_answer,
                        "ground_truth": step.sample.ground_truth,
                        "skill_ids_cited": step.agent_output.skill_ids,
                        "metrics": step.environment_result.metrics,
                        "feedback": step.environment_result.feedback,
                        "split": "online",
                        "step": step_idx,
                    }

                    # Save incrementally
                    writer.add_result(result, split="online")
                    results.append(result)

                except Exception as e:
                    failed_count += 1
                    if not args.quiet:
                        print(f"  ⚠️ Failed sample {step_idx}/{len(samples)}: {type(e).__name__}")
                    error_result = {
                        "sample_id": f"{args.benchmark}_{step_idx:04d}",
                        "question": sample.question[:200] if sample.question else "",
                        "error": str(e)[:500],
                        "split": "online",
                        "status": "failed",
                    }
                    writer.add_result(error_result, split="online")
                    continue

            if not args.quiet:
                print(f"  ✓ Online learning complete: {len(results)} successful, {failed_count} failed")

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

            # Train on training samples WITH INCREMENTAL SAVING
            if len(train_samples) > 0:
                if not args.quiet:
                    print(f"📚 Training on {len(train_samples)} samples...")

                total_train_steps = len(train_samples) * args.epochs
                step_count = 0
                failed_count = 0

                for epoch_idx in range(1, args.epochs + 1):
                    for sample_idx, sample in enumerate(train_samples, start=1):
                        step_count += 1

                        if not args.quiet and step_count % 10 == 0:
                            print(f"  Training progress: {step_count}/{total_train_steps} (epoch {epoch_idx}/{args.epochs})")

                        try:
                            # Process single sample through ACE pipeline
                            step = adapter._process_sample(
                                sample,
                                environment,
                                epoch=epoch_idx,
                                total_epochs=args.epochs,
                                step_index=sample_idx,
                                total_steps=len(train_samples),
                            )

                            # Build the prompt for logging
                            prompt_used = agent.prompt_template.format(
                                skillbook=adapter.skillbook.as_prompt() or "(empty skillbook)",
                                reflection="(none)",
                                question=step.sample.question,
                                context=step.sample.context or "(none)",
                            )

                            result = {
                                "sample_id": f"{args.benchmark}_train_{step_count:04d}",
                                "question": step.sample.question,
                                "context": step.sample.context or "",
                                "prompt": prompt_used,
                                "reasoning": step.agent_output.reasoning,
                                "prediction": step.agent_output.final_answer,
                                "ground_truth": step.sample.ground_truth,
                                "skill_ids_cited": step.agent_output.skill_ids,
                                "metrics": step.environment_result.metrics,
                                "feedback": step.environment_result.feedback,
                                "split": "train",
                                "epoch": epoch_idx,
                            }

                            # Save incrementally
                            writer.add_result(result, split="train")
                            train_results.append(result)

                        except Exception as e:
                            failed_count += 1
                            # Log error but continue
                            if not args.quiet:
                                print(f"  ⚠️ Failed sample {step_count}/{total_train_steps}: {type(e).__name__}")
                            # Save failure record
                            error_result = {
                                "sample_id": f"{args.benchmark}_train_{step_count:04d}",
                                "question": sample.question[:200] if sample.question else "",
                                "error": str(e)[:500],
                                "split": "train",
                                "epoch": epoch_idx,
                                "status": "failed",
                            }
                            writer.add_result(error_result, split="train")
                            continue

                if not args.quiet:
                    print(f"  ✓ Training complete: {len(train_results)} successful, {failed_count} failed")

            # Test on unseen test samples using learned skillbook WITH INCREMENTAL SAVING
            if len(test_samples) > 0:
                if not args.quiet:
                    print(f"🧪 Testing on {len(test_samples)} unseen samples...")

                test_failed_count = 0
                for i, sample in enumerate(test_samples):
                    if not args.quiet and i % 10 == 0:
                        print(f"  Test progress: {i}/{len(test_samples)}")

                    try:
                        # Generate response with learned skillbook
                        output = agent.generate(
                            question=sample.question,
                            context=sample.context,
                            skillbook=adapter.skillbook,
                        )

                        # Evaluate
                        env_result = environment.evaluate(sample, output)

                        # Build the prompt for logging
                        prompt_used = agent.prompt_template.format(
                            skillbook=adapter.skillbook.as_prompt() or "(empty skillbook)",
                            reflection="(none)",
                            question=sample.question,
                            context=sample.context or "(none)",
                        )

                        result = {
                            "sample_id": f"{args.benchmark}_test_{i:04d}",
                            "question": sample.question,
                            "context": sample.context or "",
                            "prompt": prompt_used,
                            "reasoning": output.reasoning,
                            "prediction": output.final_answer,
                            "ground_truth": sample.ground_truth,
                            "skill_ids_cited": output.skill_ids,
                            "metrics": env_result.metrics,
                            "feedback": env_result.feedback,
                            "split": "test",
                        }

                        # Save incrementally
                        writer.add_result(result, split="test")
                        test_results.append(result)

                    except Exception as e:
                        test_failed_count += 1
                        if not args.quiet:
                            print(f"  ⚠️ Failed test sample {i}/{len(test_samples)}: {type(e).__name__}")
                        # Save failure record
                        error_result = {
                            "sample_id": f"{args.benchmark}_test_{i:04d}",
                            "question": sample.question[:200] if sample.question else "",
                            "error": str(e)[:500],
                            "split": "test",
                            "status": "failed",
                        }
                        writer.add_result(error_result, split="test")
                        continue

                if not args.quiet:
                    print(f"  ✓ Testing complete: {len(test_results)} successful, {test_failed_count} failed")

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
