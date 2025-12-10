#!/usr/bin/env python3
"""
Evaluate Claude Sonnet 4.5 on FiNER, GSM8K, and MMLU benchmarks.
"""

from __future__ import annotations

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

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# API key should be set via ANTHROPIC_API_KEY environment variable
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("Warning: ANTHROPIC_API_KEY not set. Set it via environment or .env file.")
    print("Usage: set ANTHROPIC_API_KEY=your-key-here && python scripts/evaluate_sonnet_4_5.py")

from ace import Agent, Skillbook, Sample
from ace.llm_providers import LiteLLMClient
from benchmarks import BenchmarkTaskManager

# Suppress LiteLLM debug messages
import litellm
litellm.suppress_debug_info = True


def create_llm_client(model: str = "claude-sonnet-4-20250514") -> LiteLLMClient:
    """Create LLM client for Sonnet 4.5."""
    return LiteLLMClient(
        model=model,
        temperature=0.0,
        max_tokens=2048,
        timeout=120,
    )


def load_benchmark_data(benchmark: str, manager: BenchmarkTaskManager, limit: int = 50) -> List[Sample]:
    """Load and convert benchmark data to Sample format.

    The BenchmarkTaskManager's load_benchmark_data method returns pre-processed data
    via the HuggingFaceLoader + processors. Data has standardized keys:
    - question: The formatted question
    - ground_truth: The expected answer
    - context: Optional context
    - metadata: Optional metadata dict
    """
    print(f"Loading {benchmark} data...")

    try:
        raw_data = list(manager.load_benchmark_data(benchmark))
    except Exception as e:
        print(f"Error loading benchmark data: {e}")
        return []

    # Apply limit
    raw_data = raw_data[:limit]
    print(f"Loaded {len(raw_data)} samples")

    samples = []

    for i, data in enumerate(raw_data):
        # Data is already pre-processed by the HuggingFaceLoader + processors
        # All benchmarks return standardized format with question, ground_truth, context
        sample = Sample(
            question=data.get("question", ""),
            ground_truth=data.get("ground_truth", ""),
            context=data.get("context", ""),
        )
        samples.append(sample)

    return samples


def evaluate_benchmark(
    benchmark: str,
    client: LiteLLMClient,
    manager: BenchmarkTaskManager,
    limit: int = 50
) -> Dict[str, Any]:
    """Run evaluation on a specific benchmark."""
    print(f"\n{'='*60}")
    print(f"Evaluating {benchmark.upper()} benchmark")
    print(f"{'='*60}")

    samples = load_benchmark_data(benchmark, manager, limit)
    if not samples:
        return {"error": f"No samples loaded for {benchmark}"}

    environment = manager.get_benchmark(benchmark)
    agent = Agent(client)
    skillbook = Skillbook()

    results = []
    correct = 0
    total = 0

    for i, sample in enumerate(samples):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(samples)} samples processed")

        try:
            # Generate response
            output = agent.generate(
                question=sample.question,
                context=sample.context,
                skillbook=skillbook
            )

            # Evaluate
            env_result = environment.evaluate(sample, output)

            # Check if correct (based on available metrics)
            # Different benchmarks return different metrics:
            # - FiNER: precision, recall, f1
            # - GSM8K: exact_match, accuracy, within_X_percent
            # - MMLU: exact_match, accuracy
            is_correct = (
                env_result.metrics.get("exact_match", 0) == 1.0 or
                env_result.metrics.get("accuracy", 0) == 1.0 or
                env_result.metrics.get("f1", 0) >= 0.8  # FiNER: consider 80%+ F1 as correct
            )

            if is_correct:
                correct += 1
            total += 1

            results.append({
                "sample_id": f"{benchmark}_{i:04d}",
                "question": sample.question[:200] + "..." if len(sample.question) > 200 else sample.question,
                "prediction": output.final_answer,
                "ground_truth": sample.ground_truth,
                "metrics": env_result.metrics,
                "feedback": env_result.feedback,
                "correct": is_correct,
            })

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            results.append({
                "sample_id": f"{benchmark}_{i:04d}",
                "error": str(e),
                "correct": False,
            })
            total += 1

    # Compute summary
    accuracy = correct / total if total > 0 else 0

    # Aggregate all metrics
    all_metrics = {}
    for result in results:
        if "metrics" in result:
            for metric_name, value in result["metrics"].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

    summary_metrics = {}
    for metric_name, values in all_metrics.items():
        summary_metrics[f"{metric_name}_mean"] = mean(values) if values else 0

    summary_metrics["accuracy"] = accuracy
    summary_metrics["correct"] = correct
    summary_metrics["total"] = total

    print(f"\n{benchmark.upper()} Results:")
    print(f"  Accuracy: {accuracy:.2%} ({correct}/{total})")
    for metric, value in summary_metrics.items():
        if metric.endswith("_mean"):
            print(f"  {metric}: {value:.4f}")

    return {
        "benchmark": benchmark,
        "model": client.config.model,
        "samples_evaluated": total,
        "accuracy": accuracy,
        "correct": correct,
        "summary": summary_metrics,
        "detailed_results": results,
    }


def main():
    """Main entry point."""
    print("="*60)
    print("Sonnet 4.5 Benchmark Evaluation")
    print("="*60)

    # Initialize
    manager = BenchmarkTaskManager()

    # Use Claude Sonnet 4.5 (claude-sonnet-4-20250514 is the model ID)
    print("\nInitializing Claude Sonnet 4.5...")
    client = create_llm_client("claude-sonnet-4-20250514")

    # Run evaluations
    benchmarks = ["finer_ord", "gsm8k", "mmlu"]
    all_results = {}

    for benchmark in benchmarks:
        try:
            result = evaluate_benchmark(benchmark, client, manager, limit=50)
            all_results[benchmark] = result
        except Exception as e:
            print(f"Error running {benchmark}: {e}")
            import traceback
            traceback.print_exc()
            all_results[benchmark] = {"error": str(e)}

    # Save results
    output_dir = Path("benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"sonnet_4_5_evaluation_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY - Sonnet 4.5 Performance")
    print("="*60)

    for benchmark, result in all_results.items():
        if "error" in result and not result.get("accuracy"):
            print(f"\n{benchmark.upper()}: ERROR - {result['error']}")
        else:
            accuracy = result.get("accuracy", 0)
            total = result.get("samples_evaluated", 0)
            print(f"\n{benchmark.upper()}:")
            print(f"  Accuracy: {accuracy:.2%}")
            print(f"  Samples: {total}")
            if "summary" in result:
                for metric, value in result["summary"].items():
                    if metric.endswith("_mean") and value > 0:
                        print(f"  {metric}: {value:.4f}")

    print(f"\nResults saved to: {output_file}")
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
