#!/usr/bin/env python3
"""
ACELiteLLM Async Learning Example

Demonstrates the difference between sync and async learning:
- Sync: Learning blocks after each sample
- Async: Learning runs in background, samples processed faster

Uses real LLM calls (gpt-4o-mini) to show actual behavior.
"""

import os
import time
from dotenv import load_dotenv

from ace import ACELiteLLM, Sample, SimpleEnvironment

load_dotenv()


def run_sync_learning():
    """Run learning in SYNC mode (blocking)."""
    print("\n" + "=" * 60)
    print("SYNC MODE - Learning blocks after each sample")
    print("=" * 60)

    agent = ACELiteLLM(model="claude-sonnet-4-5-20250929")

    samples = [
        Sample(question="What is 2+2?", ground_truth="4"),
        Sample(question="What color is the sky?", ground_truth="blue"),
        Sample(question="Capital of France?", ground_truth="Paris"),
        Sample(question="What is 10-3?", ground_truth="7"),
        Sample(question="How many days in a week?", ground_truth="7"),
    ]

    environment = SimpleEnvironment()

    print(f"Processing {len(samples)} samples...")
    start = time.time()

    # Sync learning (default)
    results = agent.learn(
        samples,
        environment,
        epochs=1,
        async_learning=False,  # Blocking mode
    )

    elapsed = time.time() - start

    print(f"\nResults:")
    print(f"  - Samples processed: {len(results)}")
    print(f"  - Time elapsed: {elapsed:.2f}s")
    print(f"  - Strategies learned: {len(agent.skillbook.skills())}")

    return elapsed


def run_async_learning():
    """Run learning in ASYNC mode (non-blocking)."""
    print("\n" + "=" * 60)
    print("ASYNC MODE - Learning runs in background")
    print("=" * 60)

    agent = ACELiteLLM(model="claude-sonnet-4-5-20250929")

    samples = [
        Sample(question="What is 2+2?", ground_truth="4"),
        Sample(question="What color is the sky?", ground_truth="blue"),
        Sample(question="Capital of France?", ground_truth="Paris"),
        Sample(question="What is 10-3?", ground_truth="7"),
        Sample(question="How many days in a week?", ground_truth="7"),
    ]

    environment = SimpleEnvironment()

    print(f"Processing {len(samples)} samples...")
    start = time.time()

    # Async learning - Generator returns faster, learning in background
    results = agent.learn(
        samples,
        environment,
        epochs=1,
        async_learning=True,  # Non-blocking mode
        max_reflector_workers=3,
    )

    results_time = time.time() - start
    print(f"\n✅ Results returned in: {results_time:.2f}s")
    print(f"   (Learning still running in background...)")

    # Check learning stats before waiting
    stats = agent.learning_stats
    print(f"\nLearning stats (before wait):")
    print(f"  - Pipeline running: {stats.get('is_running', 'N/A')}")

    # Wait for learning to complete
    print("\n⏳ Waiting for background learning to complete...")
    wait_start = time.time()
    agent.wait_for_learning(timeout=60.0)
    wait_time = time.time() - wait_start

    total_time = time.time() - start

    print(f"\nResults:")
    print(f"  - Samples processed: {len(results)}")
    print(f"  - Results returned in: {results_time:.2f}s")
    print(f"  - Learning wait time: {wait_time:.2f}s")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Strategies learned: {len(agent.skillbook.skills())}")

    # Show complete skillbook
    print(f"\n📚 COMPLETE SKILLBOOK:")
    if agent.skillbook.skills():
        print(str(agent.skillbook))
    else:
        print("(empty)")

    return results_time, total_time


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY in your .env file")
        return

    print("=" * 60)
    print("ACELiteLLM ASYNC LEARNING DEMO")
    print("=" * 60)
    print("\nThis demo shows the difference between sync and async modes.")
    print("In async mode, Generator results return faster while learning")
    print("continues in the background.")

    # Run sync demo
    sync_time = run_sync_learning()

    # Run async demo
    async_results_time, async_total_time = run_async_learning()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nSync mode total time:    {sync_time:.2f}s")
    print(f"Async mode results time: {async_results_time:.2f}s")
    print(f"Async mode total time:   {async_total_time:.2f}s")

    if async_results_time < sync_time:
        speedup = sync_time / async_results_time
        print(f"\n✅ Async mode returned results {speedup:.1f}x faster!")
    else:
        print("\n⚠️ Async mode didn't show speedup (samples may be too few)")


if __name__ == "__main__":
    main()
