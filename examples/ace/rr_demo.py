#!/usr/bin/env python3
"""Demo of the Recursive Reflector (RR) pipeline with a real LLM.

Shows the RR analyzing agent traces, iterating in its Python REPL sandbox,
and producing structured learnings.  Requires an API key for LiteLLM.

Usage:
    # Default model (Bedrock Claude Haiku):
    uv run python examples/ace/rr_demo.py

    # Custom model:
    ACE_MODEL=bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0 uv run python examples/ace/rr_demo.py
"""

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is importable
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
load_dotenv(_root / ".env")

from ace.steps.rr_step import RRConfig, RRStep, TraceSandbox
from ace.core.context import ACEStepContext, SkillbookView
from ace.core.skillbook import Skillbook

MODEL = os.getenv("ACE_MODEL", "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")

# Show what the RR is doing at each iteration
logging.basicConfig(
    level=logging.INFO,
    format="  %(name)s | %(message)s",
)
# Quiet the noisy libraries
for name in ("LiteLLM", "litellm", "httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.WARNING)


def section(name: str) -> None:
    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}\n")


def print_result(result):
    """Print a ReflectorOutput nicely."""
    print(f"\n  --- Result ---")
    print(f"  Reasoning: {result.reasoning[:300]}")
    print(f"  Key insight: {result.key_insight}")
    if result.error_identification:
        print(f"  Error: {result.error_identification}")
    if result.root_cause_analysis:
        print(f"  Root cause: {result.root_cause_analysis}")
    if result.correct_approach:
        print(f"  Correct approach: {result.correct_approach}")
    raw = result.raw or {}
    if "rr_trace" in raw:
        rt = raw["rr_trace"]
        print(f"\n  RR trace: depth={rt.get('depth')}, "
              f"iterations={rt.get('total_iterations')}, "
              f"compactions={rt.get('compactions')}, "
              f"timed_out={rt.get('timed_out')}")
    if "usage" in raw:
        u = raw["usage"]
        print(f"  Usage: {u.get('input_tokens')} in, "
              f"{u.get('output_tokens')} out, "
              f"{u.get('total_tokens')} total, "
              f"{u.get('requests')} requests")


# ---------------------------------------------------------------------------
# Demo 1: RRStep — agent got the wrong answer (simple)
# ---------------------------------------------------------------------------


def demo_wrong_answer():
    """RR analyzes a trace where the agent answered incorrectly."""
    section("Demo 1: RRStep — wrong answer")

    rr = RRStep(
        MODEL,
        config=RRConfig(max_requests=15, max_depth=0),
    )

    ctx = ACEStepContext(
        trace={
            "question": "What is the largest planet in our solar system by mass?",
            "ground_truth": "Jupiter",
            "feedback": "Incorrect. The correct answer is Jupiter, not Saturn.",
            "steps": [
                {
                    "role": "agent",
                    "reasoning": (
                        "The user is asking about the largest planet. "
                        "Saturn has those huge rings and is very large. "
                        "I'll go with Saturn."
                    ),
                    "answer": "Saturn",
                    "skill_ids": [],
                }
            ],
        },
        skillbook=SkillbookView(Skillbook()),
    )

    result_ctx = rr(ctx)
    print_result(result_ctx.reflections[0])


# ---------------------------------------------------------------------------
# Demo 2: RRStep — multi-step tool-use failure
# ---------------------------------------------------------------------------


def demo_tool_failure():
    """RR analyzes a trace with tool-use errors."""
    section("Demo 2: RRStep — tool-use failure trace")

    rr = RRStep(
        MODEL,
        config=RRConfig(max_requests=15, max_depth=0),
    )

    ctx = ACEStepContext(
        trace={
            "question": "What's the current weather in Tokyo?",
            "ground_truth": '{"temp_c": 22, "condition": "partly cloudy", "humidity": 65}',
            "feedback": (
                "Failed. Agent called the weather API with 'Tokio' (misspelled) "
                "and got a 404 error, then guessed instead of retrying."
            ),
            "steps": [
                {
                    "role": "agent",
                    "reasoning": (
                        "I need to call the weather API for Tokyo. "
                        "Let me use get_weather(city='Tokio')."
                    ),
                    "answer": "Error: 404 - City 'Tokio' not found",
                    "skill_ids": [],
                },
                {
                    "role": "agent",
                    "reasoning": (
                        "The API returned an error. I'll estimate based on "
                        "general knowledge — Tokyo is warm in summer."
                    ),
                    "answer": "It's probably around 28C and sunny in Tokyo.",
                    "skill_ids": [],
                },
            ],
        },
        skillbook=SkillbookView(Skillbook()),
    )

    result_ctx = rr(ctx)
    print_result(result_ctx.reflections[0])


# ---------------------------------------------------------------------------
# Demo 3: Real benchmark trace (if available)
# ---------------------------------------------------------------------------


def demo_real_trace():
    """RR analyzes a real benchmark trace."""
    section("Demo 3: Real benchmark trace")

    traces_path = _root / "ace-eval" / "results" / "benchmarks" / "bench_20260314_154608" / "benchmark" / "traces.json"
    if not traces_path.exists():
        print("  Benchmark traces not found, skipping.")
        return

    data = json.loads(traces_path.read_text())
    # Find a failed trace (reward=0)
    trace_dict = None
    for key, val in data.items():
        for trial in val.get("trials", []):
            if trial.get("reward", 1.0) == 0.0 and trial.get("trace"):
                trace_dict = trial["trace"]
                print(f"  Using trace: task {key}, question: {trace_dict.get('question', '')[:100]}...")
                break
        if trace_dict:
            break

    if not trace_dict:
        print("  No failed traces found, skipping.")
        return

    rr = RRStep(
        MODEL,
        config=RRConfig(max_requests=20, max_depth=0),
    )

    ctx = ACEStepContext(
        trace=trace_dict,
        skillbook=SkillbookView(Skillbook()),
    )

    result_ctx = rr(ctx)
    print_result(result_ctx.reflections[0])


# ---------------------------------------------------------------------------
# Demo 4: Batch traces with recursion
# ---------------------------------------------------------------------------


def demo_batch_recursion():
    """RR analyzes multiple traces using recurse tool."""
    section("Demo 4: Batch traces with recursion (depth=1)")

    traces_path = _root / "ace-eval" / "results" / "benchmarks" / "bench_20260314_154608" / "benchmark" / "traces.json"
    if not traces_path.exists():
        print("  Benchmark traces not found, skipping.")
        return

    data = json.loads(traces_path.read_text())
    # Collect first 3 failed traces as batch items
    batch_items = []
    for key, val in data.items():
        for trial in val.get("trials", []):
            if trial.get("reward", 1.0) == 0.0 and trial.get("trace"):
                t = trial["trace"]
                batch_items.append({
                    "task_id": f"task_{key}",
                    "question": t.get("question", ""),
                    "feedback": t.get("feedback", ""),
                    "trace": t,
                })
                if len(batch_items) >= 3:
                    break
        if len(batch_items) >= 3:
            break

    if len(batch_items) < 2:
        print(f"  Only {len(batch_items)} failed traces found, need at least 2. Skipping.")
        return

    print(f"  Batch: {len(batch_items)} failed traces")
    for bi in batch_items:
        print(f"    - {bi['task_id']}: {bi['question'][:80]}...")

    rr = RRStep(
        MODEL,
        config=RRConfig(
            max_requests=30,
            max_depth=1,  # allow one level of recursion
        ),
    )

    ctx = ACEStepContext(
        trace={
            "question": "Analyze these failed agent traces and extract common patterns",
            "batch_items": batch_items,
            "item_ids": [bi["task_id"] for bi in batch_items],
        },
        skillbook=SkillbookView(Skillbook()),
    )

    result_ctx = rr(ctx)
    for i, ref in enumerate(result_ctx.reflections):
        print(f"\n  --- Reflection {i} ---")
        print_result(ref)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RR Demo")
    parser.add_argument("--demo", type=int, default=0,
                        help="Run specific demo (1-4), 0=all")
    args = parser.parse_args()

    print(f"Model: {MODEL}")

    demos = {
        1: demo_wrong_answer,
        2: demo_tool_failure,
        3: demo_real_trace,
        4: demo_batch_recursion,
    }

    if args.demo:
        demos[args.demo]()
    else:
        for d in demos.values():
            d()

    section("Done")
