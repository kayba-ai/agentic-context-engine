#!/usr/bin/env python3
"""Stress test: RR analyzing 30 real benchmark traces at once.

Usage:
    uv run python examples/ace/rr_stress_test.py
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
load_dotenv(_root / ".env")

from ace.steps.rr import RRConfig, RRStep
from ace.core.context import ACEStepContext, SkillbookView
from ace.core.skillbook import Skillbook

MODEL = os.getenv("ACE_MODEL", "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")

logging.basicConfig(level=logging.INFO, format="  %(name)s | %(message)s")
for name in ("LiteLLM", "litellm", "httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.WARNING)


def load_traces(n: int = 30) -> list[dict]:
    """Load n traces from benchmark results, deduplicating by task."""
    traces_path = _root / "ace-eval" / "results" / "benchmarks" / "bench_20260314_154608" / "benchmark" / "traces.json"
    data = json.loads(traces_path.read_text())

    batch = []
    for key, val in data.items():
        for trial in val.get("trials", []):
            if trial.get("trace"):
                t = trial["trace"]
                batch.append({
                    "task_id": f"task_{key}_r{trial.get('reward', '?')}",
                    "question": t.get("question", ""),
                    "feedback": t.get("feedback", ""),
                    "trace": t,
                })
                if len(batch) >= n:
                    return batch
    return batch


def main():
    traces = load_traces(30)
    print(f"Model: {MODEL}")
    print(f"Loaded {len(traces)} traces")
    print(f"Total trace chars: {sum(len(t['trace'].get('reasoning', '')) for t in traces):,}")
    print()

    rr = RRStep(
        MODEL,
        config=RRConfig(
            max_requests=80,
            max_depth=1,       # allow recursion
            max_tokens=1_500_000,
        ),
    )

    ctx = ACEStepContext(
        trace={
            "question": "Analyze these agent traces from a customer service benchmark. Identify common failure patterns, categorize them, and extract actionable learnings.",
            "batch_items": traces,
            "item_ids": [t["task_id"] for t in traces],
        },
        skillbook=SkillbookView(Skillbook()),
    )

    print("Running RR...")
    t0 = time.time()
    result_ctx = rr(ctx)
    elapsed = time.time() - t0

    print(f"\n{'=' * 60}")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Reflections: {len(result_ctx.reflections)}")
    print(f"{'=' * 60}\n")

    for i, ref in enumerate(result_ctx.reflections):
        print(f"--- Reflection {i} ---")
        print(f"  Reasoning: {ref.reasoning[:200]}...")
        print(f"  Key insight: {ref.key_insight[:200] if ref.key_insight else '(none)'}")
        if ref.error_identification:
            print(f"  Error: {ref.error_identification[:200]}")
        if ref.root_cause_analysis:
            print(f"  Root cause: {ref.root_cause_analysis[:200]}")
        if ref.correct_approach:
            print(f"  Correct approach: {ref.correct_approach[:200]}")
        raw = ref.raw or {}
        if "rr_trace" in raw:
            rt = raw["rr_trace"]
            print(f"  RR trace: depth={rt.get('depth')}, iters={rt.get('total_iterations')}, "
                  f"compactions={rt.get('compactions')}, timed_out={rt.get('timed_out')}")
        if "usage" in raw:
            u = raw["usage"]
            print(f"  Usage: {u.get('total_tokens'):,} tokens, {u.get('requests')} requests")
        print()


if __name__ == "__main__":
    main()
