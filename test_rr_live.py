#!/usr/bin/env python3
"""Live RR test — run RR against a real benchmark trace and observe behavior.

Instruments the sandbox to print every code execution and its output,
giving full visibility into how the RR iterates.

Usage:
    uv run python test_rr_live.py
    uv run python test_rr_live.py --task 12
    uv run python test_rr_live.py --all-failed
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from ace.steps.rr import RRStep, RRConfig, TraceSandbox
from ace.steps.rr.sandbox import ExecutionResult
from ace.core.context import ACEStepContext, SkillbookView
from ace.core.skillbook import Skillbook

# ── Config ──────────────────────────────────────────────────────────────

MODEL = os.getenv(
    "ACE_MODEL", "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
)
TRACES_FILE = Path(
    "ace-eval/results/e2e/run_7f757d765ba5/benchmark/traces.json"
)

# Only show RR-level logs, not every HTTP request
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s | %(message)s",
    datefmt="%H:%M:%S",
)
for lib in (
    "LiteLLM", "litellm", "httpx", "httpcore", "urllib3",
    "botocore", "boto3", "pydantic_ai",
):
    logging.getLogger(lib).setLevel(logging.WARNING)


# ── Sandbox instrumentation ─────────────────────────────────────────────

_orig_execute = TraceSandbox.execute
_iteration_counter = 0


def _instrumented_execute(self, code: str, timeout: float = 30.0) -> ExecutionResult:
    global _iteration_counter
    _iteration_counter += 1
    n = _iteration_counter

    print(f"\n{'━' * 70}")
    print(f"  EXECUTE_CODE  (iteration {n})")
    print(f"{'━' * 70}")
    for i, line in enumerate(code.strip().splitlines(), 1):
        print(f"  {i:3d} │ {line}")
    print(f"{'─' * 70}")

    start = time.time()
    result = _orig_execute(self, code, timeout)
    elapsed = time.time() - start

    if result.stdout:
        out = result.stdout.strip()
        lines = out.splitlines()
        if len(lines) > 40:
            for line in lines[:30]:
                print(f"  out │ {line}")
            print(f"  out │ ... ({len(lines) - 30} more lines)")
        else:
            for line in lines:
                print(f"  out │ {line}")
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            print(f"  err │ {line}")
    if result.exception:
        print(f"  EXC │ {type(result.exception).__name__}: {result.exception}")
    if result.final_value is not None:
        print(f"  FIN │ FINAL() called with keys: {list(result.final_value.keys()) if isinstance(result.final_value, dict) else type(result.final_value).__name__}")

    print(f"  [{elapsed:.2f}s]")
    return result


TraceSandbox.execute = _instrumented_execute


# ── Helpers ──────────────────────────────────────────────────────────────


def load_traces() -> dict:
    with open(TRACES_FILE) as f:
        return json.load(f)


def get_failed_tasks(data: dict) -> list[tuple[str, dict]]:
    failed = []
    for task_id, entry in data.items():
        trial = entry["trials"][0]
        if trial["reward"] == 0.0:
            failed.append((task_id, trial["trace"]))
    return failed


def run_rr_on_trace(task_id: str, trace: dict) -> None:
    global _iteration_counter
    _iteration_counter = 0

    print(f"\n{'=' * 70}")
    print(f"  Task {task_id} | reward=0.0 | {trace.get('outcome', '?')}")
    print(f"{'=' * 70}")
    print(f"  Question: {trace['question'][:200]}")
    print(f"  Feedback: {trace['feedback'][:200]}")
    print(f"  Answer:   {str(trace.get('answer', ''))[:200]}")
    print(f"  Reasoning: {len(trace.get('reasoning', ''))} chars")
    print(f"  Messages:  {len(trace.get('messages', []))} entries")
    print()

    rr = RRStep(
        MODEL,
        config=RRConfig(
            max_iterations=10,
            max_requests=20,
            enable_subagent=False,
            max_output_chars=10_000,
        ),
    )

    ctx = ACEStepContext(
        trace=trace,
        skillbook=SkillbookView(Skillbook()),
    )

    start = time.time()
    result_ctx = rr(ctx)
    elapsed = time.time() - start

    print(f"\n{'=' * 70}")
    print(f"  RESULT  (task {task_id}, {elapsed:.1f}s, {_iteration_counter} code executions)")
    print(f"{'=' * 70}")

    if not result_ctx.reflections:
        print("  No reflections produced!")
        return

    r = result_ctx.reflections[0]

    print(f"\n  Reasoning:\n    {r.reasoning[:600]}")
    print(f"\n  Key insight:\n    {r.key_insight}")
    if r.error_identification:
        print(f"\n  Error identification:\n    {r.error_identification}")
    if r.root_cause_analysis:
        print(f"\n  Root cause:\n    {r.root_cause_analysis}")
    if r.correct_approach:
        print(f"\n  Correct approach:\n    {r.correct_approach[:400]}")
    # Print raw metadata
    raw = r.raw or {}
    usage = raw.get("usage", {})
    rr_trace = raw.get("rr_trace", {})
    print(f"\n  Metadata:")
    print(f"    Tokens: {usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out")
    print(f"    LLM requests: {usage.get('requests', '?')}")
    print(f"    Tool iterations: {rr_trace.get('total_iterations', '?')}")
    print(f"    Timed out: {rr_trace.get('timed_out', '?')}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Live RR test")
    parser.add_argument("--task", type=str, help="Specific task ID to analyze")
    parser.add_argument(
        "--all-failed", action="store_true", help="Run on all failed tasks"
    )
    args = parser.parse_args()

    data = load_traces()
    failed = get_failed_tasks(data)
    print(f"Loaded {len(data)} tasks, {len(failed)} failed")
    print(f"Failed task IDs: {[t[0] for t in failed]}")
    print(f"Model: {MODEL}")

    if args.task:
        if args.task not in data:
            print(f"Task {args.task} not found. Available: {list(data.keys())}")
            sys.exit(1)
        trace = data[args.task]["trials"][0]["trace"]
        run_rr_on_trace(args.task, trace)
    elif args.all_failed:
        for task_id, trace in failed:
            run_rr_on_trace(task_id, trace)
    else:
        if not failed:
            print("No failed tasks found!")
            sys.exit(1)
        task_id, trace = failed[0]
        run_rr_on_trace(task_id, trace)


if __name__ == "__main__":
    main()
