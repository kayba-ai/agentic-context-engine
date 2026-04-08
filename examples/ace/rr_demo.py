#!/usr/bin/env python3
"""Demo of the Recursive Reflector (RR) pipeline with a real LLM.

Shows the RR analyzing agent traces, iterating in its Python REPL sandbox,
and producing structured learnings.  Requires an API key for LiteLLM.

Usage:
    # Default model (Claude Haiku):
    uv run python examples/ace/rr_demo.py

    # Custom model:
    ACE_MODEL=openai/gpt-4o-mini uv run python examples/ace/rr_demo.py

    # With Logfire tracing:
    #   from ace.observability import configure_logfire
    #   configure_logfire()
"""

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is importable
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
load_dotenv(_root / ".env")

from ace.steps.rr import RRConfig, RRStep, TraceSandbox, TraceContext, TraceStep
from ace.core.context import ACEStepContext, SkillbookView
from ace.core.skillbook import Skillbook

MODEL = os.getenv("ACE_MODEL", "anthropic/claude-haiku-4-5-20251001")

# Show what the RR is doing at each iteration
logging.basicConfig(
    level=logging.INFO,
    format="  %(name)s | %(message)s",
)
# Quiet the noisy libraries
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def section(name: str) -> None:
    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Demo 1: RRStep — agent got the wrong answer
# ---------------------------------------------------------------------------


def demo_wrong_answer():
    """RR analyzes a trace where the agent answered incorrectly."""
    section("Demo 1: RRStep — wrong answer")

    rr = RRStep(
        MODEL,
        config=RRConfig(max_iterations=8, enable_subagent=False),
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
    assert len(result_ctx.reflections) > 0
    result = result_ctx.reflections[0]

    print(f"\n  --- Result ---")
    print(f"  Reasoning: {result.reasoning[:200]}")
    print(f"  Key insight: {result.key_insight}")
    if result.error_identification:
        print(f"  Error: {result.error_identification}")
    if result.correct_approach:
        print(f"  Correct approach: {result.correct_approach}")
    print(f"  Learnings ({len(result.extracted_learnings)}):")
    for l in result.extracted_learnings:
        print(f"    - [{l.atomicity_score:.1f}] {l.learning}")
        if l.evidence:
            print(f"      Evidence: {l.evidence[:120]}")


# ---------------------------------------------------------------------------
# Demo 2: RRStep as pipeline step — multi-step tool-use failure
# ---------------------------------------------------------------------------


def demo_pipeline_step():
    """RR used as a pipeline step via __call__, analyzing a tool-use failure."""
    section("Demo 2: RRStep.__call__() — tool-use failure trace")

    rr = RRStep(
        MODEL,
        config=RRConfig(max_iterations=8, enable_subagent=False),
    )

    sb = Skillbook()
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
        skillbook=SkillbookView(sb),
    )

    result_ctx = rr(ctx)
    r = result_ctx.reflections[0]

    print(f"\n  --- Result ---")
    print(f"  Reasoning: {r.reasoning[:200]}")
    print(f"  Key insight: {r.key_insight}")
    if r.error_identification:
        print(f"  Error: {r.error_identification}")
    if r.root_cause_analysis:
        print(f"  Root cause: {r.root_cause_analysis}")
    print(f"  Learnings ({len(r.extracted_learnings)}):")
    for l in r.extracted_learnings:
        print(f"    - [{l.atomicity_score:.1f}] {l.learning}")


# ---------------------------------------------------------------------------
# Demo 3: TraceSandbox — run code against trace data directly
# ---------------------------------------------------------------------------


def demo_sandbox():
    """Use TraceSandbox standalone to show how the REPL environment works."""
    section("Demo 3: TraceSandbox — direct code execution")

    trace = TraceContext(
        steps=[
            TraceStep(
                0, "user_message", "Find flights from NYC to London under $500", ""
            ),
            TraceStep(
                1,
                "tool_call:search_flights",
                '{"from": "NYC", "to": "London", "max_price": 500}',
                '[{"airline": "BA", "price": 450}, {"airline": "AA", "price": 520}]',
            ),
            TraceStep(
                2,
                "agent_reasoning",
                "Found 2 results. BA at $450 is under budget. AA at $520 is over.",
                "",
            ),
            TraceStep(
                3,
                "tool_call:book_flight",
                '{"airline": "AA", "price": 520}',
                "Error: Price $520 exceeds budget of $500",
            ),
            TraceStep(4, "agent_response", "I've booked your AA flight for $520.", ""),
        ],
        raw_reasoning="Agent searched flights, found options, but booked the wrong one.",
    )

    sandbox = TraceSandbox(trace=trace)

    # Show the trace structure
    print("  Running: trace exploration code")
    result = sandbox.execute(
        """
print(trace.summary())
print()

errors = trace.get_errors()
print(f"Errors found: {len(errors)}")
for e in errors:
    print(f"  Step {e.index} [{e.action}]: {e.observation[:80]}")

print()
tool_calls = trace.get_actions("tool_call")
print(f"Tool calls: {len(tool_calls)}")
for t in tool_calls:
    print(f"  {t.action}: input={t.thought[:60]}")
    print(f"    output={t.observation[:80]}")
"""
    )
    print(f"  Output:\n{result.stdout}")

    # Show FINAL mechanism
    print("  Running: FINAL() call")
    sandbox.execute(
        """
FINAL({
    "reasoning": "Agent booked AA ($520) instead of BA ($450) despite it exceeding the $500 budget",
    "key_insight": "Always validate constraints before executing actions",
    "error_identification": "Booked over-budget flight when a cheaper option was available",
})
"""
    )
    print(f"  FINAL called: {sandbox.final_called}")
    print(f"  Final value keys: {list(sandbox.final_value.keys())}")
    print(f"  Key insight: {sandbox.final_value['key_insight']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Model: {MODEL}")

    demo_wrong_answer()
    demo_pipeline_step()
    demo_sandbox()

    section("All demos completed")
