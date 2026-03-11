#!/usr/bin/env python3
"""Demo: RR pipeline with Opik tracing.

Runs the Recursive Reflector on a sample trace and logs the REPL
iterations to Opik as a hierarchical trace.

Usage:
    uv run python examples/ace_next/rr_opik_demo.py
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))
load_dotenv(_root / ".env")

# Quiet noisy libraries
import logging

logging.basicConfig(level=logging.INFO, format="  %(name)s | %(message)s")
for name in ("LiteLLM", "litellm", "httpx", "httpcore"):
    logging.getLogger(name).setLevel(logging.WARNING)

from pipeline import Pipeline
from ace_next.providers.litellm import LiteLLMClient
from ace_next.rr import RRStep, RRConfig, RROpikStep
from ace_next.core.context import ACEStepContext, SkillbookView
from ace_next.core.skillbook import Skillbook

MODEL = os.getenv("ACE_MODEL", "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")


def main():
    print(f"Model: {MODEL}")

    # --- Build pipeline: RR + Opik ---
    llm = LiteLLMClient(model=MODEL, max_tokens=4096)
    rr = RRStep(llm, config=RRConfig(max_iterations=8, enable_subagent=False))
    rr_opik = RROpikStep(project_name="ace-rr-demo")

    pipe = Pipeline([rr, rr_opik])

    print(f"  Opik enabled: {rr_opik.enabled}")

    # --- Run ---
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

    result_ctx: ACEStepContext = pipe(ctx)  # type: ignore[assignment]

    # --- Print results ---
    assert len(result_ctx.reflections) > 0
    r = result_ctx.reflections[0]
    print(f"\n  Key insight: {r.key_insight}")
    print(f"  Learnings: {len(r.extracted_learnings)}")

    rr_trace = r.raw.get("rr_trace", {})
    print(f"  Iterations logged: {rr_trace.get('total_iterations', 0)}")
    print(f"  Subagent calls: {len(rr_trace.get('subagent_calls', []))}")

    if rr_opik.enabled:
        rr_opik.flush()
        print("  Trace sent to Opik. Check your dashboard.")
    else:
        print("\n  Opik not available — skipping trace upload.")
        print("  Install: uv add opik")
        print("  Set: OPIK_API_KEY and OPIK_WORKSPACE in .env")


if __name__ == "__main__":
    main()
