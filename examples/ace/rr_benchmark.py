#!/usr/bin/env python3
"""E2E benchmark: RR (PydanticAI) over 30 traces.

Generates 30 synthetic agent traces with known errors, runs the full
ACE learning pipeline (RRStep -> Tag -> Update -> Apply), and reports:
- Success rate (RR produced valid learnings)
- Skills extracted
- Timing

Usage:
    uv run python examples/ace/rr_benchmark.py
    ACE_MODEL=bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0 uv run python examples/ace/rr_benchmark.py
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

from ace.core.skillbook import Skillbook
from ace.implementations import SkillManager
from ace.steps.rr_step import RRConfig, RRStep
from ace.runners.trace_analyser import TraceAnalyser

MODEL = os.getenv("ACE_MODEL", "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")

logging.basicConfig(level=logging.WARNING, format="%(name)s | %(message)s")
logging.getLogger("ace.steps.rr").setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Synthetic traces — 30 traces with realistic agent errors
# ---------------------------------------------------------------------------

TRACES = [
    # Math errors (5)
    {"question": "What is 17 x 23?", "ground_truth": "391",
     "feedback": "Incorrect.",
     "steps": [{"role": "agent", "reasoning": "17x23 = 17x20 + 17x3 = 340 + 51 = 381", "answer": "381", "skill_ids": []}]},
    {"question": "What is 144 / 12?", "ground_truth": "12",
     "feedback": "Incorrect.",
     "steps": [{"role": "agent", "reasoning": "144/12 = 14", "answer": "14", "skill_ids": []}]},
    {"question": "What is 25% of 80?", "ground_truth": "20",
     "feedback": "Incorrect.",
     "steps": [{"role": "agent", "reasoning": "25% of 80 = 80/25 = 3.2", "answer": "3.2", "skill_ids": []}]},
    {"question": "What is sqrt(169)?", "ground_truth": "13",
     "feedback": "Incorrect.",
     "steps": [{"role": "agent", "reasoning": "sqrt(169) is about 14", "answer": "14", "skill_ids": []}]},
    {"question": "What is 2^10?", "ground_truth": "1024",
     "feedback": "Incorrect.",
     "steps": [{"role": "agent", "reasoning": "2^10 = 2x10 = 20", "answer": "20", "skill_ids": []}]},

    # Geography errors (5)
    {"question": "What is the capital of Australia?", "ground_truth": "Canberra",
     "feedback": "Incorrect. The capital is Canberra, not Sydney.",
     "steps": [{"role": "agent", "reasoning": "Sydney is the largest city, so it must be the capital.", "answer": "Sydney", "skill_ids": []}]},
    {"question": "What is the capital of Brazil?", "ground_truth": "Brasilia",
     "feedback": "Incorrect.",
     "steps": [{"role": "agent", "reasoning": "Sao Paulo is the biggest city.", "answer": "Sao Paulo", "skill_ids": []}]},
    {"question": "What is the capital of Turkey?", "ground_truth": "Ankara",
     "feedback": "Incorrect.",
     "steps": [{"role": "agent", "reasoning": "Istanbul is the most famous city.", "answer": "Istanbul", "skill_ids": []}]},
    {"question": "What is the capital of Myanmar?", "ground_truth": "Naypyidaw",
     "feedback": "Incorrect.",
     "steps": [{"role": "agent", "reasoning": "Yangon is the largest city.", "answer": "Yangon", "skill_ids": []}]},
    {"question": "What is the capital of Nigeria?", "ground_truth": "Abuja",
     "feedback": "Incorrect.",
     "steps": [{"role": "agent", "reasoning": "Lagos is the most well-known city.", "answer": "Lagos", "skill_ids": []}]},

    # Science errors (5)
    {"question": "What is the boiling point of water in Fahrenheit?", "ground_truth": "212",
     "feedback": "Incorrect.",
     "steps": [{"role": "agent", "reasoning": "Water boils at 100 degrees.", "answer": "100", "skill_ids": []}]},
    {"question": "How many chromosomes do humans have?", "ground_truth": "46",
     "feedback": "Incorrect.",
     "steps": [{"role": "agent", "reasoning": "Humans have 23 chromosomes.", "answer": "23", "skill_ids": []}]},
    {"question": "What is the speed of light in km/s?", "ground_truth": "299,792",
     "feedback": "Incorrect.",
     "steps": [{"role": "agent", "reasoning": "Speed of light is about 300,000 miles per second.", "answer": "300,000 miles/s", "skill_ids": []}]},
    {"question": "What is the atomic number of gold?", "ground_truth": "79",
     "feedback": "Incorrect.",
     "steps": [{"role": "agent", "reasoning": "Gold is Au, atomic number around 80.", "answer": "80", "skill_ids": []}]},
    {"question": "What planet is closest to the sun?", "ground_truth": "Mercury",
     "feedback": "Incorrect.",
     "steps": [{"role": "agent", "reasoning": "Venus is very hot so it must be closest.", "answer": "Venus", "skill_ids": []}]},

    # Correct answers (5) — RR should find nothing or minimal learnings
    {"question": "What is 2+2?", "ground_truth": "4",
     "feedback": "Correct.",
     "steps": [{"role": "agent", "reasoning": "2+2=4.", "answer": "4", "skill_ids": []}]},
    {"question": "What is the capital of France?", "ground_truth": "Paris",
     "feedback": "Correct.",
     "steps": [{"role": "agent", "reasoning": "The capital of France is Paris.", "answer": "Paris", "skill_ids": []}]},
    {"question": "What color is the sky?", "ground_truth": "Blue",
     "feedback": "Correct.",
     "steps": [{"role": "agent", "reasoning": "The sky appears blue due to Rayleigh scattering.", "answer": "Blue", "skill_ids": []}]},
    {"question": "How many days in a week?", "ground_truth": "7",
     "feedback": "Correct.",
     "steps": [{"role": "agent", "reasoning": "A week has 7 days.", "answer": "7", "skill_ids": []}]},
    {"question": "What is H2O?", "ground_truth": "Water",
     "feedback": "Correct.",
     "steps": [{"role": "agent", "reasoning": "H2O is the chemical formula for water.", "answer": "Water", "skill_ids": []}]},

    # Reasoning errors (5)
    {"question": "If all roses are flowers and some flowers fade quickly, do all roses fade quickly?",
     "ground_truth": "No, not necessarily.",
     "feedback": "Incorrect. 'Some flowers fade quickly' doesn't mean all roses do.",
     "steps": [{"role": "agent", "reasoning": "Roses are flowers, flowers fade, so roses fade.", "answer": "Yes", "skill_ids": []}]},
    {"question": "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?",
     "ground_truth": "$0.05",
     "feedback": "Incorrect. If ball=$0.10, bat=$1.10, total=$1.20.",
     "steps": [{"role": "agent", "reasoning": "$1.10 - $1.00 = $0.10.", "answer": "$0.10", "skill_ids": []}]},
    {"question": "Is a whale a fish?", "ground_truth": "No, a whale is a mammal.",
     "feedback": "Incorrect.",
     "steps": [{"role": "agent", "reasoning": "Whales live in water, so they are fish.", "answer": "Yes", "skill_ids": []}]},
    {"question": "Can you fold a piece of paper more than 7 times?",
     "ground_truth": "Yes, the record is 12 folds.",
     "feedback": "Incorrect.",
     "steps": [{"role": "agent", "reasoning": "It's physically impossible to fold paper more than 7 times.", "answer": "No", "skill_ids": []}]},
    {"question": "Does hot water freeze faster than cold water?",
     "ground_truth": "It can, under certain conditions (Mpemba effect).",
     "feedback": "Incorrect — the Mpemba effect is real.",
     "steps": [{"role": "agent", "reasoning": "Hot water has more energy so it takes longer to cool.", "answer": "No, never.", "skill_ids": []}]},

    # Tool-use errors (5)
    {"question": "Search for 'Python 3.12 release date'", "ground_truth": "October 2, 2023",
     "feedback": "Agent searched for wrong query.",
     "steps": [{"role": "agent", "reasoning": "I'll search for 'Python release'.", "answer": "Python 3.11 was released Oct 2022", "skill_ids": []},
               {"role": "agent", "reasoning": "Found a result about 3.11.", "answer": "October 2022", "skill_ids": []}]},
    {"question": "Convert 100 USD to EUR", "ground_truth": "~92 EUR",
     "feedback": "Agent used wrong conversion direction.",
     "steps": [{"role": "agent", "reasoning": "1 EUR = 1.08 USD, so 100*1.08=108.", "answer": "108 EUR", "skill_ids": []}]},
    {"question": "Get weather in Tokyo", "ground_truth": "22C, partly cloudy",
     "feedback": "Agent misspelled the city name.",
     "steps": [{"role": "agent", "reasoning": "I'll call weather API for Tokio.", "answer": "Error: 404 city not found", "skill_ids": []},
               {"role": "agent", "reasoning": "API failed, I'll guess.", "answer": "Probably warm and humid", "skill_ids": []}]},
    {"question": "Calculate the area of a circle with radius 5", "ground_truth": "78.54",
     "feedback": "Agent used diameter instead of radius.",
     "steps": [{"role": "agent", "reasoning": "Area = pi * r^2 = pi * 10^2 = 314.16", "answer": "314.16", "skill_ids": []}]},
    {"question": "List files in /home/user/docs", "ground_truth": "report.pdf, notes.txt, budget.xlsx",
     "feedback": "Agent listed wrong directory.",
     "steps": [{"role": "agent", "reasoning": "I'll list /home/user.", "answer": "Desktop/, docs/, Downloads/", "skill_ids": []}]},
]

assert len(TRACES) == 30, f"Expected 30 traces, got {len(TRACES)}"


# ---------------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------------

def main():
    print(f"Model: {MODEL}")
    print(f"Traces: {len(TRACES)}")
    print(f"{'=' * 60}")

    skillbook = Skillbook()
    rr = RRStep(
        MODEL,
        config=RRConfig(
            max_requests=20,
            max_iterations=10,
            timeout=15.0,
            enable_subagent=False,
        ),
    )
    sm = SkillManager(MODEL)

    analyser = TraceAnalyser.from_roles(
        reflector=rr,
        skill_manager=sm,
        skillbook=skillbook,
    )

    t0 = time.time()
    results = analyser.run(TRACES, epochs=1)
    elapsed = time.time() - t0

    # Report
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'=' * 60}")

    successes = sum(1 for r in results if r.error is None)
    failures = sum(1 for r in results if r.error is not None)
    print(f"\n  Traces processed: {len(results)}/{len(TRACES)}")
    print(f"  Successes: {successes}")
    print(f"  Failures: {failures}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/len(TRACES):.1f}s/trace)")

    if failures > 0:
        print(f"\n  Errors:")
        for r in results:
            if r.error is not None:
                print(f"    - {r.error}")

    skills = skillbook.skills()
    print(f"\n  Skills extracted: {len(skills)}")
    for s in skills[:20]:
        print(f"    [{s.id}] {s.content[:80]}")
    if len(skills) > 20:
        print(f"    ... and {len(skills) - 20} more")

    # Save results
    out_dir = _root / "examples" / "ace" / "benchmark_output"
    out_dir.mkdir(exist_ok=True)
    skillbook.save_to_file(str(out_dir / "skillbook.json"))
    print(f"\n  Skillbook saved to: {out_dir / 'skillbook.json'}")

    # Summary
    print(f"\n{'=' * 60}")
    rate = successes / len(results) * 100 if results else 0
    print(f"  Success rate: {rate:.0f}%")
    print(f"  Skills learned: {len(skills)}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    return 0 if rate >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
