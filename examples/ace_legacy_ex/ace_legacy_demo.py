#!/usr/bin/env python3
"""
ace_legacy demo — offline + online ACE learning.

Offline:  ReplayAgent replays recorded traces → Reflector + SkillManager learn.
Online:   Agent generates new answers using the skillbook → learns from feedback.

Both use the same ACE class — the only difference is which agent you pass in.

Usage:
    python examples/ace_legacy_demo.py                    # both (default)
    python examples/ace_legacy_demo.py --mode offline     # traces only
    python examples/ace_legacy_demo.py --mode online      # live generation only
    python examples/ace_legacy_demo.py --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from ace_legacy import (
    ACE,
    Agent,
    PromptManager,
    Reflector,
    ReplayAgent,
    Sample,
    SimpleEnvironment,
    Skillbook,
    SkillManager,
)
from ace_legacy.llm_providers import LiteLLMClient


# ── Data ────────────────────────────────────────────────────────────────────


def build_offline_traces() -> list[Sample]:
    """Recorded traces — responses baked into metadata."""
    return [
        Sample(
            question="What is the capital of Japan?",
            context="Answer with just the city name.",
            ground_truth="Tokyo",
            metadata={"response": "Tokyo"},
        ),
        Sample(
            question="What is 15% of 200?",
            context="Show your work, then give the number.",
            ground_truth="30",
            metadata={"response": "15% of 200 = 0.15 × 200 = 30"},
        ),
        Sample(
            question="Translate 'hello' to French.",
            context="One word answer.",
            ground_truth="bonjour",
            metadata={"response": "salut"},  # wrong
        ),
        Sample(
            question="What element has the chemical symbol 'Au'?",
            context="Answer with the element name.",
            ground_truth="Gold",
            metadata={"response": "Gold"},
        ),
        Sample(
            question="What is the square root of 144?",
            context="Just the number.",
            ground_truth="12",
            metadata={"response": "14"},  # wrong
        ),
    ]


def build_online_samples() -> list[Sample]:
    """New questions — the Agent generates answers live."""
    return [
        Sample(
            question="What is the capital of Australia?",
            context="Answer with just the city name.",
            ground_truth="Canberra",
        ),
        Sample(
            question="What is 25% of 80?",
            context="Just the number.",
            ground_truth="20",
        ),
        Sample(
            question="Translate 'goodbye' to Spanish.",
            context="One word answer.",
            ground_truth="adiós",
        ),
    ]


# ── Runners ─────────────────────────────────────────────────────────────────


def run_offline(llm: LiteLLMClient, pm: PromptManager, skillbook: Skillbook, epochs: int) -> None:
    """Offline: replay traces, learn from historical data."""
    print("=" * 50)
    print("OFFLINE — Learning from recorded traces")
    print("=" * 50)

    ace = ACE(
        skillbook=skillbook,
        agent=ReplayAgent(),
        reflector=Reflector(llm, prompt_template=pm.get_reflector_prompt()),
        skill_manager=SkillManager(llm, prompt_template=pm.get_skill_manager_prompt()),
    )

    traces = build_offline_traces()
    print(f"Traces: {len(traces)}, Epochs: {epochs}")
    print(f"LLM calls: ~{len(traces) * epochs * 2}\n")

    results = ace.run(traces, SimpleEnvironment(), epochs=epochs)

    correct = sum(1 for r in results if r.environment_result.metrics.get("correct", 0))
    print(f"\nOffline: {correct}/{len(results)} correct")
    print(f"Skillbook: {skillbook.stats()['skills']} skills\n")


def run_online(llm: LiteLLMClient, pm: PromptManager, skillbook: Skillbook) -> None:
    """Online: agent generates answers live, learns from feedback."""
    print("=" * 50)
    print("ONLINE — Agent generates + learns in real time")
    print("=" * 50)

    ace = ACE(
        skillbook=skillbook,
        agent=Agent(llm, prompt_template=pm.get_agent_prompt()),
        reflector=Reflector(llm, prompt_template=pm.get_reflector_prompt()),
        skill_manager=SkillManager(llm, prompt_template=pm.get_skill_manager_prompt()),
    )

    samples = build_online_samples()
    print(f"Samples: {len(samples)}")
    print(f"LLM calls: ~{len(samples) * 3}")
    print(f"Starting skillbook: {skillbook.stats()['skills']} skills\n")

    results = ace.run(samples, SimpleEnvironment())

    correct = sum(1 for r in results if r.environment_result.metrics.get("correct", 0))
    print(f"\nOnline: {correct}/{len(results)} correct")
    print(f"Skillbook: {skillbook.stats()['skills']} skills\n")

    for r in results:
        status = "correct" if r.environment_result.metrics.get("correct", 0) else "wrong"
        print(f"  Q: {r.sample.question}")
        print(f"  A: {r.agent_output.final_answer} [{status}]\n")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="ace_legacy demo")
    parser.add_argument("--model", default="us.anthropic.claude-haiku-4-5-20251001-v1:0")
    parser.add_argument("--mode", choices=["offline", "online", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    llm = LiteLLMClient(model=args.model)
    pm = PromptManager()
    skillbook = Skillbook()

    print(f"Model: {args.model}\n")

    if args.mode in ("offline", "both"):
        run_offline(llm, pm, skillbook, args.epochs)

    if args.mode in ("online", "both"):
        run_online(llm, pm, skillbook)

    print("=" * 50)
    print("FINAL SKILLBOOK")
    print("=" * 50)
    for skill in skillbook.skills():
        print(f"  [{skill.id}] {skill.content}")

    out_path = "ace_legacy_skillbook.json"
    skillbook.save_to_file(out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
