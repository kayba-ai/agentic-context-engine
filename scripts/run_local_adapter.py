#!/usr/bin/env python3
"""Run a minimal ACE offline adaptation loop with a local transformers model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ace import (
    SkillManager,
    EnvironmentResult,
    Agent,
    OfflineACE,
    Skillbook,
    Reflector,
    Sample,
    TaskEnvironment,
    TransformersLLMClient,
)


@dataclass
class SimpleSample(Sample):
    """Extends Sample with an identifier for logging."""

    sample_id: str | None = None


class SimpleQAEnvironment(TaskEnvironment):
    """Evaluates predictions by string equality against ground truth."""

    def evaluate(self, sample: Sample, agent_output) -> EnvironmentResult:
        ground_truth = sample.ground_truth or ""
        prediction = agent_output.final_answer
        correct = prediction.strip().lower() == ground_truth.strip().lower()
        feedback = (
            "Prediction matched ground truth."
            if correct
            else f"Expected '{ground_truth}' but model returned '{prediction}'."
        )
        return EnvironmentResult(
            feedback=feedback,
            ground_truth=ground_truth,
            metrics={"accuracy": 1.0 if correct else 0.0},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/data/models/openai/gpt-oss-20b",
        help="Local directory that contains the gpt-oss-20b weights.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="2,3",
        help="Comma-separated CUDA device ids to expose (default: 2,3).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate per call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print(
        f"Loading model from {args.model_path} on GPUs {args.cuda_visible_devices}..."
    )
    client = TransformersLLMClient(
        args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        torch_dtype="bfloat16",
        device_map="auto",
    )

    agent = Agent(client)
    reflector = Reflector(client)
    skill_manager = SkillManager(client)
    adapter = OfflineACE(
        skillbook=Skillbook(),
        agent=agent,
        reflector=reflector,
        skill_manager=skill_manager,
        max_refinement_rounds=3,
    )

    samples: List[Sample] = [
        SimpleSample(
            sample_id="demo-1",
            question="Answer the question by returning the digits 42. Respond strictly via the JSON schema.",
            context="The correct answer is 42.",
            ground_truth="42",
        ),
    ]
    environment = SimpleQAEnvironment()

    print("Starting offline adaptation with 1 sample...")
    results = adapter.run(samples, environment, epochs=1)

    for step, result in enumerate(results, start=1):
        print(f"\nStep {step}:")
        print(f"  Question: {result.sample.question}")
        print(f"  Model final answer: {result.agent_output.final_answer}")
        print(f"  Feedback: {result.environment_result.feedback}")
        print("  Reflection:")
        print(json.dumps(result.reflection.raw, ensure_ascii=False, indent=2))
        print("  SkillManager operations:")
        print(json.dumps(result.skill_manager_output.raw, ensure_ascii=False, indent=2))

    print("\nFinal skillbook:\n")
    print(adapter.skillbook.as_prompt() or "(skillbook is empty)")


if __name__ == "__main__":
    main()
