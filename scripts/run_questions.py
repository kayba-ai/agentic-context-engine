#!/usr/bin/env python3
"""Run ACE adaptation on the sample questions and generate a report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ace import (  # noqa: E402
    AdapterStepResult,
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
class QuestionSample(Sample):
    """Adds a stable identifier to each sample."""

    sample_id: str = ""


class FireInvestigationEnvironment(TaskEnvironment):
    """Evaluates long-form answers using string similarity against ground truth."""

    def __init__(self, similarity_threshold: float = 0.7) -> None:
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        return SequenceMatcher(None, a.strip(), b.strip()).ratio()

    def evaluate(self, sample: QuestionSample, agent_output) -> EnvironmentResult:
        ground_truth = sample.ground_truth or ""
        prediction = agent_output.final_answer or ""
        score = self._similarity(prediction, ground_truth)
        status = "aligned" if score >= self.similarity_threshold else "divergent"
        feedback = (
            f"Similarity {score:.2%} -> {status}. "
            "If divergent, incorporate missing technical details from the reference answer."
        )
        metrics = {"similarity": score}
        return EnvironmentResult(
            feedback=feedback, ground_truth=ground_truth, metrics=metrics
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="/data/models/openai/gpt-oss-20b",
        help="Local directory that contains the model weights.",
    )
    parser.add_argument(
        "--questions",
        default=str(ROOT / "questions.json"),
        help="Path to the questions JSON file.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "reports" / "questions_report.md"),
        help="Path to write the markdown report.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default="2,3",
        help="Comma-separated CUDA device ids to expose (default: 2,3).",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of offline adaptation epochs."
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
        help="Sampling temperature (default deterministic).",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.7,
        help="Threshold used in the environment feedback.",
    )
    return parser.parse_args()


def load_questions(path: Path) -> List[QuestionSample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    samples: List[QuestionSample] = []
    for idx, entry in enumerate(data, start=1):
        question = entry["question"]
        answer = entry["answer"]
        samples.append(
            QuestionSample(
                sample_id=f"q{idx:02d}",
                question=question,
                context="请提供详细、结构化、可执行的结论。",
                ground_truth=answer,
                metadata={"reference_answer": answer},
            )
        )
    return samples


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def summarize_results(results: Iterable[AdapterStepResult]) -> Dict[str, float]:
    scores = [
        step.environment_result.metrics.get("similarity", 0.0) for step in results
    ]
    if not scores:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {"avg": mean(scores), "min": min(scores), "max": max(scores)}


def truncate(text: str, limit: int = 120) -> str:
    cleaned = " ".join(text.split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


def build_report(
    args: argparse.Namespace,
    results: List[AdapterStepResult],
    skillbook: Skillbook,
) -> str:
    stats = summarize_results(results)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines: List[str] = []
    lines.append("# Questions Test Report")
    lines.append("")
    lines.append(f"- Generated: {timestamp}")
    lines.append(f"- Model: `{args.model_path}`")
    lines.append(f"- CUDA devices: `{args.cuda_visible_devices}`")
    lines.append(f"- Epochs: {args.epochs}")
    lines.append(f"- Samples: {len(results) }")
    lines.append(
        f"- Similarity (avg/min/max): {stats['avg']:.2%} / "
        f"{stats['min']:.2%} / {stats['max']:.2%}"
    )
    lines.append("")
    lines.append("## Per-Question Results")
    lines.append("")
    lines.append("| # | Similarity | Question | Final Answer (truncated) |")
    lines.append("|---|------------|----------|--------------------------|")
    for step in results:
        score = step.environment_result.metrics.get("similarity", 0.0)
        question = truncate(step.sample.question)
        final_answer = truncate(step.agent_output.final_answer or "")
        lines.append(
            f"| {step.sample.sample_id} | {score:.2%} | {question} | {final_answer} |"
        )
    lines.append("")
    lines.append("## Detailed Findings")
    lines.append("")
    for step in results:
        score = step.environment_result.metrics.get("similarity", 0.0)
        lines.append(f"### {step.sample.sample_id} — Similarity {score:.2%}")
        lines.append("")
        lines.append("**Question**")
        lines.append("")
        lines.append(step.sample.question)
        lines.append("")
        lines.append("**Model Final Answer**")
        lines.append("")
        lines.append(step.agent_output.final_answer or "(empty)")
        lines.append("")
        lines.append("**Reference Answer**")
        lines.append("")
        lines.append(step.environment_result.ground_truth or "(none)")
        lines.append("")
        lines.append("**Environment Feedback**")
        lines.append("")
        lines.append(step.environment_result.feedback)
        lines.append("")
        lines.append("**Reflection Snapshot**")
        lines.append("")
        lines.append(json.dumps(step.reflection.raw, ensure_ascii=False, indent=2))
        lines.append("")
        lines.append("**SkillManager Operations**")
        lines.append("")
        lines.append(
            json.dumps(step.skill_manager_output.raw, ensure_ascii=False, indent=2)
        )
        lines.append("")
    lines.append("## Final Skillbook")
    lines.append("")
    lines.append(skillbook.as_prompt() or "(skillbook is empty)")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    questions_path = Path(args.questions)
    samples = load_questions(questions_path)

    print(f"Loaded {len(samples)} questions from {questions_path}.")
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

    environment = FireInvestigationEnvironment(
        similarity_threshold=args.similarity_threshold
    )

    print("Starting offline adaptation...")
    results = adapter.run(samples, environment, epochs=args.epochs)

    report_markdown = build_report(args, results, adapter.skillbook)
    output_path = Path(args.output)
    ensure_parent(output_path)
    output_path.write_text(report_markdown, encoding="utf-8")
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
