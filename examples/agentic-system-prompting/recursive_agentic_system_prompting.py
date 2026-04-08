#!/usr/bin/env python3
"""
Agentic System Prompting — Offline Trace Analysis

Feeds pre-recorded agent traces through TraceAnalyser (offline mode) to
extract reusable strategies into a skillbook.

Each trace file is loaded as a traces-format dict and passed directly
to RRStep via a thin adapter step, so the sandbox
receives the full conversation data.

TraceAnalyser handles the rest of the learning-tail pipeline:
    [RRTraceStep] → UpdateStep → ApplyStep

Usage:
    python recursive_agentic_system_prompting.py /path/to/traces
    python recursive_agentic_system_prompting.py /path/to/traces --model gpt-4o
    python recursive_agentic_system_prompting.py /path/to/traces --input-skillbook existing.json
    python recursive_agentic_system_prompting.py /path/to/traces --epochs 2

Options:
    traces_dir              Path to directory containing .json, .md, or .toon trace files
    --model, -m             LLM model for analysis (default: bedrock/us.anthropic.claude-sonnet-4-6)
    --threshold, -t         Deduplication similarity threshold 0.0-1.0 (default: 0.7)
    --epochs, -e            Number of passes over all traces (default: 1)
    --input-skillbook, -i   Path to existing skillbook to continue from
    --output-dir, -o        Output directory for results (default: script directory)
"""

import argparse
import json
import logging
import os
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Show RR iteration progress
_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
)
_logger = logging.getLogger("ace.steps.rr")
_logger.setLevel(logging.DEBUG)
_logger.addHandler(_handler)

from pipeline import Pipeline

from ace import TraceAnalyser, SkillManager, Skillbook
from ace.steps.rr import RRStep, RRConfig
from ace.core.context import ACEStepContext
from ace.deduplication import DeduplicationManager
from ace.protocols.deduplication import DeduplicationConfig
from ace.implementations.prompts import wrap_skillbook_for_external_agent
from ace.steps import UpdateStep, ApplyStep, DeduplicateStep
from ace.steps.rr.prompts import REFLECTOR_RECURSIVE_PROMPT


# ---------------------------------------------------------------------------
# Adapter step: normalises raw traces into the dict format RRStep expects.
# ---------------------------------------------------------------------------
class RRTraceStep:
    """Bridge between TraceAnalyser's per-trace context and RRStep.

    TraceAnalyser places the raw trace on ``ctx.trace``.  RRStep.__call__
    expects a traces-format dict with a ``steps`` key.  This adapter
    normalises the trace and delegates to ``RRStep.__call__``.
    """

    requires = frozenset({"trace", "skillbook"})
    provides = frozenset({"reflection"})

    def __init__(self, rr: RRStep) -> None:
        self.rr = rr

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        trace = ctx.trace
        # If the trace is already a traces-format dict, pass it through.
        # Otherwise wrap it so the sandbox can access it via traces["steps"].
        if isinstance(trace, dict) and "steps" in trace:
            traces_dict = trace
        else:
            traces_dict = {
                "question": str(trace.get("id", "")) if isinstance(trace, dict) else "",
                "steps": [trace],
            }
        return self.rr(ctx.replace(trace=traces_dict))


def load_traces(traces_dir: Path) -> Dict[str, Any]:
    """Load all trace files into a single batch trace dict.

    All files are combined into one traces-format dict so the REPL agent
    receives every conversation at once and can analyze cross-trace patterns.
    """
    if not traces_dir.exists():
        print(f"Directory not found: {traces_dir}")
        return {}

    steps: List[Dict[str, Any]] = []
    for ext in ("*.json", "*.md", "*.toon"):
        for file_path in sorted(traces_dir.glob(ext)):
            try:
                raw = file_path.read_text(encoding="utf-8")
                content = json.loads(raw) if file_path.suffix == ".json" else raw
                steps.append(
                    {
                        "role": "conversation",
                        "id": file_path.name,
                        "content": content,
                    }
                )
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")

    print(f"Loaded {len(steps)} traces")
    if not steps:
        return {}

    return {
        "question": f"Analyze {len(steps)} conversation traces",
        "ground_truth": None,
        "feedback": None,
        "steps": steps,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Offline trace analysis — extract strategies into a skillbook"
    )
    parser.add_argument(
        "traces_dir", type=Path, help="Directory containing trace files"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="bedrock/eu.anthropic.claude-sonnet-4-6",
        help="LLM model for analysis",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.7,
        help="Deduplication similarity threshold (0.0-1.0)",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=1, help="Number of passes over all traces"
    )
    parser.add_argument(
        "-i", "--input-skillbook", type=Path, default=None, help="Existing skillbook"
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None, help="Output directory"
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY required for deduplication embeddings!")
        return

    # Load all traces into a single batch dict
    batch_trace = load_traces(args.traces_dir)
    if not batch_trace:
        print(f"\nAdd .json, .md, or .toon trace files to {args.traces_dir}/")
        return
    n_traces = len(batch_trace["steps"])

    # Skillbook (existing or empty)
    skillbook = Skillbook()
    if args.input_skillbook and args.input_skillbook.exists():
        skillbook = Skillbook.load_from_file(str(args.input_skillbook))
        print(f"Loaded skillbook: {len(skillbook.skills())} skills")

    # Build PydanticAI-backed roles directly from model strings
    rr = RRStep(
        args.model,
        config=RRConfig(
            subagent_model="bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0",
            max_iterations=60,
            max_requests=60,
        ),
        prompt_template=REFLECTOR_RECURSIVE_PROMPT,
    )
    skill_manager = SkillManager(args.model)
    dedup = DeduplicationManager(
        DeduplicationConfig(
            enabled=True,
            similarity_threshold=args.threshold,
            embedding_model="text-embedding-3-small",
        )
    )

    # Build pipeline: RRTraceStep → Update → Apply → Dedup
    steps: list[Any] = [RRTraceStep(rr)]
    steps.extend(
        [
            UpdateStep(skill_manager),
            ApplyStep(skillbook),
            DeduplicateStep(dedup, skillbook),
        ]
    )
    analyser = TraceAnalyser(pipeline=Pipeline(steps), skillbook=skillbook)

    print(
        f"\nStarting analysis: {n_traces} traces (single batch), "
        f"epochs={args.epochs}, model={args.model}"
    )
    start = datetime.now()

    # Run — single batch trace through the pipeline
    results = analyser.run([batch_trace], epochs=args.epochs)

    # Surface any pipeline errors (the pipeline catches exceptions silently)
    failed = [r for r in results if r.error is not None]
    if failed:
        print(f"\n{len(failed)}/{len(results)} traces FAILED:")
        for r in failed:
            print(f"  - {r.failed_at}: {r.error}")

    duration = (datetime.now() - start).total_seconds()

    # Save results
    output_dir = args.output_dir or Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_skillbook = output_dir / f"skillbook_{timestamp}.json"
    analyser.save(str(output_skillbook))

    skills = analyser.skillbook.skills()
    print(f"\nCompleted in {duration:.1f}s")
    print(f"Analyzed: {n_traces} traces (single batch) × {args.epochs} epoch(s)")
    print(f"Generated: {len(skills)} skills")
    print(f"Saved to: {output_skillbook}")

    # Markdown export
    output_md = output_dir / f"skills_{timestamp}.md"
    with open(output_md, "w") as f:
        for section, section_skills in groupby(
            sorted(skills, key=lambda s: s.section), key=lambda s: s.section
        ):
            f.write(f"## {section}\n\n")
            for skill in section_skills:
                f.write(f"- {skill.content}\n")
                if skill.justification:
                    f.write(f"  Justification: {skill.justification}\n")
                if skill.evidence:
                    f.write(f"  Evidence: {skill.evidence}\n")
            f.write("\n")
    print(f"Skills: {output_md}")

    if skills:
        print("\nTop skills:")
        for i, skill in enumerate(
            sorted(skills, key=lambda s: s.helpful, reverse=True)[:5], 1
        ):
            print(f"  {i}. [{skill.section}] {skill.content[:80]}...")

    # External agent injection
    injection = wrap_skillbook_for_external_agent(analyser.skillbook)
    if injection:
        output_injection = output_dir / f"external_agent_injection_{timestamp}.txt"
        with open(output_injection, "w") as f:
            f.write(injection)
        print(f"External agent injection: {output_injection}")


if __name__ == "__main__":
    main()
