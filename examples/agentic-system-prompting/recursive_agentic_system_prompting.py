#!/usr/bin/env python3
"""
Agentic System Prompting — Offline Trace Analysis

Feeds pre-recorded agent traces through TraceAnalyser (offline mode) to
extract reusable strategies into a skillbook.

Each trace file is loaded as a traces-format dict and passed directly
to RRStep.reflect(traces=...) via a thin adapter step, so the sandbox
receives the full conversation data.

TraceAnalyser handles the rest of the learning-tail pipeline:
    [RRTraceStep] → TagStep → UpdateStep → ApplyStep

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
_logger = logging.getLogger("ace_next.rr")
_logger.setLevel(logging.DEBUG)
_logger.addHandler(_handler)

from pipeline import Pipeline

from ace_next import TraceAnalyser, SkillManager, Skillbook
from ace_next.rr import RRStep, RRConfig
from ace_next.core.context import ACEStepContext
from ace_next.deduplication import DeduplicationManager
from ace_next.protocols.deduplication import DeduplicationConfig
from ace_next.providers.litellm import LiteLLMClient, LiteLLMConfig
from ace_next.implementations.prompts import wrap_skillbook_for_external_agent
from ace_next.steps import TagStep, UpdateStep, ApplyStep, DeduplicateStep
from ace.reflector.prompts_rr_v5 import REFLECTOR_RECURSIVE_V5_PROMPT


# ---------------------------------------------------------------------------
# Adapter step: passes raw trace as the `traces` kwarg to RRStep.reflect()
# so the sandbox receives the full conversation data.
# ---------------------------------------------------------------------------
class RRTraceStep:
    """Bridge between TraceAnalyser's per-trace context and RRStep.reflect().

    TraceAnalyser places the raw trace on ``ctx.trace``.  RRStep.reflect()
    needs it as the ``traces`` keyword argument to populate the sandbox
    ``traces`` variable and prompt metadata correctly.
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
        reflection = self.rr.reflect(
            traces=traces_dict,
            skillbook=ctx.skillbook,
        )
        return ctx.replace(reflection=reflection)


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


def setup_opik() -> None:
    """Configure Opik tracing if credentials are available.

    Per-call project routing is handled via ``metadata.opik.project_name``
    in each LiteLLMClient's ``extra_params``.  The callback registered here
    acts as the fallback default.
    """
    opik_key = os.environ.get("OPIK_API_KEY")
    if not opik_key:
        print("OPIK_API_KEY not found — Opik tracing disabled")
        return

    try:
        import opik

        opik.configure(
            api_key=opik_key,
            workspace=os.environ.get("OPIK_WORKSPACE", "default"),
        )
    except (ImportError, Exception) as e:
        print(f"Opik init failed: {e}")
        return

    try:
        from ace_next.steps.opik import register_opik_litellm_callback

        # Register a single callback; per-call metadata overrides project
        if register_opik_litellm_callback(project_name="REPL-agent"):
            print("Opik tracing enabled (REPL-agent / REPL-Subagent)")
    except ImportError:
        print("Opik LiteLLM callback not available")


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
        default="bedrock/us.anthropic.claude-sonnet-4-6",
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

    setup_opik()

    # LLM clients — separate Opik projects for main REPL vs sub-agent
    llm = LiteLLMClient(
        config=LiteLLMConfig(
            model=args.model,
            max_tokens=8192,
            temperature=1,
            extra_params={"metadata": {"opik": {"project_name": "REPL-agent"}}},
        )
    )
    subagent_llm = LiteLLMClient(
        config=LiteLLMConfig(
            model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            max_tokens=4096,
            temperature=0.3,
            extra_params={"metadata": {"opik": {"project_name": "REPL-Subagent"}}},
        )
    )
    rr = RRStep(
        llm=llm,
        config=RRConfig(subagent_model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        prompt_template=REFLECTOR_RECURSIVE_V5_PROMPT,
        subagent_llm=subagent_llm,
    )
    skill_manager = SkillManager(llm=llm)
    dedup = DeduplicationManager(
        DeduplicationConfig(
            enabled=True,
            similarity_threshold=args.threshold,
            embedding_model="text-embedding-3-small",
        )
    )

    # Build pipeline manually: RRTraceStep replaces ReflectStep to pass
    # traces correctly, then the standard learning-tail steps follow.
    steps = [
        RRTraceStep(rr),
        TagStep(skillbook),
        UpdateStep(skill_manager),
        ApplyStep(skillbook),
        DeduplicateStep(dedup, skillbook),
    ]
    analyser = TraceAnalyser(pipeline=Pipeline(steps), skillbook=skillbook)

    print(
        f"\nStarting analysis: {n_traces} traces (single batch), "
        f"epochs={args.epochs}, model={args.model}"
    )
    start = datetime.now()

    # Run — single batch trace through RRTraceStep → Tag → Update → Apply → Dedup
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
