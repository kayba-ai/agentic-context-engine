#!/usr/bin/env python3
"""
Run Recursive Reflector (RR) on CAR-bench traces using batched analysis.

Reads batch_assignment.json to split 129 traces into 6 semantically coherent
batches, runs RRStep on each batch independently, then merges all per-batch
skillbooks and deduplicates via embedding similarity.

Design decisions:
- Single run (no multi-run consensus)
- No token budget (unbounded)
- Task types never mixed within a batch
- Merge strategy: concatenate + embedding dedup (Option A)
  Caveat: this only removes near-duplicates, does not consolidate
  related-but-different skills or resolve cross-batch contradictions.
  Future work: try Opus compression (Option C) on the merged result.

Usage:
    uv run python scripts/run_rr_car_bench.py
    uv run python scripts/run_rr_car_bench.py --model bedrock/us.anthropic.claude-sonnet-4-6
    uv run python scripts/run_rr_car_bench.py --batch base_vehicle_climate  # single batch
    uv run python scripts/run_rr_car_bench.py --dry-run  # show batch plan only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional

# Project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load env from known location (worktree can't find .env via find_dotenv)
ENV_FILE = Path("/scratch/tzerweck/other/Kayba/.env")
try:
    from dotenv import load_dotenv

    if ENV_FILE.exists():
        load_dotenv(ENV_FILE, override=True)
    else:
        load_dotenv()  # fallback
except ImportError:
    pass

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
# Adapter step (same as recursive_agentic_system_prompting.py)
# ---------------------------------------------------------------------------
class RRTraceStep:
    """Bridge between TraceAnalyser's per-trace context and RRStep."""

    requires = frozenset({"trace", "skillbook"})
    provides = frozenset({"reflection"})

    def __init__(self, rr: RRStep) -> None:
        self.rr = rr

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        trace = ctx.trace
        if isinstance(trace, dict) and "steps" in trace:
            traces_dict = trace
        else:
            traces_dict = {
                "question": str(trace.get("id", "")) if isinstance(trace, dict) else "",
                "steps": [trace],
            }
        return self.rr(ctx.replace(trace=traces_dict))


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------
TRACES_DIR = ROOT / "results" / "traces_car_bench"
BATCH_FILE = TRACES_DIR / "batch_assignment.json"


def load_batch_assignment() -> Dict[str, Any]:
    """Load the batch assignment JSON."""
    with open(BATCH_FILE) as f:
        return json.load(f)


def load_traces_for_batch(trace_ids: List[str]) -> Dict[str, Any]:
    """Load .toon trace files for the given task IDs into a single batch dict."""
    steps: List[Dict[str, Any]] = []
    for task_id in trace_ids:
        # Find the file (task_id -> {task_id}_trial0.toon)
        pattern = f"{task_id}_trial*.toon"
        matches = sorted(TRACES_DIR.glob(pattern))
        if not matches:
            print(f"  WARNING: No trace file for {task_id}")
            continue
        file_path = matches[0]  # Take first trial
        try:
            raw = file_path.read_text(encoding="utf-8")
            steps.append(
                {
                    "role": "conversation",
                    "id": task_id,
                    "content": raw,
                }
            )
        except Exception as e:
            print(f"  ERROR reading {file_path.name}: {e}")

    return {
        "question": f"Analyze {len(steps)} conversation traces",
        "ground_truth": None,
        "feedback": None,
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# Per-batch RR analysis
# ---------------------------------------------------------------------------
def run_batch(
    batch_name: str,
    batch_info: Dict[str, Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> Optional[Skillbook]:
    """Run RR analysis on a single batch. Returns the per-batch skillbook."""
    trace_ids = batch_info["trace_ids"]
    print(f"\n{'='*60}")
    print(f"Batch: {batch_name}")
    print(f"  Type: {batch_info['task_type']}")
    print(f"  Domains: {batch_info['domains']}")
    print(f"  Traces: {len(trace_ids)}")
    print(f"{'='*60}")

    # Load traces
    batch_trace = load_traces_for_batch(trace_ids)
    n_traces = len(batch_trace["steps"])
    if n_traces == 0:
        print("  No traces loaded, skipping.")
        return None

    print(f"  Loaded {n_traces} traces")

    # Fresh skillbook per batch
    skillbook = Skillbook()

    # LLM clients
    llm = LiteLLMClient(
        config=LiteLLMConfig(
            model=args.model,
            max_tokens=8192,
            temperature=1,
        )
    )
    subagent_llm = LiteLLMClient(
        config=LiteLLMConfig(
            model=args.subagent_model,
            max_tokens=4096,
            temperature=0.3,
        )
    )

    rr = RRStep(
        llm=llm,
        config=RRConfig(
            subagent_model=args.subagent_model,
            max_iterations=args.max_iterations,
            max_llm_calls=args.max_llm_calls,
        ),
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

    # Pipeline: RRTraceStep → Tag → Update → Apply → Dedup
    pipeline_steps: list[Any] = [
        RRTraceStep(rr),
        TagStep(skillbook),
        UpdateStep(skill_manager),
        ApplyStep(skillbook),
        DeduplicateStep(dedup, skillbook),
    ]
    analyser = TraceAnalyser(pipeline=Pipeline(pipeline_steps), skillbook=skillbook)

    start = datetime.now()
    results = analyser.run([batch_trace], epochs=1)
    duration = (datetime.now() - start).total_seconds()

    # Check for errors
    failed = [r for r in results if r.error is not None]
    if failed:
        print(f"  {len(failed)}/{len(results)} pipeline errors:")
        for r in failed:
            print(f"    - {r.failed_at}: {r.error}")

    skills = analyser.skillbook.skills()
    print(f"  Completed in {duration:.1f}s — {len(skills)} skills")

    # Save per-batch outputs
    batch_dir = output_dir / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    analyser.save(str(batch_dir / "skillbook.json"))

    # Markdown export
    md_path = batch_dir / "skills.md"
    with open(md_path, "w") as f:
        f.write(f"# {batch_name}\n\n")
        f.write(f"Task type: {batch_info['task_type']}\n")
        f.write(f"Domains: {', '.join(batch_info['domains'])}\n")
        f.write(f"Traces: {n_traces} | Skills: {len(skills)} | Time: {duration:.1f}s\n\n")
        for section, section_skills in groupby(
            sorted(skills, key=lambda s: s.section), key=lambda s: s.section
        ):
            f.write(f"## {section}\n\n")
            for skill in section_skills:
                f.write(f"- {skill.content}\n")
                if skill.justification:
                    f.write(f"  _Justification: {skill.justification}_\n")
            f.write("\n")
    print(f"  Saved: {batch_dir}/")

    return analyser.skillbook


# ---------------------------------------------------------------------------
# Merge + dedup
# ---------------------------------------------------------------------------
def merge_skillbooks(
    skillbooks: Dict[str, Skillbook],
    threshold: float,
) -> Skillbook:
    """Merge per-batch skillbooks into one, then deduplicate.

    Strategy: concatenate all skills (preserving sections), then run
    embedding-based deduplication across the merged set.

    Caveat: this only removes near-duplicates. It does not consolidate
    related-but-different skills across batches or resolve contradictions.
    Consider Opus compression (Option C) as a future enhancement.
    """
    merged = Skillbook()

    total_before = 0
    for batch_name, sb in skillbooks.items():
        for skill in sb.skills():
            # Prefix section with batch source for traceability
            section = f"[{batch_name}] {skill.section}"
            merged.add_skill(
                section=section,
                content=skill.content,
                metadata={
                    "helpful": skill.helpful,
                    "harmful": skill.harmful,
                    "neutral": skill.neutral,
                },
                justification=skill.justification,
                evidence=skill.evidence,
            )
            total_before += 1

    print(f"\nMerged: {total_before} skills from {len(skillbooks)} batches")

    # Deduplicate using embedding similarity directly
    # (DeduplicationManager works through the SkillManager pipeline;
    #  for standalone merge we use SimilarityDetector to find pairs
    #  and remove the shorter skill from each pair.)
    from ace_next.deduplication.detector import SimilarityDetector

    config = DeduplicationConfig(
        enabled=True,
        similarity_threshold=threshold,
        embedding_model="text-embedding-3-small",
        within_section_only=False,  # cross-batch dedup
    )
    detector = SimilarityDetector(config)
    detector.ensure_embeddings(merged)
    pairs = detector.detect_similar_pairs(merged, threshold=threshold)
    print(f"Found {len(pairs)} similar pairs (threshold={threshold})")

    to_remove: set[str] = set()
    for skill_a, skill_b, sim in pairs:
        if skill_a.id in to_remove or skill_b.id in to_remove:
            continue
        # Keep the longer (more detailed) skill
        if len(skill_a.content) >= len(skill_b.content):
            to_remove.add(skill_b.id)
        else:
            to_remove.add(skill_a.id)

    for skill_id in to_remove:
        merged.remove_skill(skill_id)

    removed = len(to_remove)
    total_after = len(merged.skills())
    print(f"Deduplication: {total_before} → {total_after} skills ({removed} removed)")

    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-m", "--model",
        default="bedrock/us.anthropic.claude-sonnet-4-6",
        help="Main RR model (default: Sonnet 4.6 via Bedrock)",
    )
    parser.add_argument(
        "--subagent-model",
        default="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        help="Sub-agent model (default: Haiku 4.5 via Bedrock)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=60,
        help="Max RR iterations per batch (default: 60)",
    )
    parser.add_argument(
        "--max-llm-calls", type=int, default=60,
        help="Max LLM calls per batch (default: 60)",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.7,
        help="Dedup similarity threshold (default: 0.7)",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Output directory (default: results/rr_car_bench_<timestamp>)",
    )
    parser.add_argument(
        "--batch", type=str, default=None,
        help="Run a single batch by name (e.g., base_vehicle_climate)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show batch plan without running",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate env
    if not args.dry_run and not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY required for deduplication embeddings")
        sys.exit(1)

    # Load batch assignment
    assignment = load_batch_assignment()
    batches = assignment["batches"]

    # Filter to single batch if requested
    if args.batch:
        if args.batch not in batches:
            print(f"ERROR: Unknown batch '{args.batch}'")
            print(f"Available: {', '.join(batches.keys())}")
            sys.exit(1)
        batches = {args.batch: batches[args.batch]}

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (ROOT / "results" / f"rr_car_bench_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print plan
    print("=" * 60)
    print("RR CAR-bench Experiment")
    print("=" * 60)
    print(f"Model:          {args.model}")
    print(f"Sub-agent:      {args.subagent_model}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Max LLM calls:  {args.max_llm_calls}")
    print(f"Dedup threshold:{args.threshold}")
    print(f"Output:         {output_dir}")
    print(f"Batches:        {len(batches)}")
    print()
    for name, info in batches.items():
        print(f"  {name}: {len(info['trace_ids'])} traces ({info['task_type']}, {info['domains']})")
    print()

    if args.dry_run:
        print("DRY RUN — exiting.")
        return

    # Save run config
    config = {
        "model": args.model,
        "subagent_model": args.subagent_model,
        "max_iterations": args.max_iterations,
        "max_llm_calls": args.max_llm_calls,
        "dedup_threshold": args.threshold,
        "prompt": "REFLECTOR_RECURSIVE_V5_PROMPT",
        "batches": {k: len(v["trace_ids"]) for k, v in batches.items()},
        "timestamp": timestamp,
        "merge_strategy": "concatenate + embedding dedup (Option A)",
    }
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run each batch
    per_batch_skillbooks: Dict[str, Skillbook] = {}
    total_start = datetime.now()

    for batch_name, batch_info in batches.items():
        sb = run_batch(batch_name, batch_info, args, output_dir)
        if sb is not None:
            per_batch_skillbooks[batch_name] = sb

    if not per_batch_skillbooks:
        print("\nNo skillbooks produced. Exiting.")
        return

    # Merge + dedup
    print(f"\n{'='*60}")
    print("Merging per-batch skillbooks")
    print(f"{'='*60}")
    merged = merge_skillbooks(per_batch_skillbooks, args.threshold)

    # Save merged outputs
    merged.save_to_file(str(output_dir / "skillbook_merged.json"))

    # Markdown export
    skills = merged.skills()
    md_path = output_dir / "skills_merged.md"
    with open(md_path, "w") as f:
        f.write("# RR CAR-bench — Merged Skillbook\n\n")
        f.write(f"Batches: {len(per_batch_skillbooks)} | Skills: {len(skills)}\n")
        f.write(f"Merge strategy: concatenate + embedding dedup (threshold={args.threshold})\n\n")
        for section, section_skills in groupby(
            sorted(skills, key=lambda s: s.section), key=lambda s: s.section
        ):
            f.write(f"## {section}\n\n")
            for skill in section_skills:
                f.write(f"- {skill.content}\n")
            f.write("\n")

    # External agent injection (for downstream eval)
    injection = wrap_skillbook_for_external_agent(merged)
    if injection:
        injection_path = output_dir / "external_agent_injection.txt"
        with open(injection_path, "w") as f:
            f.write(injection)

    total_duration = (datetime.now() - total_start).total_seconds()

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total time: {total_duration:.1f}s ({total_duration/60:.1f}m)")
    print(f"Batches run: {len(per_batch_skillbooks)}")
    per_batch_counts = {
        name: len(sb.skills()) for name, sb in per_batch_skillbooks.items()
    }
    for name, count in per_batch_counts.items():
        print(f"  {name}: {count} skills")
    print(f"Merged: {len(skills)} skills")
    print(f"\nOutputs:")
    print(f"  {output_dir}/skillbook_merged.json")
    print(f"  {output_dir}/skills_merged.md")
    print(f"  {output_dir}/external_agent_injection.txt")


if __name__ == "__main__":
    main()
