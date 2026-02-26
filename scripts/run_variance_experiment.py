#!/usr/bin/env python3
"""Variance Experiment — statistical analysis of token budget effects.

Runs 5 repetitions for each of 7 budget settings (no-budget + 6 budget
levels) across 25 traces, processing one trace at a time with full
skillbook history.  Deduplication enabled with threshold 0.7.

Results:
    results/variance_experiment/
    ├── config.json
    ├── no-budget/run_1/ ... run_5/
    ├── budget-500/run_1/ ... run_5/
    ├── budget-1000/run_1/ ... run_5/
    ├── budget-2000/run_1/ ... run_5/
    ├── budget-3000/run_1/ ... run_5/
    ├── budget-5000/run_1/ ... run_5/
    └── budget-10000/run_1/ ... run_5/

Each run_N/ contains:
    skillbook_00.json ... skillbook_24.json   (intermediate snapshots)
    skills_final.md                           (human-readable final)
    run_info.json                             (per-trace stats)

Usage:
    # Run everything (~3.5 hours)
    uv run python scripts/run_variance_experiment.py

    # Test: single run, no-budget only
    uv run python scripts/run_variance_experiment.py --test

    # Run specific budget(s)
    uv run python scripts/run_variance_experiment.py --budgets none 500 1000

    # Run specific repetition(s)
    uv run python scripts/run_variance_experiment.py --runs 1 2

    # Resume: skip completed runs
    uv run python scripts/run_variance_experiment.py --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Any

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT.parent / ".env", override=True)

import os
import statistics
import tiktoken

from ace_next import (
    DeduplicationConfig,
    DeduplicationManager,
    LiteLLMClient,
    LiteLLMConfig,
    Reflector,
    Skillbook,
    SkillManager,
    TraceAnalyser,
)
from ace_next.implementations.prompts import wrap_skillbook_for_external_agent
from ace_next.steps import learning_tail, OpikStep
from ace_next.steps.opik import register_opik_litellm_callback, OPIK_AVAILABLE
from pipeline import Pipeline

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEDUP_THRESHOLD = 0.7
DEDUP_INTERVAL = 5  # Dedup every 5 traces (matches prior experiments)

TRACES_DIR = ROOT / "results" / "traces_25"
RESULTS_DIR = ROOT / "results" / "variance_experiment"

BUDGETS: list[int | None] = [None, 500, 1000, 2000, 3000, 5000, 10000]
RUNS_PER_BUDGET = 5

# Tokenizer (reused)
_enc = tiktoken.encoding_for_model("gpt-4")


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def budget_label(budget: int | None) -> str:
    return "no-budget" if budget is None else f"budget-{budget}"


def budget_dir(budget: int | None) -> Path:
    return RESULTS_DIR / budget_label(budget)


def run_dir(budget: int | None, run_num: int) -> Path:
    return budget_dir(budget) / f"run_{run_num}"


def load_traces() -> tuple[list[dict[str, str]], list[str]]:
    """Load .toon files in sorted order — returns (trace_dicts, trace_names)."""
    toon_files = sorted(TRACES_DIR.glob("*.toon"))
    if not toon_files:
        raise FileNotFoundError(f"No .toon files in {TRACES_DIR}")

    trace_dicts: list[dict[str, str]] = []
    trace_names: list[str] = []
    for path in toon_files:
        content = path.read_text(encoding="utf-8")
        trace_dicts.append(
            {
                "answer": content,
                "question": "-",
                "reasoning": "",
                "ground_truth": "",
                "feedback": "",
            }
        )
        trace_names.append(path.stem)

    return trace_dicts, trace_names


def save_skills_md(skillbook: Skillbook, path: Path) -> None:
    """Save skills grouped by section in markdown format."""
    skills = skillbook.skills()
    with open(path, "w", encoding="utf-8") as f:
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


def compute_skillbook_stats(skillbook: Skillbook) -> dict[str, Any]:
    """Compute comprehensive stats for a skillbook snapshot."""
    stats = skillbook.stats()
    toon_text = skillbook.as_prompt()
    injection_text = wrap_skillbook_for_external_agent(skillbook)

    skills = skillbook.skills()
    sections = set(s.section for s in skills)

    return {
        "skills": stats["skills"],
        "sections": len(sections),
        "section_names": sorted(sections),
        "toon_tokens": count_tokens(toon_text),
        "toon_chars": len(toon_text),
        "injection_tokens": count_tokens(injection_text),
        "injection_chars": len(injection_text),
        "helpful_tags": stats["tags"]["helpful"],
        "harmful_tags": stats["tags"]["harmful"],
    }


def is_run_complete(budget: int | None, run_num: int) -> bool:
    """Check if a run has already completed (has run_info.json with all 25 traces)."""
    info_path = run_dir(budget, run_num) / "run_info.json"
    if not info_path.exists():
        return False
    try:
        with open(info_path) as f:
            info = json.load(f)
        return len(info.get("traces", [])) == 25
    except (json.JSONDecodeError, KeyError):
        return False


# ---------------------------------------------------------------------------
# Single run: process 25 traces sequentially, save history
# ---------------------------------------------------------------------------


def execute_run(
    budget: int | None,
    run_num: int,
    trace_dicts: list[dict[str, str]],
    trace_names: list[str],
) -> dict[str, Any]:
    """Execute a single run: 25 traces sequentially, saving intermediate skillbooks."""
    out = run_dir(budget, run_num)
    out.mkdir(parents=True, exist_ok=True)

    label = budget_label(budget)
    print(f"\n{'='*60}")
    print(f"  {label} / run_{run_num}")
    print(f"{'='*60}")

    # Build components
    config = LiteLLMConfig(model=MODEL, max_tokens=8192, temperature=1)
    llm = LiteLLMClient(config=config)
    skillbook = Skillbook()
    reflector = Reflector(llm)
    skill_manager = SkillManager(llm, token_budget=budget)
    dedup_manager = DeduplicationManager(
        DeduplicationConfig(
            enabled=True,
            similarity_threshold=DEDUP_THRESHOLD,
            embedding_model="text-embedding-3-small",
        )
    )

    steps = learning_tail(
        reflector,
        skill_manager,
        skillbook,
        dedup_manager=dedup_manager,
        dedup_interval=DEDUP_INTERVAL,
    )
    steps.append(
        OpikStep(
            project_name="ace-variance-experiment",
            tags=[
                "variance-experiment",
                budget_label(budget),
                f"run-{run_num}",
            ],
        )
    )
    analyser = TraceAnalyser(pipeline=Pipeline(steps), skillbook=skillbook)

    run_start = time.time()
    trace_stats: list[dict[str, Any]] = []

    for i, (trace_dict, trace_name) in enumerate(zip(trace_dicts, trace_names)):
        t0 = time.time()
        analyser.run([trace_dict], epochs=1)
        trace_duration = time.time() - t0

        # Save intermediate skillbook
        skillbook_path = out / f"skillbook_{i:02d}.json"
        skillbook.save_to_file(str(skillbook_path))

        # Compute stats for this snapshot
        snap_stats = compute_skillbook_stats(skillbook)
        snap_stats["trace_index"] = i
        snap_stats["trace_name"] = trace_name
        snap_stats["duration_s"] = round(trace_duration, 1)
        trace_stats.append(snap_stats)

        n = snap_stats["skills"]
        toon = snap_stats["toon_tokens"]
        print(
            f"  [{i+1:2d}/25] {trace_name:<10s}  "
            f"{n:3d} skills  {toon:5,} TOON  {trace_duration:.1f}s"
        )

    total_duration = time.time() - run_start

    # Save final outputs
    save_skills_md(skillbook, out / "skills_final.md")

    # Save run_info.json
    final_stats = trace_stats[-1] if trace_stats else {}
    run_info = {
        "budget": budget,
        "budget_label": label,
        "run_num": run_num,
        "model": MODEL,
        "dedup_threshold": DEDUP_THRESHOLD,
        "dedup_interval": DEDUP_INTERVAL,
        "total_traces": len(trace_dicts),
        "trace_order": trace_names,
        "total_duration_s": round(total_duration, 1),
        "started_at": datetime.now().isoformat(),
        "final_skills": final_stats.get("skills", 0),
        "final_sections": final_stats.get("sections", 0),
        "final_toon_tokens": final_stats.get("toon_tokens", 0),
        "final_injection_tokens": final_stats.get("injection_tokens", 0),
        "traces": trace_stats,
    }
    with open(out / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(
        f"\n  Done: {final_stats.get('skills', 0)} skills, "
        f"{final_stats.get('toon_tokens', 0):,} TOON, "
        f"{total_duration:.0f}s total"
    )
    print(f"  Saved to: {out}")

    return run_info


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def _agg(values: list[float | int]) -> dict[str, Any]:
    """Compute mean/std/min/max for a list of numbers."""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "values": []}
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": round(m, 1),
        "std": round(s, 1),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def generate_budget_summary(
    budget: int | None, runs: list[int]
) -> dict[str, Any] | None:
    """Generate summary.json for a single budget level from completed run_info.json files."""
    skills_vals: list[int] = []
    sections_vals: list[int] = []
    toon_vals: list[int] = []
    injection_vals: list[int] = []
    duration_vals: list[float] = []

    for run_num in runs:
        info_path = run_dir(budget, run_num) / "run_info.json"
        if not info_path.exists():
            continue
        try:
            with open(info_path) as f:
                info = json.load(f)
        except (json.JSONDecodeError, KeyError):
            continue
        skills_vals.append(info.get("final_skills", 0))
        sections_vals.append(info.get("final_sections", 0))
        toon_vals.append(info.get("final_toon_tokens", 0))
        injection_vals.append(info.get("final_injection_tokens", 0))
        duration_vals.append(info.get("total_duration_s", 0))

    if not skills_vals:
        return None

    summary = {
        "budget": budget,
        "runs": len(skills_vals),
        "skills": _agg(skills_vals),
        "sections": _agg(sections_vals),
        "toon_tokens": _agg(toon_vals),
        "injection_tokens": _agg(injection_vals),
        "duration_s": _agg(duration_vals),
    }
    out_path = budget_dir(budget) / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Wrote {out_path}")
    return summary


def generate_cross_budget_summary(budgets: list[int | None], runs: list[int]) -> None:
    """Generate SUMMARY.md with a comparison table across all budgets."""
    summaries: list[dict[str, Any]] = []
    for budget in budgets:
        s = generate_budget_summary(budget, runs)
        if s:
            summaries.append(s)

    if not summaries:
        return

    lines = [
        "# Variance Experiment Summary\n",
        "",
        "| Budget | Runs | Skills | Sections | TOON tokens | Injection tokens | Duration (s) |",
        "|--------|------|--------|----------|-------------|------------------|--------------|",
    ]
    for s in summaries:
        label = "None" if s["budget"] is None else str(s["budget"])
        lines.append(
            f"| {label:<6s} | {s['runs']:<4d} "
            f"| {s['skills']['mean']:.1f} \u00b1 {s['skills']['std']:.1f} "
            f"| {s['sections']['mean']:.1f} \u00b1 {s['sections']['std']:.1f} "
            f"| {s['toon_tokens']['mean']:,.1f} \u00b1 {s['toon_tokens']['std']:.1f} "
            f"| {s['injection_tokens']['mean']:,.1f} \u00b1 {s['injection_tokens']['std']:.1f} "
            f"| {s['duration_s']['mean']:.1f} \u00b1 {s['duration_s']['std']:.1f} |"
        )
    lines.append("")

    out_path = RESULTS_DIR / "SUMMARY.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {out_path}")


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------


def save_experiment_config(budgets: list[int | None], runs: list[int]) -> None:
    """Save experiment configuration for reproducibility."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config = {
        "experiment": "variance_experiment",
        "description": (
            "Statistical analysis of token budget effects on skillbook generation. "
            "5 repetitions per budget level, 25 traces processed sequentially "
            "with full skillbook history."
        ),
        "model": MODEL,
        "dedup_threshold": DEDUP_THRESHOLD,
        "dedup_interval": DEDUP_INTERVAL,
        "traces_dir": str(TRACES_DIR),
        "budgets": [b if b is not None else "none" for b in budgets],
        "runs_per_budget": len(runs),
        "run_numbers": runs,
        "total_runs": len(budgets) * len(runs),
        "traces_per_run": 25,
        "created_at": datetime.now().isoformat(),
    }
    with open(RESULTS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Variance experiment runner")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: single run of no-budget only",
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=None,
        help="Budget levels to run (e.g., none 500 1000). Default: all.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=int,
        default=None,
        help="Run numbers to execute (e.g., 1 2 3). Default: 1-5.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already have complete run_info.json.",
    )
    args = parser.parse_args()

    # Validate API keys
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or "your-" in key.lower():
        print("ERROR: valid OPENAI_API_KEY required for dedup embeddings!")
        return

    aws_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    if not aws_key:
        print("WARNING: AWS credentials may be needed for Bedrock model.")

    # Opik setup
    opik_ok = register_opik_litellm_callback(project_name="ace-variance-experiment")
    if opik_ok:
        print("Opik LiteLLM callback registered for token/cost tracking.")
    else:
        print("Opik not available — continuing without observability.")

    # Determine budgets and runs
    if args.test:
        budgets: list[int | None] = [2000]
        runs = [1]
    else:
        if args.budgets:
            budgets = []
            for b in args.budgets:
                budgets.append(None if b.lower() == "none" else int(b))
        else:
            budgets = BUDGETS

        runs = args.runs or list(range(1, RUNS_PER_BUDGET + 1))

    # Load traces
    trace_dicts, trace_names = load_traces()
    print(f"Loaded {len(trace_dicts)} traces from {TRACES_DIR}")
    print(f"Budgets: {[budget_label(b) for b in budgets]}")
    print(f"Runs: {runs}")
    print(f"Total runs: {len(budgets) * len(runs)}")

    # Save config
    save_experiment_config(budgets, runs)

    # Execute
    completed = 0
    skipped = 0
    total = len(budgets) * len(runs)
    experiment_start = time.time()

    for budget in budgets:
        for run_num in runs:
            if args.resume and is_run_complete(budget, run_num):
                print(f"\nSkipping {budget_label(budget)}/run_{run_num} (complete)")
                skipped += 1
                continue

            execute_run(budget, run_num, trace_dicts, trace_names)
            completed += 1
            elapsed = time.time() - experiment_start
            remaining = total - completed - skipped
            if completed > 0:
                avg = elapsed / completed
                eta = remaining * avg
                print(
                    f"\n  Progress: {completed + skipped}/{total} "
                    f"({skipped} skipped) — ETA: {eta/60:.0f} min"
                )

        # Per-budget summary after all runs for this budget complete
        if all(is_run_complete(budget, r) for r in runs):
            generate_budget_summary(budget, runs)

    # Cross-budget summary
    print(f"\n{'='*60}")
    print("  Generating summaries")
    print(f"{'='*60}")
    generate_cross_budget_summary(budgets, runs)

    total_time = time.time() - experiment_start
    print(f"\n{'='*60}")
    print(f"Experiment complete: {completed} runs in {total_time/60:.1f} min")
    if skipped:
        print(f"  ({skipped} runs skipped via --resume)")
    print(f"Results: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
