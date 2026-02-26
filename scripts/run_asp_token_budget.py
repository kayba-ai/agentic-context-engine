#!/usr/bin/env python3
"""ASP Token Budget Experiment — ace_next vs old ace, with deduplication.

Runs three variants of Agentic System Prompting on 25 traces:
  A) ace_next, no token budget  → results/ace_next_no-budget_25-traces_dedupped/
  B) ace_next, budget 4000      → results/ace_next_budget-4000_25-traces_dedupped/
  C) old ace module              → results/old-asp_25-traces_dedupped/

Then generates a COMPARISON_dedupped.md with side-by-side stats
(does NOT overwrite existing COMPARISON.md from run 1).

Usage:
    cd feature-tzerweck-token-budget && uv run python scripts/run_asp_token_budget.py
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# ace_next is not installed as a package — ensure project root is on sys.path
# (also makes the local ace/ module importable)
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import os
import time
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment — load .env from Kayba root (project dir only has .env.example)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT.parent / ".env", override=True)

# ---------------------------------------------------------------------------
# Imports — ace_next (new pipeline)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Imports — old ace module
# ---------------------------------------------------------------------------

from ace import (
    OfflineACE,
    ReplayAgent,
    SimpleEnvironment,
    Sample,
    Skillbook as OldSkillbook,
    Reflector as OldReflector,
    SkillManager as OldSkillManager,
)
from ace.deduplication import DeduplicationConfig as OldDeduplicationConfig
from ace.prompt_manager import (
    PromptManager,
    wrap_skillbook_for_external_agent as old_wrap,
)
from ace.llm_providers.litellm_client import (
    LiteLLMClient as OldLiteLLMClient,
    LiteLLMConfig as OldLiteLLMConfig,
)
from ace.llm_providers.instructor_client import wrap_with_instructor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
EPOCHS = 1
DEDUP_THRESHOLD = 0.7
DEDUP_INTERVAL = 5

TRACES_DIR = ROOT / "results" / "traces_25"
RESULTS_DIR = ROOT / "results"

VARIANTS: list[dict[str, Any]] = [
    {
        "name": "ace_next_no-budget_25-traces_dedupped",
        "module": "ace_next",
        "token_budget": None,
    },
    {
        "name": "ace_next_budget-4000_25-traces_dedupped",
        "module": "ace_next",
        "token_budget": 4000,
    },
    {
        "name": "old-asp_25-traces_dedupped",
        "module": "ace",
        "token_budget": None,
    },
]

KNOWN_DIFFERENCES = """\
1. **Deduplication**: Old = similarity report in SkillManager prompt each call. New = separate `DeduplicateStep` every 5 samples.
2. **Tagging**: Old = part of SkillManager output. New = separate `TagStep` before `UpdateStep`.
3. **Progress string**: Old = caller-provided. New = auto-generated (`"Epoch 1/1, sample 5/25"`).
4. **question_context**: Old = `"-"` from `sample.question`. New = `"-"` extracted from trace dict `question` field.
5. **SkillManager prompt**: Old v3 = no token budget text. New v3 = extra paragraph in `<skillbook_size_management>` about token_budget (won't trigger without budget set).
6. **Reflector prompt**: Identical between old and new.
7. **Pipeline concurrency**: New uses `ReflectStep` with `max_workers=3` (parallel reflections)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_traces() -> tuple[list[dict[str, str]], list[str]]:
    """Load .toon files — returns (trace_dicts, trace_names)."""
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

    print(f"Loaded {len(trace_dicts)} traces from {TRACES_DIR}")
    return trace_dicts, trace_names


def count_tokens(text: str) -> int:
    """Count tokens using cl100k_base (tiktoken)."""
    enc = tiktoken.encoding_for_model("gpt-4")
    return len(enc.encode(text))


def save_skills_md(skillbook: Skillbook | OldSkillbook, path: Path) -> None:
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


def compute_token_stats_ace_next(
    skillbook: Skillbook,
) -> dict[str, dict[str, int]]:
    """Compute token/char counts for ace_next skillbook formats."""
    toon_text = skillbook.as_prompt()
    md_text = skillbook._as_markdown_debug()
    injection_text = wrap_skillbook_for_external_agent(skillbook)

    return {
        "toon": {"tokens": count_tokens(toon_text), "chars": len(toon_text)},
        "markdown": {"tokens": count_tokens(md_text), "chars": len(md_text)},
        "injection": {
            "tokens": count_tokens(injection_text),
            "chars": len(injection_text),
        },
    }


def compute_token_stats_old_ace(
    skillbook: OldSkillbook,
) -> dict[str, dict[str, int]]:
    """Compute token/char counts for old ace skillbook formats."""
    toon_text = skillbook.as_prompt()
    md_text = skillbook._as_markdown_debug()
    injection_text = old_wrap(skillbook, version="3")

    return {
        "toon": {"tokens": count_tokens(toon_text), "chars": len(toon_text)},
        "markdown": {"tokens": count_tokens(md_text), "chars": len(md_text)},
        "injection": {
            "tokens": count_tokens(injection_text),
            "chars": len(injection_text),
        },
    }


def write_run_info(
    *,
    output_dir: Path,
    timestamp: str,
    token_budget: int | None,
    trace_names: list[str],
    stats: dict[str, object],
    duration_s: float,
    token_stats: dict[str, dict[str, int]],
    module: str = "ace_next",
) -> None:
    """Write RUN_INFO.md for a single variant."""
    n_skills = stats["skills"]

    toon = token_stats["toon"]
    md = token_stats["markdown"]
    inj = token_stats["injection"]

    if module == "ace_next":
        budget_str = str(token_budget) if token_budget is not None else "None"
        module_line = "**Module**: `ace_next` (pipeline-based)"
        budget_line = f"**Token budget**: {budget_str}"
        known_diff_section = f"""
## Known Differences from Old Baseline (ace module)
{KNOWN_DIFFERENCES}
"""
    else:
        module_line = "**Module**: `ace` (OfflineACE, async_learning=True, max_reflector_workers=3)"
        budget_line = "**Token budget**: N/A (not supported)"
        known_diff_section = ""

    content = f"""\
# Run Info

**Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Script**: `scripts/run_asp_token_budget.py`
{module_line}
**Model**: `{MODEL}`
**Epochs**: {EPOCHS}
**Dedup threshold**: {DEDUP_THRESHOLD}
{budget_line}
"""

    if module == "ace":
        content += "**Prompts**: SkillManager v3.0, Reflector v2.1\n"

    content += f"""
## Traces
25 `.toon` files from `results/traces_25/`: {", ".join(trace_names)}
{known_diff_section}
## Results
- **{n_skills} skills** generated
- **Duration**: ~{duration_s:.0f}s (~{duration_s / 60:.1f} min)
- `skillbook_{timestamp}.json` — full skillbook
- `skills_{timestamp}.md` — skills grouped by section
- `external_agent_injection_{timestamp}.txt` — injection text

## Token Budget
Tokenizer: `cl100k_base` (tiktoken). Context window: 200k (Haiku 4.5).

### Skillbook formats
```
┌─────────────────────────────────────┬────────┬────────┐
│               Format                │ Tokens │ Chars  │
├─────────────────────────────────────┼────────┼────────┤
│ TOON (as_prompt())                  │ {toon["tokens"]:>6,} │ {toon["chars"]:>6,} │
├─────────────────────────────────────┼────────┼────────┤
│ Markdown (_as_markdown_debug())     │ {md["tokens"]:>6,} │ {md["chars"]:>6,} │
├─────────────────────────────────────┼────────┼────────┤
│ Full injection text (the .txt file) │ {inj["tokens"]:>6,} │ {inj["chars"]:>6,} │
└─────────────────────────────────────┴────────┴────────┘
```
"""
    (output_dir / "RUN_INFO.md").write_text(content, encoding="utf-8")


def write_comparison(variant_results: list[dict[str, Any]]) -> None:
    """Write COMPARISON_dedupped.md under results/ with 3-column table."""
    a = variant_results[0]  # ace_next no-budget
    b = variant_results[1]  # ace_next budget-4000
    c = variant_results[2]  # old ace

    def _fmt(v: int | float) -> str:
        if isinstance(v, float):
            return f"{v:,.1f}"
        return f"{v:,}"

    rows = [
        ("Skills", a["skills"], b["skills"], c["skills"]),
        ("Sections", a["sections"], b["sections"], c["sections"]),
        (
            "Duration (s)",
            f"~{a['duration_s']:.0f}",
            f"~{b['duration_s']:.0f}",
            f"~{c['duration_s']:.0f}",
        ),
        (
            "TOON tokens",
            _fmt(a["toon_tokens"]),
            _fmt(b["toon_tokens"]),
            _fmt(c["toon_tokens"]),
        ),
        (
            "TOON chars",
            _fmt(a["toon_chars"]),
            _fmt(b["toon_chars"]),
            _fmt(c["toon_chars"]),
        ),
        (
            "Injection tokens",
            _fmt(a["injection_tokens"]),
            _fmt(b["injection_tokens"]),
            _fmt(c["injection_tokens"]),
        ),
        (
            "Injection chars",
            _fmt(a["injection_chars"]),
            _fmt(b["injection_chars"]),
            _fmt(c["injection_chars"]),
        ),
        ("Helpful tags total", a["helpful_tags"], b["helpful_tags"], c["helpful_tags"]),
        ("Harmful tags total", a["harmful_tags"], b["harmful_tags"], c["harmful_tags"]),
    ]

    table_lines = [
        "| Metric                | ace_next no-budget (A) | ace_next budget-4000 (B) | old ace (C) |",
        "|-----------------------|------------------------|--------------------------|-------------|",
    ]
    for metric, va, vb, vc in rows:
        table_lines.append(
            f"| {metric:<21} | {str(va):>22} | {str(vb):>24} | {str(vc):>11} |"
        )

    # Auto-generate observations
    observations = []
    skill_diff_ab = a["skills"] - b["skills"]
    skill_diff_ac = a["skills"] - c["skills"]
    sign_ab = "+" if skill_diff_ab >= 0 else ""
    sign_ac = "+" if skill_diff_ac >= 0 else ""
    observations.append(
        f"- Variant A (no budget) produced {a['skills']} skills ({sign_ac}{skill_diff_ac} vs old ace)."
    )
    observations.append(f"- Variant B (budget 4000) produced {b['skills']} skills.")
    observations.append(f"- Variant C (old ace) produced {c['skills']} skills.")
    if b["toon_tokens"] <= 4000:
        observations.append(
            f"- Variant B TOON tokens ({b['toon_tokens']:,}) stayed within the 4000-token budget."
        )
    else:
        observations.append(
            f"- Variant B TOON tokens ({b['toon_tokens']:,}) exceeded the 4000-token budget."
        )
    if a["skills"] != c["skills"]:
        observations.append(
            "- Skill count difference between ace_next (A) and old ace (C) is expected — "
            "see Known Differences in RUN_INFO.md (dedup timing, tagging, progress string, pipeline concurrency)."
        )

    content = f"""\
# ASP Token Budget Comparison (with deduplication)

## Configuration
- **Model**: `{MODEL}`
- **Traces**: 25 `.toon` files from `results/traces_25/`
- **Epochs**: {EPOCHS}
- **Dedup threshold**: {DEDUP_THRESHOLD}
- **Variant A**: ace_next, no token budget
- **Variant B**: ace_next, token_budget=4000
- **Variant C**: old ace module (SkillManager v3.0, Reflector v2.1)

## Results
{chr(10).join(table_lines)}

## Observations
{chr(10).join(observations)}
"""
    out_path = RESULTS_DIR / "COMPARISON_dedupped.md"
    out_path.write_text(content, encoding="utf-8")
    print(f"\nComparison: {out_path}")


# ---------------------------------------------------------------------------
# Variant runners
# ---------------------------------------------------------------------------


def run_ace_next_variant(
    trace_dicts: list[dict[str, str]],
    trace_names: list[str],
    variant: dict[str, Any],
) -> dict[str, Any]:
    """Run an ace_next variant (TraceAnalyser pipeline) and save outputs."""
    name = variant["name"]
    token_budget = variant["token_budget"]
    output_dir = RESULTS_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    budget_label = str(token_budget) if token_budget is not None else "None"
    print(f"\n{'='*60}")
    print(f"Variant: {name}  (module=ace_next, token_budget={budget_label})")
    print(f"{'='*60}")

    # Components
    config = LiteLLMConfig(model=MODEL, max_tokens=8192, temperature=1)
    llm = LiteLLMClient(config=config)
    skillbook = Skillbook()
    reflector = Reflector(llm)
    skill_manager = SkillManager(llm, token_budget=token_budget)
    dedup_manager = DeduplicationManager(
        DeduplicationConfig(
            enabled=True,
            similarity_threshold=DEDUP_THRESHOLD,
            embedding_model="text-embedding-3-small",
        )
    )

    analyser = TraceAnalyser.from_roles(
        reflector=reflector,
        skill_manager=skill_manager,
        skillbook=skillbook,
        dedup_manager=dedup_manager,
        dedup_interval=DEDUP_INTERVAL,
    )

    print(f"Starting analysis: {len(trace_dicts)} traces, {EPOCHS} epoch(s)")
    t0 = time.time()
    analyser.run(trace_dicts, epochs=EPOCHS)
    duration_s = time.time() - t0

    # Save outputs
    skillbook_path = output_dir / f"skillbook_{timestamp}.json"
    skills_md_path = output_dir / f"skills_{timestamp}.md"
    injection_path = output_dir / f"external_agent_injection_{timestamp}.txt"

    skillbook.save_to_file(str(skillbook_path))
    save_skills_md(skillbook, skills_md_path)

    injection_text = wrap_skillbook_for_external_agent(skillbook)
    if injection_text:
        injection_path.write_text(injection_text, encoding="utf-8")

    stats = skillbook.stats()
    token_stats = compute_token_stats_ace_next(skillbook)

    write_run_info(
        output_dir=output_dir,
        timestamp=timestamp,
        token_budget=token_budget,
        trace_names=trace_names,
        stats=stats,
        duration_s=duration_s,
        token_stats=token_stats,
        module="ace_next",
    )

    n_skills = stats["skills"]
    n_sections = stats["sections"]
    tags = stats["tags"]
    print(f"Done in {duration_s:.1f}s — {n_skills} skills, {n_sections} sections")
    print(f"  TOON tokens: {token_stats['toon']['tokens']:,}")
    print(f"  Injection tokens: {token_stats['injection']['tokens']:,}")
    print(f"  Saved to: {output_dir}")

    return {
        "name": name,
        "skills": n_skills,
        "sections": n_sections,
        "duration_s": duration_s,
        "toon_tokens": token_stats["toon"]["tokens"],
        "toon_chars": token_stats["toon"]["chars"],
        "injection_tokens": token_stats["injection"]["tokens"],
        "injection_chars": token_stats["injection"]["chars"],
        "helpful_tags": tags["helpful"],
        "harmful_tags": tags["harmful"],
    }


def run_old_ace_variant(
    trace_dicts: list[dict[str, str]],
    trace_names: list[str],
    variant: dict[str, Any],
) -> dict[str, Any]:
    """Run the old ace variant (OfflineACE + ReplayAgent) and save outputs."""
    name = variant["name"]
    output_dir = RESULTS_DIR / name
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"Variant: {name}  (module=ace, SkillManager v3.0, Reflector v2.1)")
    print(f"{'='*60}")

    # Components — old ace with PromptManager for v3 SkillManager, v2.1 Reflector
    # Wrap with Instructor for robust structured output parsing — the feature
    # branch added complete_structured() to the base LLMClient which prevents
    # the roles' auto-wrapping logic from kicking in.
    prompt_mgr = PromptManager()
    config = OldLiteLLMConfig(model=MODEL, max_tokens=8192, temperature=1)
    llm = wrap_with_instructor(OldLiteLLMClient(config=config))

    adapter = OfflineACE(
        agent=ReplayAgent(),
        reflector=OldReflector(llm, prompt_template=prompt_mgr.get_reflector_prompt()),
        skill_manager=OldSkillManager(
            llm,
            prompt_template=prompt_mgr.get_skill_manager_prompt(version="3.0"),
        ),
        dedup_config=OldDeduplicationConfig(
            enabled=True,
            similarity_threshold=DEDUP_THRESHOLD,
            embedding_model="text-embedding-3-small",
        ),
        async_learning=True,
        max_reflector_workers=3,
    )

    # Build Sample objects — ReplayAgent reads content from sample.metadata["response"]
    samples = [
        Sample(
            question="-",
            ground_truth="",
            id=trace_names[i],
            metadata={"response": td["answer"]},
        )
        for i, td in enumerate(trace_dicts)
    ]

    print(f"Starting analysis: {len(samples)} samples, {EPOCHS} epoch(s)")
    t0 = time.time()
    adapter.run(samples, SimpleEnvironment(), epochs=EPOCHS, wait_for_learning=True)
    duration_s = time.time() - t0

    skillbook = adapter.skillbook

    # Save outputs
    skillbook_path = output_dir / f"skillbook_{timestamp}.json"
    skills_md_path = output_dir / f"skills_{timestamp}.md"
    injection_path = output_dir / f"external_agent_injection_{timestamp}.txt"

    skillbook.save_to_file(str(skillbook_path))
    save_skills_md(skillbook, skills_md_path)

    injection_text = old_wrap(skillbook, version="3")
    if injection_text:
        injection_path.write_text(injection_text, encoding="utf-8")

    stats = skillbook.stats()
    token_stats = compute_token_stats_old_ace(skillbook)

    write_run_info(
        output_dir=output_dir,
        timestamp=timestamp,
        token_budget=None,
        trace_names=trace_names,
        stats=stats,
        duration_s=duration_s,
        token_stats=token_stats,
        module="ace",
    )

    n_skills = stats["skills"]
    n_sections = stats["sections"]
    tags = stats["tags"]
    print(f"Done in {duration_s:.1f}s — {n_skills} skills, {n_sections} sections")
    print(f"  TOON tokens: {token_stats['toon']['tokens']:,}")
    print(f"  Injection tokens: {token_stats['injection']['tokens']:,}")
    print(f"  Saved to: {output_dir}")

    return {
        "name": name,
        "skills": n_skills,
        "sections": n_sections,
        "duration_s": duration_s,
        "toon_tokens": token_stats["toon"]["tokens"],
        "toon_chars": token_stats["toon"]["chars"],
        "injection_tokens": token_stats["injection"]["tokens"],
        "injection_chars": token_stats["injection"]["chars"],
        "helpful_tags": tags["helpful"],
        "harmful_tags": tags["harmful"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Validate OpenAI API key (needed for dedup embeddings)
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or "your-" in key.lower():
        print(
            "ERROR: valid OPENAI_API_KEY required for deduplication embeddings!\n"
            "       Check that Kayba/.env contains a real key."
        )
        return

    trace_dicts, trace_names = load_traces()

    variant_results: list[dict[str, Any]] = []
    for variant in VARIANTS:
        if variant["module"] == "ace_next":
            result = run_ace_next_variant(trace_dicts, trace_names, variant)
        else:
            result = run_old_ace_variant(trace_dicts, trace_names, variant)
        variant_results.append(result)

    write_comparison(variant_results)
    print("\nDone — all variants complete.")


if __name__ == "__main__":
    main()
