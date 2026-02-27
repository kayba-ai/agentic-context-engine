#!/usr/bin/env python3
"""Learn from OpenClaw session transcripts.

Reads OpenClaw session JSONL files, feeds them through the ACE learning
pipeline (TraceAnalyser), and writes the updated skillbook as JSON and
markdown to the output directory.

Designed to live inside an OpenClaw skills directory::

    ~/.openclaw/workspace/skills/kayba-ace/
        learn_from_traces.py   ← this script
        SKILL.md
        ace_skillbook.json     ← generated
        ace_skillbook.md       ← generated

All paths are derived from the script's location. OPENCLAW_HOME is
inferred as three directories up (workspace/skills/<name>/ → .openclaw/).
"""

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — derive everything from the script's location
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# Expected layout: OPENCLAW_HOME/workspace/skills/<name>/script.py
# Override with OPENCLAW_HOME env var if the script lives elsewhere.
OPENCLAW_HOME = Path(
    os.getenv("OPENCLAW_HOME", SCRIPT_DIR.parents[2])
).expanduser()

# Ensure ace_next is importable (dev: repo root; prod: pip install)
try:
    import ace_next  # noqa: F401
except ImportError:
    # Fallback for running from the repo checkout
    _repo_root = Path(__file__).resolve().parents[2]
    if (_repo_root / "ace_next").is_dir():
        sys.path.insert(0, str(_repo_root))

from dotenv import load_dotenv

load_dotenv(OPENCLAW_HOME / ".env")
load_dotenv(Path.home() / ".env")

from ace_next import (
    LiteLLMClient,
    OpikStep,
    Reflector,
    Skillbook,
    SkillManager,
    TraceAnalyser,
    register_opik_litellm_callback,
)
from ace_next.core.context import ACEStepContext
from ace_next.steps.load_traces import LoadTracesStep
from ace_next.integrations.openclaw import OpenClawToTraceStep
from ace_next.steps.export_markdown import ExportSkillbookMarkdownStep


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = os.getenv("ACE_MODEL", "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0")


# ---------------------------------------------------------------------------
# Processed-session tracking (US3: FR-007, FR-008)
# ---------------------------------------------------------------------------


def load_processed(path: Path) -> set[str]:
    """Load the set of already-processed session filenames."""
    if path.exists():
        return set(path.read_text().splitlines())
    return set()


def save_processed(path: Path, processed: set[str]) -> None:
    """Persist the set of processed session filenames."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sorted(processed)) + "\n")


# ---------------------------------------------------------------------------
# Session parsing via pipeline steps
# ---------------------------------------------------------------------------

_load_step = LoadTracesStep()
_convert_step = OpenClawToTraceStep()


def parse_session(path: Path) -> object | None:
    """Parse a single session JSONL file using pipeline steps.

    Returns the trace object (raw events for now, structured dict later)
    or None if the file is empty/unparseable.
    """
    ctx = ACEStepContext(sample=str(path))
    ctx = _load_step(ctx)

    # Skip empty sessions
    if not ctx.trace:
        return None

    ctx = _convert_step(ctx)
    return ctx.trace


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def section(name: str) -> None:
    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn from OpenClaw session transcripts."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse sessions but skip learning and sync.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Ignore the processed log and reprocess all sessions.",
    )
    parser.add_argument(
        "--agent",
        default=os.getenv("OPENCLAW_AGENT_ID", "main"),
        help="OpenClaw agent ID for session discovery (default: $OPENCLAW_AGENT_ID or 'main').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR,
        help="Output directory for skillbook files (default: script directory).",
    )
    parser.add_argument(
        "--opik",
        action="store_true",
        help="Enable Opik observability logging.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="JSONL trace files to process directly (skips session discovery).",
    )
    args = parser.parse_args()

    # -- Resolve paths --
    output_dir = args.output.expanduser().resolve()
    processed_log = output_dir / "ace_processed.txt"
    sessions_dir = OPENCLAW_HOME / "agents" / args.agent / "sessions"

    # -- Discover new sessions (FR-001, FR-012) --
    section("Discovering sessions")
    processed: set[str] = set()
    if args.files:
        new_sessions = [f.resolve() for f in args.files if f.exists()]
        missing = [f for f in args.files if not f.exists()]
        for m in missing:
            print(f"  WARNING: file not found: {m}")
        print(f"  Direct files: {len(new_sessions)}")
    else:
        if not sessions_dir.exists():
            print(f"  Sessions directory not found: {sessions_dir}")
            print("  Is OpenClaw installed and has the agent run at least once?")
            sys.exit(1)

        processed = set() if args.reprocess else load_processed(processed_log)
        session_files = sorted(sessions_dir.glob("*.jsonl"))
        new_sessions = [f for f in session_files if f.name not in processed]

        print(f"  Agent:           {args.agent}")
        print(f"  Sessions dir:    {sessions_dir}")
        print(f"  Total sessions:  {len(session_files)}")
        print(f"  Already processed: {len(processed)}")
        print(f"  New to process:  {len(new_sessions)}")

    if not new_sessions:
        print("  Nothing new to learn from.")
        return

    # -- Parse into traces (FR-002, FR-010) --
    section("Parsing sessions")
    traces: list[object] = []
    skipped = 0
    for session_file in new_sessions:
        trace = parse_session(session_file)
        if trace:
            traces.append(trace)
            print(f"  + {session_file.name}")
        else:
            skipped += 1

    print(f"  Parsed: {len(traces)}, Skipped (empty): {skipped}")

    if not traces:
        print("  No usable traces found.")
        return

    # -- Dry run: stop before learning (FR-009) --
    if args.dry_run:
        print("\n  --dry-run: stopping before learning.")
        return

    # -- Load or create skillbook (FR-004) --
    section("Loading skillbook")
    skillbook_path = output_dir / "ace_skillbook.json"
    if skillbook_path.exists():
        try:
            skillbook = Skillbook.load_from_file(str(skillbook_path))
            print(
                f"  Loaded {len(skillbook.skills())} existing strategies"
                f" from {skillbook_path}"
            )
        except Exception as exc:
            print(f"  ERROR: Failed to load skillbook: {exc}")
            print("  Starting with empty skillbook instead.")
            skillbook = Skillbook()
    else:
        skillbook = Skillbook()
        print("  Starting with empty skillbook")

    skills_before = len(skillbook.skills())

    # -- Run learning (FR-003) --
    section(f"Learning from {len(traces)} traces")
    client = LiteLLMClient(
        model=MODEL,
        api_key=os.getenv("AWS_BEARER_TOKEN_BEDROCK"),
    )

    markdown_path = output_dir / "ace_skillbook.md"
    export_md_step = ExportSkillbookMarkdownStep(markdown_path, skillbook)

    extra_steps: list = [export_md_step]
    if args.opik:
        opik_step = OpikStep(
            project_name="openclaw-trace-learning",
            tags=["openclaw", "trace-analyser"],
        )
        register_opik_litellm_callback(project_name="openclaw-trace-learning")
        extra_steps.append(opik_step)

    analyser = TraceAnalyser.from_roles(
        reflector=Reflector(client),
        skill_manager=SkillManager(client),
        skillbook=skillbook,
        extra_steps=extra_steps,  # type: ignore[arg-type]
    )

    results = analyser.run(traces, epochs=1, wait=True)

    errors = [r for r in results if r.error]
    if errors:
        for e in errors:
            print(f"  ERROR: {e.failed_at}: {e.error}")
    print(f"  Processed: {len(results) - len(errors)}/{len(results)}")

    skills_after = len(skillbook.skills())
    new_skills = skills_after - skills_before
    print(f"  New strategies: {new_skills} (total: {skills_after})")

    if skills_after > 0:
        print("\n  Latest strategies:")
        for skill in skillbook.skills()[-3:]:
            print(f"    [{skill.id}] {skill.content[:70]}")

    # -- Save skillbook (FR-004) --
    section("Saving")
    json_path = output_dir / "ace_skillbook.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    skillbook.save_to_file(str(json_path))
    print(f"  Skillbook JSON: {json_path}")
    print(f"  Skillbook MD:   {markdown_path}")

    # -- Mark sessions processed (FR-007) --
    if not args.files:
        processed.update(f.name for f in new_sessions)
        save_processed(processed_log, processed)
        print(f"  Processed log: {processed_log}")

    section("Done")


if __name__ == "__main__":
    main()
