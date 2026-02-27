#!/usr/bin/env python3
"""Learn from OpenClaw session transcripts and sync strategies to AGENTS.md.

Reads OpenClaw session JSONL files, feeds them through the ACE learning
pipeline (TraceAnalyser), and writes the updated skillbook back into
OpenClaw's workspace so the agent picks up learned strategies on its
next session.

Designed to run manually or via cron.

Requirements:
    uv sync  # core deps only — no extras needed

Environment:
    # Required — LLM for reflection and skill extraction
    export ANTHROPIC_API_KEY="your-api-key"

    # Optional — override defaults
    export ACE_MODEL="anthropic/claude-sonnet-4-20250514"
    export OPENCLAW_AGENT_ID="main"
    export OPENCLAW_HOME="~/.openclaw"
    export OPENCLAW_WORKSPACE="~/.openclaw/workspace"

Usage:
    # One-off run
    uv run python examples/openclaw/learn_from_traces.py

    # Dry run — parse sessions without learning
    uv run python examples/openclaw/learn_from_traces.py --dry-run

    # Reprocess all sessions (ignore processed log)
    uv run python examples/openclaw/learn_from_traces.py --reprocess
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Ensure project root is importable
_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root))

from dotenv import load_dotenv

load_dotenv(_root / ".env")

from ace_next import (
    LiteLLMClient,
    OpikStep,
    Reflector,
    Skillbook,
    SkillManager,
    TraceAnalyser,
    register_opik_litellm_callback,
    wrap_skillbook_context,
)
from ace_next.core.context import ACEStepContext
from ace_next.steps.load_traces import LoadTracesStep
from ace_next.integrations.openclaw import OpenClawToTraceStep


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENCLAW_HOME = Path(os.getenv("OPENCLAW_HOME", Path.home() / ".openclaw")).expanduser()
OPENCLAW_AGENT_ID = os.getenv("OPENCLAW_AGENT_ID", "main")
OPENCLAW_WORKSPACE = Path(
    os.getenv("OPENCLAW_WORKSPACE", OPENCLAW_HOME / "workspace")
).expanduser()

SESSIONS_DIR = OPENCLAW_HOME / "agents" / OPENCLAW_AGENT_ID / "sessions"
SKILLBOOK_PATH = OPENCLAW_HOME / "ace_skillbook.json"
PROCESSED_LOG = OPENCLAW_HOME / "ace_processed.txt"

MODEL = os.getenv("ACE_MODEL", "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0")

# AGENTS.md markers for the skillbook section
MARKER_START = "<!-- ACE:SKILLBOOK:START -->"
MARKER_END = "<!-- ACE:SKILLBOOK:END -->"


# ---------------------------------------------------------------------------
# Processed-session tracking (US3: FR-007, FR-008)
# ---------------------------------------------------------------------------


def load_processed() -> set[str]:
    """Load the set of already-processed session filenames."""
    if PROCESSED_LOG.exists():
        return set(PROCESSED_LOG.read_text().splitlines())
    return set()


def save_processed(processed: set[str]) -> None:
    """Persist the set of processed session filenames."""
    PROCESSED_LOG.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_LOG.write_text("\n".join(sorted(processed)) + "\n")


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
# AGENTS.md sync (US2: FR-005, FR-006)
# ---------------------------------------------------------------------------


def sync_to_agents_md(skillbook: Skillbook) -> None:
    """Write the skillbook into AGENTS.md between marker comments.

    Creates AGENTS.md if it doesn't exist.  Replaces the section between
    ``<!-- ACE:SKILLBOOK:START -->`` and ``<!-- ACE:SKILLBOOK:END -->``
    markers, preserving all other content.
    """
    agents_md = OPENCLAW_WORKSPACE / "AGENTS.md"
    existing = agents_md.read_text() if agents_md.exists() else ""

    context = wrap_skillbook_context(skillbook)
    if not context:
        print("  Skillbook empty — skipping AGENTS.md sync")
        return

    ace_section = (
        f"{MARKER_START}\n"
        f"## Learned Strategies\n\n"
        f"These strategies were learned from your past sessions. Use relevant\n"
        f"ones to improve your responses. Cite strategy IDs (e.g.\n"
        f"[web-scraping-00001]) when you apply them.\n\n"
        f"{context}\n"
        f"{MARKER_END}"
    )

    if MARKER_START in existing:
        # Replace existing section
        pattern = re.escape(MARKER_START) + r".*?" + re.escape(MARKER_END)
        content = re.sub(pattern, ace_section, existing, flags=re.DOTALL)
    else:
        content = existing.rstrip() + "\n\n" + ace_section + "\n"

    agents_md.parent.mkdir(parents=True, exist_ok=True)
    agents_md.write_text(content)
    print(f"  Synced {len(skillbook.skills())} strategies to {agents_md}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def section(name: str) -> None:
    print(f"\n{'=' * 60}\n  {name}\n{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn from OpenClaw session transcripts and sync to AGENTS.md."
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
        "files",
        nargs="*",
        type=Path,
        help="JSONL trace files to process directly (skips session discovery).",
    )
    args = parser.parse_args()

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
        if not SESSIONS_DIR.exists():
            print(f"  Sessions directory not found: {SESSIONS_DIR}")
            print("  Is OpenClaw installed and has the agent run at least once?")
            sys.exit(1)

        processed = set() if args.reprocess else load_processed()
        session_files = sorted(SESSIONS_DIR.glob("*.jsonl"))
        new_sessions = [f for f in session_files if f.name not in processed]

        print(f"  Sessions dir:  {SESSIONS_DIR}")
        print(f"  Total sessions: {len(session_files)}")
        print(f"  Already processed: {len(processed)}")
        print(f"  New to process: {len(new_sessions)}")

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
    if SKILLBOOK_PATH.exists():
        try:
            skillbook = Skillbook.load_from_file(str(SKILLBOOK_PATH))
            print(
                f"  Loaded {len(skillbook.skills())} existing strategies"
                f" from {SKILLBOOK_PATH}"
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

    opik_step = OpikStep(
        project_name="openclaw-trace-learning",
        tags=["openclaw", "trace-analyser"],
    )
    register_opik_litellm_callback(project_name="openclaw-trace-learning")

    analyser = TraceAnalyser.from_roles(
        reflector=Reflector(client),
        skill_manager=SkillManager(client),
        skillbook=skillbook,
        extra_steps=[opik_step],  # type: ignore[arg-type]
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
    SKILLBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    skillbook.save_to_file(str(SKILLBOOK_PATH))
    print(f"  Skillbook: {SKILLBOOK_PATH}")

    # -- Mark sessions processed (FR-007) --
    if not args.files:
        processed.update(f.name for f in new_sessions)
        save_processed(processed)
        print(f"  Processed log: {PROCESSED_LOG}")

    # -- Sync to AGENTS.md (FR-005) --
    section("Syncing to OpenClaw")
    sync_to_agents_md(skillbook)

    section("Done")


if __name__ == "__main__":
    main()
