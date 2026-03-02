#!/usr/bin/env python3
"""Set up the ACE skill for an OpenClaw agent.

Copies the skill folder into the OpenClaw workspace and optionally
appends auto-learning instructions to AGENTS.md.

Usage:
    python examples/openclaw/setup.py                # interactive
    python examples/openclaw/setup.py --no-agents    # skip AGENTS.md
    python examples/openclaw/setup.py --openclaw-home /path/to/.openclaw
"""

import argparse
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_SRC = SCRIPT_DIR / "kayba-ace"
AGENTS_SNIPPET = SCRIPT_DIR / "AGENTS.md.snippet"

SKILL_NAME = "kayba-ace"

# Marker to detect if the snippet was already appended
AGENTS_MARKER = "## Auto-Learning"


def find_openclaw_home(override: str | None) -> Path:
    """Resolve OPENCLAW_HOME, checking common locations."""
    if override:
        p = Path(override).expanduser().resolve()
        if p.exists():
            return p
        print(f"  ERROR: --openclaw-home path does not exist: {p}")
        sys.exit(1)

    candidates = [
        Path.home() / ".openclaw",
    ]
    for c in candidates:
        if c.exists():
            return c

    print("  ERROR: Could not find OpenClaw installation.")
    print("  Checked: " + ", ".join(str(c) for c in candidates))
    print("  Use --openclaw-home to specify the path manually.")
    sys.exit(1)


def copy_skill(openclaw_home: Path) -> Path:
    """Copy the skill folder into the OpenClaw workspace."""
    dest = openclaw_home / "workspace" / "skills" / SKILL_NAME
    dest.mkdir(parents=True, exist_ok=True)

    # Copy all files (don't overwrite generated skillbook/processed files)
    generated = {"ace_skillbook.json", "ace_skillbook.md", "ace_processed.txt"}
    for src_file in SKILL_SRC.iterdir():
        if src_file.is_file():
            target = dest / src_file.name
            if target.exists() and src_file.name in generated:
                print(f"  SKIP (generated): {target}")
            elif target.exists():
                shutil.copy2(src_file, target)
                print(f"  UPDATED: {src_file.name} -> {target}")
            else:
                shutil.copy2(src_file, target)
                print(f"  COPIED: {src_file.name} -> {target}")

    return dest


def patch_agents_md(openclaw_home: Path) -> bool:
    """Append auto-learning instructions to AGENTS.md if not already present."""
    agents_md = openclaw_home / "workspace" / "AGENTS.md"

    if not AGENTS_SNIPPET.exists():
        print(f"  WARNING: snippet not found at {AGENTS_SNIPPET}")
        return False

    snippet_text = AGENTS_SNIPPET.read_text()

    # Strip the comment header from the snippet (lines starting with #)
    lines = snippet_text.splitlines()
    content_lines = []
    in_header = True
    for line in lines:
        if in_header and line.startswith("#") and not line.startswith("##"):
            continue
        in_header = False
        content_lines.append(line)
    snippet_body = "\n".join(content_lines).strip()

    if agents_md.exists():
        existing = agents_md.read_text()
        if AGENTS_MARKER in existing:
            print(f"  SKIP: AGENTS.md already contains '{AGENTS_MARKER}'")
            return False
        # Append
        with open(agents_md, "a") as f:
            f.write("\n\n" + snippet_body + "\n")
        print(f"  UPDATED: {agents_md}")
    else:
        agents_md.parent.mkdir(parents=True, exist_ok=True)
        agents_md.write_text(snippet_body + "\n")
        print(f"  CREATED: {agents_md}")

    return True


def section(name: str) -> None:
    print(f"\n{'=' * 50}\n  {name}\n{'=' * 50}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up the ACE skill for an OpenClaw agent."
    )
    parser.add_argument(
        "--openclaw-home",
        default=None,
        help="Path to the OpenClaw home directory (default: ~/.openclaw).",
    )
    parser.add_argument(
        "--no-agents",
        action="store_true",
        help="Skip patching AGENTS.md.",
    )
    args = parser.parse_args()

    section("Finding OpenClaw")
    openclaw_home = find_openclaw_home(args.openclaw_home)
    print(f"  Found: {openclaw_home}")

    section("Copying skill folder")
    skill_dest = copy_skill(openclaw_home)
    print(f"  Skill directory: {skill_dest}")

    if not args.no_agents:
        section("Patching AGENTS.md")
        patch_agents_md(openclaw_home)
    else:
        print("\n  Skipping AGENTS.md (--no-agents)")

    section("Done")
    print(f"""
  Skill installed at: {skill_dest}

  Next steps:
    1. Build the ACE Docker image (see Dockerfile.ace)
    2. Pass your LLM API key in docker-compose.yml
    3. Restart the gateway: docker compose down && docker compose up -d
    4. Send a message — the agent will run ace-learn automatically

  Full guide: https://kayba-ai.github.io/agentic-context-engine/integrations/openclaw/
""")


if __name__ == "__main__":
    main()
