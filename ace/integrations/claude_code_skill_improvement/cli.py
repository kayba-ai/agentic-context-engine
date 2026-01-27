"""CLI entrypoint for ace-skill-improve.

Usage:
    ace-skill-improve run --skill <path> --transcript <path> [--dry-run|--apply]
    ace-skill-improve setup --skill <path>  # Creates ~/.claude/commands/skill-retro.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run_improve(args: argparse.Namespace) -> int:
    """Run the skill improvement process."""
    from .improver import SkillImprover

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    skill_path = args.skill
    transcript_path = args.transcript
    dry_run = not args.apply

    if not Path(skill_path).exists():
        print(f"Error: Skill file not found: {skill_path}", file=sys.stderr)
        return 1

    if not Path(transcript_path).exists():
        print(f"Error: Transcript file not found: {transcript_path}", file=sys.stderr)
        return 1

    try:
        improver = SkillImprover()
        result = improver.improve(
            skill_path=skill_path,
            transcript_path=transcript_path,
            dry_run=dry_run,
        )

        # Output results
        if not result.has_changes:
            print("\nNo changes to make.")
            if result.publish_output and result.publish_output.rejected:
                print("\nRejected learnings:")
                for r in result.publish_output.rejected:
                    print(f"  - {r.learning}")
                    print(f"    Reason: {r.reason}")
            return 0

        # Show diff
        print("\n" + "=" * 60)
        print("DIFF")
        print("=" * 60)
        print(result.diff)

        # Show accepted changes
        if result.publish_output and result.publish_output.accepted:
            print("\n" + "=" * 60)
            print("ACCEPTED CHANGES")
            print("=" * 60)
            for change in result.publish_output.accepted:
                print(f"\n  + {change.change}")
                print(f"    Reason: {change.reason}")
                if change.evidence:
                    print(f"    Evidence: {change.evidence[:100]}...")

        # Show rejected learnings
        if result.publish_output and result.publish_output.rejected:
            print("\n" + "=" * 60)
            print("REJECTED LEARNINGS")
            print("=" * 60)
            for r in result.publish_output.rejected:
                print(f"\n  - {r.learning}")
                print(f"    Reason: {r.reason}")

        # Summary
        print("\n" + "=" * 60)
        if dry_run:
            print("DRY RUN - No changes written")
            print(f"To apply changes: ace-skill-improve run --skill {skill_path} --transcript {transcript_path} --apply")
        else:
            print(f"Changes applied to: {skill_path}")
            if result.backup_path:
                print(f"Backup created: {result.backup_path}")

        return 0

    except Exception as e:
        logger.error(f"Improvement failed: {e}", exc_info=args.verbose)
        print(f"\nError: {e}", file=sys.stderr)
        return 1


def create_slash_commands(skill_path: Optional[str] = None) -> Path:
    """
    Create slash commands for skill improvement.

    Creates ~/.claude/commands/skill-retro.md

    Args:
        skill_path: Optional default skill path to embed in command

    Returns:
        Path to the created command file
    """
    commands_dir = Path.home() / ".claude" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)

    # Use a sensible default if no skill path provided
    if skill_path:
        skill_arg = f'--skill "{skill_path}"'
    else:
        skill_arg = '--skill <path-to-skill-file>'

    slash_command_path = commands_dir / "skill-retro.md"
    slash_command_content = f"""Retrospectively improve a skill file based on a Claude Code session transcript.

## Instructions

1. First, identify the transcript file for the session you want to learn from:

```bash
# List recent transcripts
ls -lt ~/.claude/projects/*/*.jsonl | head -10
```

2. Run the skill improvement in dry-run mode (safe, shows diff):

```bash
ace-skill-improve run {skill_arg} --transcript <transcript-path>
```

3. Review the output:
   - **DIFF**: Shows exactly what would change
   - **ACCEPTED CHANGES**: Learnings that will be added (with reasons)
   - **REJECTED LEARNINGS**: Learnings filtered out (with reasons)

4. If the changes look good, apply them:

```bash
ace-skill-improve run {skill_arg} --transcript <transcript-path> --apply
```

This creates a backup before modifying the skill file.

## What It Does

The tool runs 3 LLM calls using your Claude Code subscription:
1. **Reflector**: Extracts learnings from the session transcript
2. **SkillManager**: Converts learnings to skill operations
3. **Publisher**: Applies minimal, high-signal edits

## Quality Constraints

- Maximum 20 new lines added
- Rejects changes if diff exceeds 30% of original
- Filters out session-specific quirks
- Deduplicates against existing strategies
"""

    slash_command_path.write_text(slash_command_content, encoding="utf-8")
    return slash_command_path


def run_setup(args: argparse.Namespace) -> int:
    """Create slash command for skill improvement."""
    skill_path = args.skill

    if not Path(skill_path).exists():
        print(f"Warning: Skill file not found: {skill_path}", file=sys.stderr)
        print("The slash command will be created anyway.", file=sys.stderr)
        print()

    # Create slash commands
    slash_command_path = create_slash_commands(skill_path)

    print(f"Created slash command: {slash_command_path}")
    print()
    print("Usage in Claude Code:")
    print("  /skill-retro")
    print()
    print("This will guide you through improving your skill file based on the session.")

    return 0


def run_show_transcript(args: argparse.Namespace) -> int:
    """Show parsed transcript in readable format."""
    from .transcript import TranscriptParser, render_full_chat, get_transcript_summary

    transcript_path = args.transcript

    if not Path(transcript_path).exists():
        print(f"Error: Transcript file not found: {transcript_path}", file=sys.stderr)
        return 1

    try:
        parser = TranscriptParser()
        transcript = parser.parse(transcript_path)

        if args.summary:
            summary = get_transcript_summary(transcript)
            print(json.dumps(summary, indent=2))
        else:
            print(render_full_chat(transcript))

        return 0

    except Exception as e:
        print(f"Error parsing transcript: {e}", file=sys.stderr)
        return 1


def main() -> None:
    """CLI entry point for ace-skill-improve."""
    parser = argparse.ArgumentParser(
        description="Improve workflow playbook skills based on session transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Review proposed changes (default, safe)
  ace-skill-improve run --skill ./SKILL.md --transcript ~/.claude/projects/.../session.jsonl

  # Apply changes with backup
  ace-skill-improve run --skill ./SKILL.md --transcript ~/.claude/projects/.../session.jsonl --apply

  # Create slash command for easier access
  ace-skill-improve setup --skill ./SKILL.md

  # View a transcript
  ace-skill-improve show --transcript ~/.claude/projects/.../session.jsonl

Process:
  1. Reflector: Extracts learnings from transcript
  2. SkillManager: Produces skill operations
  3. Publisher: Applies minimal, high-signal edits

Uses Claude Code subscription (no API keys required).
""",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Run command
    run_parser = subparsers.add_parser(
        "run",
        help="Run skill improvement on a transcript",
    )
    run_parser.add_argument(
        "--skill", "-s",
        required=True,
        help="Path to the skill file to improve",
    )
    run_parser.add_argument(
        "--transcript", "-t",
        required=True,
        help="Path to the transcript JSONL file",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview changes without writing (default)",
    )
    run_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (creates backup first)",
    )
    run_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    # Setup command
    setup_parser = subparsers.add_parser(
        "setup",
        help="Create slash command for skill improvement",
    )
    setup_parser.add_argument(
        "--skill", "-s",
        required=True,
        help="Path to the skill file (used in slash command)",
    )

    # Show command
    show_parser = subparsers.add_parser(
        "show",
        help="Show parsed transcript",
    )
    show_parser.add_argument(
        "--transcript", "-t",
        required=True,
        help="Path to the transcript JSONL file",
    )
    show_parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary only (JSON format)",
    )

    args = parser.parse_args()

    if args.command == "run":
        sys.exit(run_improve(args))
    elif args.command == "setup":
        sys.exit(run_setup(args))
    elif args.command == "show":
        sys.exit(run_show_transcript(args))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
