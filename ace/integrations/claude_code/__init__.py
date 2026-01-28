"""Claude Code integration for ACE - subscription-only learning from Claude Code sessions.

This package provides:
- ACEHookLearner: Learns from Claude Code session transcripts
- CLIClient: LLM client using Claude CLI subscription (no API keys needed)

Quick Start:
    # Setup (one time)
    ace-learn setup

    # Use Claude Code normally - ACE learns via async hooks

Commands:
    ace-learn setup      - Configure Claude Code async hook
    ace-learn doctor     - Verify prerequisites
    ace-learn insights   - Show learned strategies
    ace-learn clear      - Clear all learned strategies
"""

from .hook import (
    ACEHookLearner,
    TranscriptParser,
    ParsedTranscript,
    SkillGenerator,
    find_project_root,
    get_project_skill_dir,
    NotInProjectError,
    DEFAULT_MARKERS,
)

from .cli_client import CLIClient, CLIClientError

__all__ = [
    # Main classes
    "ACEHookLearner",
    "TranscriptParser",
    "ParsedTranscript",
    "SkillGenerator",
    "CLIClient",
    "CLIClientError",
    # Utilities
    "find_project_root",
    "get_project_skill_dir",
    "NotInProjectError",
    "DEFAULT_MARKERS",
]
