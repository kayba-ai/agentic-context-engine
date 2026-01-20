"""Claude Code integration for ACE - subscription-only learning from Claude Code sessions.

This package provides:
- ACEHookLearner: Learns from Claude Code session transcripts
- CLIClient: LLM client using Claude CLI subscription (no API keys needed)
- Daemon: Background processor for async learning
- Doctor: Verification tool for prerequisites

Quick Start:
    # Setup (one time)
    ace-learn setup
    ace-daemon start

    # Use Claude Code normally - ACE learns in background

Commands:
    ace-learn setup      - Configure Claude Code hook
    ace-learn doctor     - Verify prerequisites
    ace-learn insights   - Show learned strategies
    ace-daemon start     - Start background processor
    ace-daemon status    - Check daemon status
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
