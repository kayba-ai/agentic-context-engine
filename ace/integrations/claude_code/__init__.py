"""Claude Code integration for ACE - subscription-only learning from Claude Code sessions.

This package provides:
- ACELearner: Learns from Claude Code session transcripts
- CLIClient: LLM client using Claude CLI subscription (no API keys needed)
- update_claude_md: Updates CLAUDE.md with TOON-compressed skillbook
- get_patch_info: Diagnostics for the patched CLI status

Quick Start:
    # Learn from your latest Claude Code session
    ace-learn

    # Check prerequisites
    ace-learn doctor

Commands:
    ace-learn              - Learn from latest transcript, update CLAUDE.md
    ace-learn --lines N    - Learn from last N lines only
    ace-learn doctor       - Verify prerequisites
    ace-learn insights     - Show learned strategies
    ace-learn clear        - Clear all learned strategies
"""

from .learner import (
    ACELearner,
    ACEHookLearner,  # Backwards compatibility alias
    find_project_root,
    find_latest_transcript,
    update_claude_md,
    NotInProjectError,
    DEFAULT_MARKERS,
)

from .cli_client import CLIClient, CLIClientError

from .prompt_patcher import (
    get_patch_info,
    is_patched_cli_available,
    get_or_create_patched_cli,
)

__all__ = [
    # Main classes
    "ACELearner",
    "ACEHookLearner",  # Backwards compatibility
    "CLIClient",
    "CLIClientError",
    # Utilities
    "find_project_root",
    "find_latest_transcript",
    "update_claude_md",
    "NotInProjectError",
    "DEFAULT_MARKERS",
    # Prompt patcher utilities
    "get_patch_info",
    "is_patched_cli_available",
    "get_or_create_patched_cli",
]
