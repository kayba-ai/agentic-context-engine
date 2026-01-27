"""Claude Code Skill Improvement Integration.

This module provides tools for retrospectively improving workflow playbook
skill files based on Claude Code session transcripts.

The process uses ACE's standard 3-LLM-call pattern:
1. Reflector: Extract learnings from transcript
2. SkillManager: Produce skill operations
3. Publisher: Apply minimal, high-signal edits

Usage:
    # CLI
    ace-skill-improve run --skill SKILL.md --transcript session.jsonl
    ace-skill-improve run --skill SKILL.md --transcript session.jsonl --apply

    # Python
    from ace.integrations.claude_code_skill_improvement import (
        SkillImprover,
        render_full_chat,
    )

    improver = SkillImprover()
    result = improver.improve(
        skill_path="path/to/SKILL.md",
        transcript_path="path/to/session.jsonl",
        dry_run=True,
    )
    print(result.diff)
"""

from .improver import SkillImprover
from .models import (
    AcceptedChange,
    PublishOutput,
    RejectedLearning,
    SkillImprovementResult,
)
from .transcript import (
    ParsedTranscript,
    TranscriptParser,
    TranscriptTooLargeError,
    Turn,
    ToolCall,
    render_full_chat,
    get_transcript_summary,
)
from .cli import create_slash_commands

__all__ = [
    # Main class
    "SkillImprover",
    # Models
    "AcceptedChange",
    "PublishOutput",
    "RejectedLearning",
    "SkillImprovementResult",
    # Transcript utilities
    "ParsedTranscript",
    "TranscriptParser",
    "TranscriptTooLargeError",
    "Turn",
    "ToolCall",
    "render_full_chat",
    "get_transcript_summary",
    # Setup utilities
    "create_slash_commands",
]
