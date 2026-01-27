"""Skill improvement orchestrator for workflow playbooks.

This module provides the SkillImprover class which orchestrates the
3-LLM-call process for improving skill files based on session transcripts:

1. Reflector: Extract learnings from transcript
2. SkillManager: Produce skill operations
3. Publisher: Apply operations to produce minimal, high-signal edits
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .models import (
    AcceptedChange,
    PublishOutput,
    RejectedLearning,
    SkillImprovementResult,
)
from .transcript import (
    ParsedTranscript,
    TranscriptParser,
    render_full_chat,
    get_transcript_summary,
)

logger = logging.getLogger(__name__)

# Constraints for Publisher
MAX_LINES_ADDED = 20  # Maximum new lines the Publisher can add
MAX_DIFF_PERCENT = 30  # Reject if diff exceeds this percentage of original

# Publisher prompt template
PUBLISHER_PROMPT = """You are a skill file editor for workflow playbooks. Your job is to apply
learnings from a coding session to improve a skill file while maintaining high quality.

## Current Skill File
```markdown
{current_skill}
```

## Extracted Learnings from Session
{learnings_summary}

## Proposed Operations from SkillManager
{operations_summary}

## Hard Constraints
1. Preserve YAML frontmatter validity (keep the --- delimiters and valid YAML)
2. Maximum {max_lines} new lines can be added
3. No duplicate strategies (check for semantic duplicates, not just exact matches)
4. Reject session-specific quirks that won't generalize
5. Keep strategies atomic - one clear insight per bullet point
6. Maintain existing structure and formatting style

## Task
Review the proposed operations and decide which to apply. For each:
- ACCEPT if it's a genuinely useful, generalizable insight
- REJECT if it's session-specific, duplicate, or low quality

Produce the updated skill file content with your decisions.

Return JSON:
{{
    "updated_skill_text": "full content of updated skill file",
    "accepted": [
        {{"change": "description", "reason": "why accepted", "evidence": "from transcript"}}
    ],
    "rejected": [
        {{"learning": "the rejected learning", "reason": "why rejected"}}
    ]
}}
"""


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomic write: write temp file then os.replace()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
        ) as tmp:
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, path)
    finally:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _create_backup(skill_path: Path) -> Path:
    """Create a timestamped backup of the skill file."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_name = f"{skill_path.stem}.{timestamp}.backup{skill_path.suffix}"
    backup_path = skill_path.parent / backup_name

    if skill_path.exists():
        content = skill_path.read_text(encoding="utf-8")
        _atomic_write_text(backup_path, content)
        logger.info(f"Created backup: {backup_path}")

    return backup_path


def _compute_diff(original: str, updated: str, filename: str = "skill.md") -> str:
    """Compute unified diff between original and updated content."""
    original_lines = original.splitlines(keepends=True)
    updated_lines = updated.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        updated_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
    )

    return "".join(diff)


def _normalize_for_comparison(text: str) -> str:
    """Normalize text for duplicate checking."""
    # Lowercase, strip whitespace, remove punctuation
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _check_diff_threshold(original: str, updated: str, max_percent: float) -> bool:
    """
    Check if the diff exceeds the allowed threshold.

    Returns True if diff is acceptable, False if too large.
    """
    if not original:
        return True  # No original, any update is fine

    original_lines = len(original.splitlines())
    diff = _compute_diff(original, updated)

    # Count added/removed lines
    added = sum(1 for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff.splitlines() if line.startswith("-") and not line.startswith("---"))

    total_changes = added + removed
    change_percent = (total_changes / max(original_lines, 1)) * 100

    if change_percent > max_percent:
        logger.warning(
            f"Diff exceeds threshold: {change_percent:.1f}% > {max_percent}%"
        )
        return False

    return True


def _extract_json(text: str) -> str:
    """Extract JSON object from response text."""
    text = text.strip()

    # If it already looks like JSON, return as-is
    if text.startswith("{") and text.endswith("}"):
        return text

    # Try to find JSON object in the text
    start = text.find("{")
    if start == -1:
        logger.warning("No JSON object found in response")
        return text

    # Find matching closing brace
    depth = 0
    end = start
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if depth != 0:
        logger.warning("Unbalanced braces in JSON")
        return text[start:]

    return text[start : end + 1]


class SkillImprover:
    """
    Orchestrator for improving skill files based on session transcripts.

    Uses the ACE 3-LLM-call pattern:
    1. Reflector: Extract learnings from transcript
    2. SkillManager: Produce skill operations
    3. Publisher: Apply operations to produce minimal, high-signal edits

    All LLM calls use Claude Code subscription via CLIClient.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        max_lines_added: int = MAX_LINES_ADDED,
        max_diff_percent: float = MAX_DIFF_PERCENT,
    ):
        """
        Initialize the SkillImprover.

        Args:
            llm_client: LLM client to use (default: creates CLIClient)
            max_lines_added: Maximum new lines the Publisher can add
            max_diff_percent: Reject if diff exceeds this percentage
        """
        self.max_lines_added = max_lines_added
        self.max_diff_percent = max_diff_percent

        # Create LLM client if not provided
        if llm_client is None:
            from ..claude_code.cli_client import CLIClient

            cli_path = os.environ.get("ACE_CLI_PATH")
            self.llm = CLIClient(cli_path=cli_path)
            logger.info("Using Claude Code CLI for skill improvement")
        else:
            self.llm = llm_client

        # Import ACE roles
        from ...roles import Reflector, SkillManager, AgentOutput
        from ...prompts_v2_1 import PromptManager
        from ...skillbook import Skillbook

        prompt_mgr = PromptManager()
        self.reflector = Reflector(
            self.llm, prompt_template=prompt_mgr.get_reflector_prompt()
        )
        self.skill_manager = SkillManager(
            self.llm, prompt_template=prompt_mgr.get_skill_manager_prompt()
        )

        # Store references for creating inputs
        self._AgentOutput = AgentOutput
        self._Skillbook = Skillbook

        self.transcript_parser = TranscriptParser()

    def improve(
        self,
        skill_path: str,
        transcript_path: str,
        dry_run: bool = True,
    ) -> SkillImprovementResult:
        """
        Improve a skill file based on a session transcript.

        Args:
            skill_path: Path to the skill file to improve
            transcript_path: Path to the transcript JSONL file
            dry_run: If True, don't write changes (default: True, safe)

        Returns:
            SkillImprovementResult with all outputs and diff
        """
        skill_file = Path(skill_path)
        transcript_file = Path(transcript_path)

        # Read inputs
        if not skill_file.exists():
            raise FileNotFoundError(f"Skill file not found: {skill_path}")

        original_content = skill_file.read_text(encoding="utf-8")

        # Parse transcript
        logger.info(f"Parsing transcript: {transcript_path}")
        transcript = self.transcript_parser.parse(transcript_path)
        summary = get_transcript_summary(transcript)
        logger.info(
            f"Transcript: {summary['turns']} turns, "
            f"{summary['total_tool_calls']} tool calls, "
            f"{summary['success_rate']:.0f}% success"
        )

        # Step 1: Reflector
        logger.info("Running Reflector...")
        reflector_output = self._run_reflector(transcript)
        logger.info(f"Extracted {len(reflector_output.extracted_learnings)} learnings")

        # Step 2: SkillManager
        logger.info("Running SkillManager...")
        skill_manager_output = self._run_skill_manager(
            reflector_output, transcript.cwd
        )
        ops_count = len(skill_manager_output.update.operations)
        logger.info(f"SkillManager proposed {ops_count} operations")

        # Step 3: Publisher
        logger.info("Running Publisher...")
        publish_output = self._run_publisher(
            original_content, reflector_output, skill_manager_output
        )
        logger.info(
            f"Publisher accepted {len(publish_output.accepted)}, "
            f"rejected {len(publish_output.rejected)}"
        )

        # Compute diff
        updated_content = publish_output.updated_skill_text
        diff = _compute_diff(original_content, updated_content, skill_file.name)
        has_changes = bool(diff.strip())

        # Validate diff threshold
        if has_changes and not _check_diff_threshold(
            original_content, updated_content, self.max_diff_percent
        ):
            logger.warning("Diff exceeds threshold, rejecting all changes")
            updated_content = original_content
            diff = ""
            has_changes = False
            # Add rejection reason
            publish_output.rejected.append(
                RejectedLearning(
                    learning="All proposed changes",
                    reason=f"Diff exceeded {self.max_diff_percent}% threshold",
                )
            )
            publish_output.accepted = []
            publish_output.updated_skill_text = original_content

        # Apply changes if not dry run
        backup_path = None
        if not dry_run and has_changes:
            backup_path = _create_backup(skill_file)
            _atomic_write_text(skill_file, updated_content)
            logger.info(f"Updated skill file: {skill_path}")

        return SkillImprovementResult(
            reflector_output=reflector_output.model_dump(),
            skill_manager_output=skill_manager_output.model_dump(),
            publish_output=publish_output,
            original_content=original_content,
            updated_content=updated_content,
            diff=diff,
            has_changes=has_changes,
            transcript_path=transcript_path,
            skill_path=skill_path,
            backup_path=str(backup_path) if backup_path else None,
        )

    def _run_reflector(self, transcript: ParsedTranscript):
        """Run the Reflector to extract learnings from transcript."""
        # Create AgentOutput from transcript
        execution_trace = self._transcript_to_execution_trace(transcript)

        agent_output = self._AgentOutput(
            reasoning=execution_trace,
            final_answer=(
                transcript.turns[-1].assistant_text if transcript.turns else ""
            ),
            skill_ids=[],
            raw={
                "total_tools": transcript.total_tool_calls,
                "failed_tools": transcript.failed_tool_calls,
            },
        )

        # Get task from last user prompt
        task = "Claude Code session"
        for turn in reversed(transcript.turns):
            if turn.user_prompt:
                task = turn.user_prompt[:500]
                break

        # Generate feedback
        total = transcript.total_tool_calls
        failed = transcript.failed_tool_calls
        success_rate = ((total - failed) / total * 100) if total > 0 else 100
        feedback = f"Session completed: {total} tool calls, {success_rate:.0f}% success rate"
        if failed > 0:
            feedback += f" ({failed} failures)"

        return self.reflector.reflect(
            question=task,
            agent_output=agent_output,
            skillbook=self._Skillbook(),  # Empty skillbook for new learning
            ground_truth=None,
            feedback=feedback,
        )

    def _run_skill_manager(self, reflection, cwd: str):
        """Run the SkillManager to produce skill operations."""
        total = sum(1 for l in reflection.extracted_learnings if l.learning)
        progress = f"{total} learnings extracted"

        return self.skill_manager.update_skills(
            reflection=reflection,
            skillbook=self._Skillbook(),  # Empty skillbook
            question_context=f"Claude Code session in {cwd}",
            progress=progress,
        )

    def _run_publisher(
        self, current_skill: str, reflector_output, skill_manager_output
    ) -> PublishOutput:
        """Run the Publisher to produce final skill file edits."""
        # Format learnings summary
        learnings_lines = []
        for learning in reflector_output.extracted_learnings:
            if learning.learning:
                score = f"(atomicity: {learning.atomicity_score:.1f})"
                learnings_lines.append(f"- {learning.learning} {score}")
        learnings_summary = "\n".join(learnings_lines) or "(no learnings extracted)"

        # Format operations summary
        ops_lines = []
        for op in skill_manager_output.update.operations:
            ops_lines.append(f"- {op.type}: {op.content or op.skill_id or 'N/A'}")
        operations_summary = "\n".join(ops_lines) or "(no operations proposed)"

        # Build prompt
        prompt = PUBLISHER_PROMPT.format(
            current_skill=current_skill,
            learnings_summary=learnings_summary,
            operations_summary=operations_summary,
            max_lines=self.max_lines_added,
        )

        # Get structured response
        response = self.llm.complete_structured(prompt, PublishOutput)
        return response

    def _transcript_to_execution_trace(self, transcript: ParsedTranscript) -> str:
        """Convert transcript to execution trace format for Reflector."""
        parts = []
        step_num = 0

        for turn in transcript.turns:
            if turn.user_prompt:
                prompt_preview = turn.user_prompt[:200]
                if len(turn.user_prompt) > 200:
                    prompt_preview += "..."
                parts.append(f"[User] {prompt_preview}")

            if turn.assistant_text:
                parts.append(f"[Reasoning] {turn.assistant_text[:300]}")

            for tool in turn.tool_calls:
                step_num += 1
                status = "FAIL" if tool.is_error else "OK"

                # Format tool call based on type
                if tool.name in ["Read", "Glob", "Grep"]:
                    target = tool.input.get("file_path") or tool.input.get(
                        "pattern", ""
                    )
                    parts.append(f"[Step {step_num}] {status} {tool.name}: {target}")
                elif tool.name in ["Write", "Edit"]:
                    target = tool.input.get("file_path", "")
                    parts.append(f"[Step {step_num}] {status} {tool.name}: {target}")
                elif tool.name == "Bash":
                    cmd = tool.input.get("command", "")[:80]
                    parts.append(f"[Step {step_num}] {status} Bash: {cmd}")
                elif tool.name == "Task":
                    desc = tool.input.get("description", "")
                    parts.append(f"[Step {step_num}] {status} Task: {desc}")
                else:
                    parts.append(f"[Step {step_num}] {status} {tool.name}")

                if tool.is_error and tool.result:
                    error_preview = tool.result[:100]
                    parts.append(f"    Error: {error_preview}")

        return "\n".join(parts) if parts else "(No trace captured)"
