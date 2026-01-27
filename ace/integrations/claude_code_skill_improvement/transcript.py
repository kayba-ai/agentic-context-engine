"""Transcript parsing and rendering for skill improvement.

This module provides utilities for parsing Claude Code JSONL transcripts
and rendering them as full chat history for LLM prompting.

The TranscriptParser is reused from the claude_code hook module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum transcript size in bytes (v1: fail fast on oversized transcripts)
MAX_TRANSCRIPT_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_TRANSCRIPT_LINES = 50000  # Maximum JSONL lines


class TranscriptTooLargeError(Exception):
    """Raised when transcript exceeds size limits."""

    def __init__(self, path: str, size: int, limit: int):
        self.path = path
        self.size = size
        self.limit = limit
        super().__init__(
            f"Transcript too large: {size:,} bytes (limit: {limit:,} bytes). "
            f"File: {path}"
        )


@dataclass
class ToolCall:
    """A single tool call from the transcript."""

    name: str
    input: Dict[str, Any]
    tool_use_id: str
    result: Optional[str] = None
    is_error: bool = False


@dataclass
class Turn:
    """A single conversation turn (user prompt + assistant response)."""

    user_prompt: Optional[str]
    assistant_text: str
    tool_calls: List[ToolCall]
    timestamp: Optional[str] = None


@dataclass
class ParsedTranscript:
    """Parsed Claude Code session transcript."""

    session_id: str
    turns: List[Turn]
    cwd: str
    total_tool_calls: int
    successful_tool_calls: int
    failed_tool_calls: int


class TranscriptParser:
    """
    Parse Claude Code JSONL transcript files.

    Adapted from the ace.integrations.claude_code.hook module.
    """

    def __init__(
        self,
        max_size_bytes: int = MAX_TRANSCRIPT_SIZE_BYTES,
        max_lines: int = MAX_TRANSCRIPT_LINES,
    ):
        """
        Initialize parser with size limits.

        Args:
            max_size_bytes: Maximum transcript file size in bytes
            max_lines: Maximum number of JSONL lines to process
        """
        self.max_size_bytes = max_size_bytes
        self.max_lines = max_lines

    def parse(self, transcript_path: str) -> ParsedTranscript:
        """
        Parse a Claude Code transcript JSONL file.

        Args:
            transcript_path: Path to the .jsonl transcript file

        Returns:
            ParsedTranscript with structured conversation data

        Raises:
            FileNotFoundError: If transcript file doesn't exist
            TranscriptTooLargeError: If transcript exceeds size limits
        """
        path = Path(transcript_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")

        # Check file size
        file_size = path.stat().st_size
        if file_size > self.max_size_bytes:
            raise TranscriptTooLargeError(
                transcript_path, file_size, self.max_size_bytes
            )

        entries = []
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line_num > self.max_lines:
                    logger.warning(
                        f"Transcript exceeded max lines ({self.max_lines}), "
                        f"truncating at line {line_num}"
                    )
                    break

                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return self._process_entries(entries)

    def _process_entries(self, entries: List[Dict[str, Any]]) -> ParsedTranscript:
        """Process raw JSONL entries into structured transcript."""
        session_id = ""
        cwd = ""
        turns: List[Turn] = []

        # Track current turn state
        current_user_prompt: Optional[str] = None
        current_assistant_text = ""
        current_tool_calls: List[ToolCall] = []
        pending_tool_results: Dict[str, Any] = {}  # tool_use_id -> result

        total_tools = 0
        failed_tools = 0

        for entry in entries:
            entry_type = entry.get("type", "")

            # Extract session metadata
            if not session_id:
                session_id = entry.get("sessionId", "")
            if not cwd:
                cwd = entry.get("cwd", "")

            if entry_type == "user":
                # If we have accumulated data, save the turn
                if current_assistant_text or current_tool_calls:
                    turns.append(
                        Turn(
                            user_prompt=current_user_prompt,
                            assistant_text=current_assistant_text,
                            tool_calls=current_tool_calls,
                            timestamp=entry.get("timestamp"),
                        )
                    )
                    current_assistant_text = ""
                    current_tool_calls = []

                # Process user message content
                message = entry.get("message", {})
                content = message.get("content", [])

                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            # Regular user prompt
                            text = block.get("text", "")
                            # Skip system injected content
                            if not text.startswith("<ide_") and not text.startswith(
                                "<system"
                            ):
                                current_user_prompt = text
                        elif block.get("type") == "tool_result":
                            # Tool result - match to pending tool call
                            tool_use_id = block.get("tool_use_id", "")
                            is_error = block.get("is_error", False)
                            result_content = block.get("content", "")

                            # Store for matching
                            pending_tool_results[tool_use_id] = {
                                "result": (
                                    result_content
                                    if isinstance(result_content, str)
                                    else str(result_content)[:500]
                                ),
                                "is_error": is_error,
                            }

                            if is_error:
                                failed_tools += 1

            elif entry_type == "assistant":
                message = entry.get("message", {})
                content = message.get("content", [])

                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text = block.get("text", "")
                            if text.strip():
                                current_assistant_text = text
                        elif block.get("type") == "tool_use":
                            tool_use_id = block.get("id", "")
                            tool_call = ToolCall(
                                name=block.get("name", "unknown"),
                                input=block.get("input", {}),
                                tool_use_id=tool_use_id,
                            )

                            # Check if we have a result for this tool
                            if tool_use_id in pending_tool_results:
                                result_info = pending_tool_results[tool_use_id]
                                tool_call.result = result_info["result"]
                                tool_call.is_error = result_info["is_error"]

                            current_tool_calls.append(tool_call)
                            total_tools += 1

        # Don't forget the last turn
        if current_assistant_text or current_tool_calls:
            turns.append(
                Turn(
                    user_prompt=current_user_prompt,
                    assistant_text=current_assistant_text,
                    tool_calls=current_tool_calls,
                )
            )

        return ParsedTranscript(
            session_id=session_id,
            turns=turns,
            cwd=cwd,
            total_tool_calls=total_tools,
            successful_tool_calls=total_tools - failed_tools,
            failed_tool_calls=failed_tools,
        )


def render_full_chat(transcript: ParsedTranscript) -> str:
    """
    Render a parsed transcript as a full chat history string.

    This format is suitable for LLM prompting where the full context
    of the conversation needs to be understood.

    Args:
        transcript: Parsed transcript to render

    Returns:
        Human-readable chat history string
    """
    parts = []
    parts.append(f"# Claude Code Session Transcript")
    parts.append(f"Session ID: {transcript.session_id}")
    parts.append(f"Working Directory: {transcript.cwd}")
    parts.append(
        f"Tool Calls: {transcript.total_tool_calls} total, "
        f"{transcript.successful_tool_calls} successful, "
        f"{transcript.failed_tool_calls} failed"
    )
    parts.append("")
    parts.append("---")
    parts.append("")

    for i, turn in enumerate(transcript.turns, 1):
        parts.append(f"## Turn {i}")
        parts.append("")

        if turn.user_prompt:
            parts.append("### User")
            parts.append(turn.user_prompt)
            parts.append("")

        if turn.assistant_text:
            parts.append("### Assistant")
            parts.append(turn.assistant_text)
            parts.append("")

        if turn.tool_calls:
            parts.append("### Tool Calls")
            for tool in turn.tool_calls:
                status = "ERROR" if tool.is_error else "OK"
                parts.append(f"- **{tool.name}** [{status}]")

                # Format tool input based on type
                if tool.name in ["Read", "Write", "Edit"]:
                    file_path = tool.input.get("file_path", "")
                    parts.append(f"  - File: `{file_path}`")
                elif tool.name == "Bash":
                    command = tool.input.get("command", "")
                    # Truncate long commands
                    if len(command) > 200:
                        command = command[:200] + "..."
                    parts.append(f"  - Command: `{command}`")
                elif tool.name in ["Glob", "Grep"]:
                    pattern = tool.input.get("pattern", "")
                    parts.append(f"  - Pattern: `{pattern}`")
                elif tool.name == "Task":
                    desc = tool.input.get("description", "")
                    parts.append(f"  - Description: {desc}")
                else:
                    # Generic handling for other tools
                    for key, value in list(tool.input.items())[:3]:
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        parts.append(f"  - {key}: {value_str}")

                # Include error message if failed
                if tool.is_error and tool.result:
                    error_preview = tool.result[:200]
                    if len(tool.result) > 200:
                        error_preview += "..."
                    parts.append(f"  - Error: {error_preview}")

            parts.append("")

        parts.append("---")
        parts.append("")

    return "\n".join(parts)


def get_transcript_summary(transcript: ParsedTranscript) -> Dict[str, Any]:
    """
    Get a summary of the transcript for logging/display.

    Args:
        transcript: Parsed transcript to summarize

    Returns:
        Dictionary with summary statistics
    """
    # Collect unique tools used
    tools_used = set()
    for turn in transcript.turns:
        for tool in turn.tool_calls:
            tools_used.add(tool.name)

    # Count user messages
    user_messages = sum(1 for t in transcript.turns if t.user_prompt)

    return {
        "session_id": transcript.session_id,
        "cwd": transcript.cwd,
        "turns": len(transcript.turns),
        "user_messages": user_messages,
        "total_tool_calls": transcript.total_tool_calls,
        "successful_tool_calls": transcript.successful_tool_calls,
        "failed_tool_calls": transcript.failed_tool_calls,
        "success_rate": (
            transcript.successful_tool_calls / transcript.total_tool_calls * 100
            if transcript.total_tool_calls > 0
            else 100.0
        ),
        "tools_used": sorted(tools_used),
    }
