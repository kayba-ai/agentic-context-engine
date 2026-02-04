"""
Claude Code ACE Learning - Simple transcript-based learning.

This module enables ACE learning from Claude Code sessions by reading
existing transcript files directly. Learnings are stored directly in
CLAUDE.md using TOON compression for efficient context transfer.

Usage:
    1. Use Claude Code normally
    2. Run /ace-learn (or `ace-learn`) to learn from the session
    3. CLAUDE.md is updated with learned strategies

Commands:
    ace-learn              # Learn from latest transcript, update CLAUDE.md
    ace-learn --lines 500  # Learn from last 500 lines only (optional)
    ace-learn insights     # Show learned strategies
    ace-learn remove <id>  # Remove specific insight
    ace-learn clear --confirm  # Clear all insights
    ace-learn doctor       # Check prerequisites

Storage:
    - CLAUDE.md: TOON-compressed skillbook (for Claude Code to read)
    - .ace/skillbook.json: Full skillbook (for persistence)
"""

import json
import re
import sys
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Any

from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

# Load .env file from ~/.ace/.env or current directory
_env_paths = [
    Path.home() / ".ace" / ".env",
    Path.cwd() / ".env",
]
for _env_path in _env_paths:
    if _env_path.exists():
        load_dotenv(_env_path)
        break

from ...skillbook import Skillbook
from ...roles import Reflector, SkillManager, AgentOutput
from ...prompts_v2_1 import PromptManager
from .cli_client import CLIClient

logger = logging.getLogger(__name__)


# ============================================================================
# Project Root Detection
# ============================================================================

DEFAULT_MARKERS = [
    ".ace-root",  # Explicit ACE project root marker (highest priority for monorepos)
    ".git",
    ".hg",
    ".svn",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
]


class NotInProjectError(Exception):
    """Raised when no project root can be found."""

    def __init__(self, searched_path: str):
        self.searched_path = searched_path

    def __str__(self):
        return (
            f"error: not in a project directory\n"
            f"  searched from: {self.searched_path}\n"
            f"  looking for: .ace-root, .git, pyproject.toml, package.json, etc.\n\n"
            f"hint: run from within a project directory, or use\n"
            f"      --project <path> to specify project root\n"
            f"      or set ACE_PROJECT_DIR environment variable\n"
            f"      or create .ace-root file at monorepo root"
        )


def find_project_root(
    start: Path, markers: Optional[List[str]] = None
) -> Optional[Path]:
    """Find project root by walking up from start directory."""
    if env_dir := os.environ.get("ACE_PROJECT_DIR"):
        env_path = Path(env_dir).expanduser().resolve()
        if env_path.exists() and env_path.is_dir():
            logger.debug(f"Using ACE_PROJECT_DIR: {env_path}")
            return env_path
        else:
            logger.warning(f"ACE_PROJECT_DIR set but invalid: {env_dir}")

    markers = markers or DEFAULT_MARKERS
    start_resolved = start.resolve()

    for marker in markers:
        current = start_resolved
        while True:
            if (current / marker).exists():
                return current
            if current.parent == current:
                break
            current = current.parent

    return None


# ============================================================================
# Transcript Discovery
# ============================================================================


def find_latest_session_from_history(
    project_path: Optional[Path] = None,
) -> Optional[Path]:
    """
    Find the most recently USED transcript via history.jsonl.

    This correctly handles /resume sessions where an old transcript is
    reactivated but its file mtime may not update.

    Args:
        project_path: Optional project path to filter by

    Returns:
        Path to the most recently used transcript, or None if not found
    """
    history_file = Path.home() / ".claude" / "history.jsonl"
    if not history_file.exists():
        return None

    # Normalize project path for comparison
    if project_path:
        project_str = str(project_path.resolve())
    else:
        project_str = None

    # Find most recent entry (optionally filtered by project)
    latest_entry = None
    latest_timestamp = 0

    with history_file.open() as f:
        for line in f:
            try:
                entry = json.loads(line)
                timestamp = entry.get("timestamp", 0)
                entry_project = entry.get("project")

                # Filter by project if specified
                if project_str and entry_project != project_str:
                    continue

                if timestamp > latest_timestamp:
                    latest_timestamp = timestamp
                    latest_entry = entry
            except json.JSONDecodeError:
                continue

    if not latest_entry:
        return None

    # Build transcript path from sessionId
    session_id = latest_entry.get("sessionId")
    project = latest_entry.get("project", "")
    if not session_id or not project:
        return None

    # Convert project path to Claude's directory format
    # /Users/foo/bar -> -Users-foo-bar (keeps leading dash)
    project_dir_name = project.replace("/", "-")
    transcript_path = (
        Path.home() / ".claude" / "projects" / project_dir_name / f"{session_id}.jsonl"
    )

    if transcript_path.exists():
        return transcript_path
    return None


def find_latest_transcript(project_path: Optional[Path] = None) -> Optional[Path]:
    """
    Find the latest Claude Code transcript file.

    First tries history.jsonl for accurate "most recently used" detection,
    which correctly handles /resume sessions. Falls back to file modification
    time if history unavailable.

    Args:
        project_path: Optional project path to filter by

    Returns:
        Path to the latest transcript, or None if not found
    """
    # Try history.jsonl first (handles /resume correctly)
    transcript = find_latest_session_from_history(project_path)
    if transcript:
        return transcript

    # Fallback to mtime-based discovery
    claude_dir = Path.home() / ".claude" / "projects"
    if not claude_dir.exists():
        return None

    # Find all .jsonl files recursively
    transcripts = list(claude_dir.rglob("*.jsonl"))
    if not transcripts:
        return None

    # Return the most recently modified
    return max(transcripts, key=lambda p: p.stat().st_mtime)


def _extract_session_id(transcript_path: Path) -> Optional[str]:
    """Extract session ID from the transcript filename or first entry."""
    # The filename is typically the session ID
    session_id = transcript_path.stem

    # Validate by checking first entry
    try:
        with transcript_path.open() as f:
            first_line = f.readline()
            if first_line:
                entry = json.loads(first_line)
                return entry.get("sessionId", session_id)
    except (json.JSONDecodeError, IOError):
        pass

    return session_id


def _count_transcript_lines(transcript_path: Path) -> int:
    """Count total lines in transcript file."""
    with transcript_path.open() as f:
        return sum(1 for _ in f)


def _extract_cwd_from_transcript(transcript_path: Path) -> Optional[str]:
    """Extract cwd from the first entry of the transcript."""
    try:
        with transcript_path.open() as f:
            first_line = f.readline()
            if first_line:
                entry = json.loads(first_line)
                return entry.get("cwd")
    except (json.JSONDecodeError, IOError):
        pass
    return None


# ============================================================================
# TOON Transcript Processing
# ============================================================================


# Entry types that are not relevant for learning
_SKIP_ENTRY_TYPES = {
    "system",  # Claude Code's massive system prompt
    "file-history-snapshot",  # File backup metadata
    "file-restore",  # File restore events
    "summary",  # Conversation summaries
    "progress",  # Streaming progress indicators (tool execution updates)
    "queue-operation",  # Internal queue operations
    "thinking",  # Claude's internal reasoning (redundant with actions)
}

# Patterns that indicate ace-learn recursive content (from previous runs)
# These create malformed prompts when fed back to Claude CLI
_ACE_LEARN_PATTERNS = (
    "ace-learn",
    "ace.integrations.claude_code",
    "CLIClientError",
    "CLI returned error: Exit code",
    "Learning from transcript",
    "✓ Learning complete",
    "✗ Learning failed",
    "Running Reflector",
    "Running SkillManager",
    "ACE Doctor",
)


def _contains_ace_learn_content(text: str) -> bool:
    """Check if text contains ace-learn recursive content."""
    return any(pattern in text for pattern in _ACE_LEARN_PATTERNS)


# Tool result compression settings
MAX_RESULT_SIZE = 1000  # chars
HEAD_SIZE = 500
TAIL_SIZE = 200


def _compress_tool_result(content: str) -> str:
    """Truncate large tool results, keeping head and tail."""
    if not isinstance(content, str) or len(content) <= MAX_RESULT_SIZE:
        return content
    truncated = len(content) - HEAD_SIZE - TAIL_SIZE
    return f"{content[:HEAD_SIZE]}\n... [{truncated} chars truncated] ...\n{content[-TAIL_SIZE:]}"


def _filter_transcript_entry(entry: dict) -> Optional[dict]:
    """
    Filter a transcript entry to remove Claude Code meta-content.

    Removes:
    - System prompt entries (type: "system") - contains Claude Code's massive instructions
    - File history snapshots and restores (metadata only)
    - Summary entries (conversation summaries)
    - <system-reminder> blocks in user/assistant messages
    - <ide_*> prefixed blocks (IDE-injected content)
    - ace-learn recursive content (previous run output)
    - Metadata fields not needed for learning (uuids, timestamps, etc.)

    Returns:
        Filtered entry with only essential fields, or None if entry should be skipped.
    """
    entry_type = entry.get("type")

    # Skip meta entries that aren't relevant for learning
    if entry_type in _SKIP_ENTRY_TYPES:
        return None

    # For user/assistant messages, filter content blocks
    if entry_type in ("user", "assistant"):
        message = entry.get("message", {})
        content = message.get("content", [])

        # Handle string content (simple text messages)
        if isinstance(content, str):
            # Strip <system-reminder> blocks
            filtered_text = re.sub(
                r"<system-reminder>.*?</system-reminder>", "", content, flags=re.DOTALL
            )
            # Skip <ide_*> prefixed content
            if filtered_text.strip().startswith("<ide_"):
                return None
            # Skip ace-learn recursive content
            if _contains_ace_learn_content(filtered_text):
                return None
            if not filtered_text.strip():
                return None
            # Return minimal entry - strip metadata not needed for learning
            return {"type": entry_type, "content": filtered_text}

        # Handle list content (structured blocks)
        if isinstance(content, list):
            filtered_content = []
            for block in content:
                # Skip thinking blocks (Claude's internal reasoning - redundant with actions)
                if isinstance(block, dict) and block.get("type") == "thinking":
                    continue
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    # Strip <system-reminder> blocks
                    text = re.sub(
                        r"<system-reminder>.*?</system-reminder>",
                        "",
                        text,
                        flags=re.DOTALL,
                    )
                    # Skip <ide_*> prefixed blocks
                    if text.strip().startswith("<ide_"):
                        continue
                    # Skip ace-learn recursive content
                    if _contains_ace_learn_content(text):
                        continue
                    # Skip if empty after filtering
                    if not text.strip():
                        continue
                    filtered_content.append({**block, "text": text})
                elif isinstance(block, dict) and block.get("type") == "tool_result":
                    # Compress large tool results (but keep errors in full)
                    result_content = block.get("content", "")
                    if isinstance(result_content, str) and not block.get("is_error"):
                        result_content = _compress_tool_result(result_content)
                    filtered_content.append({**block, "content": result_content})
                else:
                    # Keep other blocks (tool_use, etc.)
                    filtered_content.append(block)

            if not filtered_content:
                return None

            # Return minimal entry - strip metadata not needed for learning
            return {"type": entry_type, "content": filtered_content}

    # Skip other entry types - they're not relevant for learning
    return None


def toon_transcript(transcript_path: Path, start_line: int = 0) -> str:
    """
    Read transcript .jsonl and convert to TOON format for Reflector.

    Filters out Claude Code meta-content:
    - System prompt entries (massive instruction set)
    - <system-reminder> blocks
    - <ide_*> prefixed content

    Args:
        transcript_path: Path to the .jsonl transcript
        start_line: Start reading from this line (0-indexed)

    Returns:
        TOON-encoded transcript entries (filtered)
    """
    # Collect and filter entries
    entries = []
    with transcript_path.open() as f:
        for i, line in enumerate(f):
            if i >= start_line and line.strip():
                try:
                    entry = json.loads(line)
                    # Filter out meta-content
                    filtered = _filter_transcript_entry(entry)
                    if filtered:
                        entries.append(filtered)
                except json.JSONDecodeError:
                    continue

    # Encode with TOON or fallback to JSON
    try:
        from toon import encode

        return encode(entries, {"delimiter": "\t"})
    except ImportError:
        logger.warning("TOON not installed, using compact JSON")
        return json.dumps(entries, separators=(",", ":"))


def _get_transcript_feedback(transcript_path: Path, start_line: int = 0) -> str:
    """Generate feedback string from transcript tool outcomes."""
    total_tools = 0
    failed_tools = 0

    with transcript_path.open() as f:
        for i, line in enumerate(f):
            if i < start_line or not line.strip():
                continue
            try:
                entry = json.loads(line)
                # Count tool results
                if entry.get("type") in ("user", "assistant"):
                    message = entry.get("message", {})
                    content = message.get("content", [])
                    if isinstance(content, list):
                        for block in content:
                            if (
                                isinstance(block, dict)
                                and block.get("type") == "tool_result"
                            ):
                                total_tools += 1
                                if block.get("is_error"):
                                    failed_tools += 1
            except json.JSONDecodeError:
                continue

    if total_tools == 0:
        return "Session completed: no tool calls recorded"

    success_rate = (total_tools - failed_tools) / total_tools * 100
    feedback = (
        f"Session completed: {total_tools} tool calls, {success_rate:.0f}% success rate"
    )
    if failed_tools > 0:
        feedback += f" ({failed_tools} failures)"
    return feedback


def _get_last_user_prompt(transcript_path: Path) -> str:
    """Extract the last user prompt from the transcript."""
    last_prompt = "Claude Code session"

    with transcript_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                if entry.get("type") == "user":
                    message = entry.get("message", {})
                    content = message.get("content", [])
                    if isinstance(content, str):
                        text = re.sub(
                            r"<system-reminder>.*?</system-reminder>",
                            "",
                            content,
                            flags=re.DOTALL,
                        ).strip()
                        if (
                            text
                            and not text.startswith("<ide_")
                            and not text.startswith("<system")
                        ):
                            last_prompt = text[:200]
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "")
                                # Skip system injected content
                                if not text.startswith("<ide_") and not text.startswith(
                                    "<system"
                                ):
                                    last_prompt = text[:200]
            except json.JSONDecodeError:
                continue

    return last_prompt


# ============================================================================
# CLAUDE.md Update
# ============================================================================

ACE_SECTION_HEADER = "## ACE Learned Strategies"
ACE_START_MARKER = "<!-- ACE:START - Do not edit manually -->"
ACE_END_MARKER = "<!-- ACE:END -->"


def update_claude_md(project_root: Path, skillbook: Skillbook) -> Path:
    """
    Update CLAUDE.md with TOON-compressed skillbook.

    Finds or creates CLAUDE.md at project root and updates the
    '## ACE Learned Strategies' section with TOON-compressed skills.

    Args:
        project_root: Path to project root directory
        skillbook: Skillbook to compress and write

    Returns:
        Path to the updated CLAUDE.md file
    """
    claude_md_path = project_root / "CLAUDE.md"

    # Generate TOON content
    skills = skillbook.skills()
    if skills:
        try:
            toon_content = skillbook.as_prompt()
        except ImportError:
            # Fallback to JSON if TOON not available
            skills_data = [s.to_llm_dict() for s in skills]
            toon_content = json.dumps({"skills": skills_data}, separators=(",", ":"))
            logger.warning("TOON not installed, using JSON fallback")
    else:
        toon_content = '{"skills": []}'

    # Build the ACE section
    ace_section = f"""{ACE_SECTION_HEADER}

{ACE_START_MARKER}
{toon_content}
{ACE_END_MARKER}"""

    # Read existing content or start fresh
    if claude_md_path.exists():
        existing_content = claude_md_path.read_text(encoding="utf-8")
    else:
        existing_content = f"# {project_root.name}\n\nProject documentation.\n"

    # Replace or append ACE section
    if ACE_START_MARKER in existing_content:
        # Replace existing section (between markers)
        pattern = (
            re.escape(ACE_SECTION_HEADER)
            + r"\s*\n\s*"
            + re.escape(ACE_START_MARKER)
            + r".*?"
            + re.escape(ACE_END_MARKER)
        )
        new_content = re.sub(pattern, ace_section, existing_content, flags=re.DOTALL)
    elif ACE_SECTION_HEADER in existing_content:
        # Section header exists but no markers - replace whole section until next ##
        pattern = re.escape(ACE_SECTION_HEADER) + r".*?(?=\n## |\Z)"
        new_content = re.sub(
            pattern, ace_section + "\n", existing_content, flags=re.DOTALL
        )
    else:
        # Append new section
        new_content = existing_content.rstrip() + "\n\n" + ace_section + "\n"

    # Write atomically
    _atomic_write_text(claude_md_path, new_content)
    logger.info(f"Updated CLAUDE.md at {claude_md_path}")

    return claude_md_path


# ============================================================================
# Skillbook Persistence
# ============================================================================


def _atomic_write_text(path: Path, content: str) -> None:
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


# ============================================================================
# Main Learner Class
# ============================================================================


class ACELearner:
    """
    Main class for learning from Claude Code sessions.

    Reads transcript files directly and uses TOON compression for efficient
    context transfer to the Reflector. Updates CLAUDE.md with learned strategies.

    Usage:
        learner = ACELearner(cwd="/path/to/project")
        learner.learn_from_transcript("/path/to/transcript.jsonl")
    """

    def __init__(
        self,
        cwd: str,
        skillbook_path: Optional[Path] = None,
        project_root: Optional[Path] = None,
        ace_llm: Optional[Any] = None,
    ):
        """
        Initialize the learner.

        Args:
            cwd: Working directory for project detection
            skillbook_path: Where to store the persistent skillbook
            project_root: Project root directory (auto-detected if not provided)
            ace_llm: Custom LLM client (optional, for testing)
        """
        self.cwd = cwd

        # Determine project root
        if project_root:
            self.project_root = project_root
        else:
            detected_root = find_project_root(Path(cwd))
            if detected_root:
                self.project_root = detected_root
            else:
                self.project_root = Path.home()
                logger.info(f"No project root found, using home: {self.project_root}")

        # Skillbook stored in .ace directory within project
        self.ace_dir = self.project_root / ".ace"
        self.skillbook_path = skillbook_path or (self.ace_dir / "skillbook.json")

        if self.skillbook_path.exists():
            self.skillbook = Skillbook.load_from_file(str(self.skillbook_path))
            logger.info(f"Loaded skillbook with {len(self.skillbook.skills())} skills")
        else:
            self.skillbook = Skillbook()
            logger.info("Created new skillbook")

        if ace_llm:
            self.ace_llm = ace_llm
        else:
            cli_path = os.environ.get("ACE_CLI_PATH")
            logger.info("Using Claude Code CLI for learning (subscription mode)")
            self.ace_llm = CLIClient(cli_path=cli_path)

        from .prompts import CLAUDE_CODE_REFLECTOR_PROMPT

        prompt_mgr = PromptManager()
        self.reflector = Reflector(
            self.ace_llm, prompt_template=CLAUDE_CODE_REFLECTOR_PROMPT
        )
        self.skill_manager = SkillManager(
            self.ace_llm, prompt_template=prompt_mgr.get_skill_manager_prompt()
        )

    def _persist_skillbook_update(self, update) -> bool:
        """Persist an UpdateBatch safely and update CLAUDE.md."""
        if self.skillbook_path.exists():
            skillbook = Skillbook.load_from_file(str(self.skillbook_path))
        else:
            skillbook = Skillbook()

        skillbook.apply_update(update)

        self.skillbook_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.skillbook_path, skillbook.dumps())

        # Update CLAUDE.md with TOON-compressed skillbook
        update_claude_md(self.project_root, skillbook)

        self.skillbook = skillbook
        return True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _run_reflector_with_retry(
        self, task: str, agent_output: AgentOutput, feedback: str
    ):
        return self.reflector.reflect(
            question=task,
            agent_output=agent_output,
            skillbook=self.skillbook,
            ground_truth=None,
            feedback=feedback,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _run_skill_manager_with_retry(self, reflection, cwd: str, progress: str):
        return self.skill_manager.update_skills(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context=f"Claude Code session in {cwd}",
            progress=progress,
        )

    def learn_from_transcript(
        self,
        transcript_path: Path,
        start_line: int = 0,
    ) -> bool:
        """
        Learn from a transcript file.

        Args:
            transcript_path: Path to Claude Code transcript JSONL
            start_line: Start learning from this line (for incremental learning)

        Returns:
            True if learning succeeded
        """
        try:
            total_lines = _count_transcript_lines(transcript_path)

            # Skip trivial sessions
            MIN_LINES = 5
            actual_lines = total_lines - start_line
            if actual_lines < MIN_LINES:
                logger.info(
                    f"Skipping trivial session ({actual_lines} lines, minimum {MIN_LINES})"
                )
                return True

            # Get TOON-compressed transcript
            toon_trace = toon_transcript(transcript_path, start_line)

            # Extract metadata
            task = _get_last_user_prompt(transcript_path)
            feedback = _get_transcript_feedback(transcript_path, start_line)
            cwd = _extract_cwd_from_transcript(transcript_path) or self.cwd

            # Create AgentOutput for Reflector
            agent_output = AgentOutput(
                reasoning=toon_trace,
                final_answer="(see trace)",
                skill_ids=[],
                raw={"total_lines": total_lines, "start_line": start_line},
            )

            # Run Reflector
            logger.info("Running Reflector...")
            reflection = self._run_reflector_with_retry(
                task=task,
                agent_output=agent_output,
                feedback=feedback,
            )

            # Run SkillManager
            logger.info("Running SkillManager...")
            skill_manager_output = self._run_skill_manager_with_retry(
                reflection=reflection,
                cwd=cwd,
                progress=f"lines {start_line + 1}-{total_lines}",
            )

            # Persist update
            if self._persist_skillbook_update(skill_manager_output.update):
                logger.info(f"Skillbook now has {len(self.skillbook.skills())} skills")

            return True

        except Exception as e:
            logger.error(f"Learning failed: {e}", exc_info=True)
            # Add helpful hints for common CLI failures
            if "CLI error" in str(e) or "Exit code" in str(e):
                logger.info(
                    "Hint: CLI failures often occur after using /resume "
                    "(transcript context may be incomplete). "
                    "Try again after more conversation, or use ace-learn-lines."
                )
            return False


# Keep old name for backwards compatibility
ACEHookLearner = ACELearner


# ============================================================================
# CLI Commands
# ============================================================================


def get_project_context(args) -> Path:
    """Get project root with priority: flag > env > auto-detect."""
    if hasattr(args, "project") and args.project:
        return Path(args.project).resolve()

    if env_dir := os.environ.get("ACE_PROJECT_DIR"):
        return Path(env_dir).resolve()

    root = find_project_root(Path.cwd())
    if root is None:
        raise NotInProjectError(str(Path.cwd()))
    return root


def cmd_learn(args):
    """Learn from the latest transcript.

    Always processes the full session. Most sessions are small (median 59 lines),
    and large sessions benefit from full context for better learnings.

    Use --lines N to optionally limit to the last N lines (for very large sessions).
    """
    # Get project context for filtering (if specified)
    filter_project = None
    if hasattr(args, "project") and args.project:
        filter_project = Path(args.project).resolve()
    elif env_dir := os.environ.get("ACE_PROJECT_DIR"):
        filter_project = Path(env_dir).resolve()

    # Find latest transcript (uses history.jsonl for correct /resume handling)
    transcript_path = find_latest_transcript(filter_project)
    if not transcript_path:
        print("No transcript found.")
        print("Use Claude Code first - transcripts are stored in ~/.claude/projects/")
        sys.exit(1)

    if not transcript_path.exists():
        print(f"Transcript not found: {transcript_path}")
        sys.exit(1)

    # Extract session info
    total_lines = _count_transcript_lines(transcript_path)
    cwd = _extract_cwd_from_transcript(transcript_path) or str(Path.cwd())

    # Determine project root
    try:
        if filter_project:
            project_root = filter_project
        else:
            project_root = find_project_root(Path(cwd))
            if not project_root:
                project_root = Path.home()
    except Exception as e:
        print(f"error: could not determine project: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine start line (simplified: always full session, optional --lines limit)
    lines_limit = getattr(args, "lines", None)
    if lines_limit and lines_limit < total_lines:
        start_line = total_lines - lines_limit
        print(f"Learning from transcript (last {lines_limit} lines): {transcript_path}")
    else:
        start_line = 0
        print(f"Learning from transcript ({total_lines} lines): {transcript_path}")

    print(f"Project: {project_root}")
    print()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Run learning
    try:
        learner = ACELearner(cwd=cwd, project_root=project_root)
        success = learner.learn_from_transcript(transcript_path, start_line=start_line)

        if success:
            print("\n✓ Learning complete!")
            print(f"Updated: {project_root / 'CLAUDE.md'}")
        else:
            print("\n✗ Learning failed")
            print(
                "\nHint: If you used /resume, the transcript context may be incomplete.",
                file=sys.stderr,
            )
            print(
                "Try again after more conversation, or use ace-learn-lines.",
                file=sys.stderr,
            )
            sys.exit(1)
    except Exception as e:
        logger.error(f"Learning failed: {e}", exc_info=True)
        print(f"\n✗ Learning failed: {e}", file=sys.stderr)
        if "CLI error" in str(e) or "Exit code" in str(e):
            print(
                "\nHint: If you used /resume, the transcript context may be incomplete.",
                file=sys.stderr,
            )
            print(
                "Try again after more conversation, or use ace-learn-lines.",
                file=sys.stderr,
            )
        sys.exit(1)


def cmd_insights(args):
    """Show current ACE learned strategies."""
    try:
        project_root = get_project_context(args)
        skillbook_path = project_root / ".ace" / "skillbook.json"
    except NotInProjectError as e:
        print(str(e), file=sys.stderr)
        return

    if not skillbook_path.exists():
        print("No insights yet. ACE will learn from your Claude Code sessions.")
        return

    try:
        skillbook = Skillbook.load_from_file(str(skillbook_path))
        skills = skillbook.skills()

        if not skills:
            print("No insights yet. ACE will learn from your Claude Code sessions.")
            return

        print(f"ACE Learned Strategies ({len(skills)} total)")
        print(f"Project: {project_root}\n")

        sections: dict = {}
        for skill in skills:
            section = skill.section
            if section not in sections:
                sections[section] = []
            sections[section].append(skill)

        for section, section_skills in sorted(sections.items()):
            print(f"## {section.replace('_', ' ').title()}")
            for s in section_skills:
                score = f"({s.helpful}↑ {s.harmful}↓)"
                print(f"  [{s.id}] {s.content} {score}")
            print()

    except Exception as e:
        print(f"Error reading skillbook: {e}")


def cmd_remove(args):
    """Remove a specific insight by ID."""
    try:
        project_root = get_project_context(args)
        ace_dir = project_root / ".ace"
        skillbook_path = ace_dir / "skillbook.json"
    except NotInProjectError as e:
        print(str(e), file=sys.stderr)
        return

    if not skillbook_path.exists():
        print(f"No skillbook found for project: {project_root}")
        return

    try:
        skillbook = Skillbook.load_from_file(str(skillbook_path))
        insight_id = args.id
        skills = skillbook.skills()
        target = None
        for s in skills:
            if (
                s.id == insight_id
                or insight_id in s.id
                or insight_id.lower() in s.content.lower()
            ):
                target = s
                break

        if not target:
            print(f"No insight found matching '{insight_id}'")
            print("Use 'ace-learn insights' to see available insights.")
            return

        skillbook = Skillbook.load_from_file(str(skillbook_path))
        skillbook.remove_skill(target.id)
        _atomic_write_text(skillbook_path, skillbook.dumps())

        # Update CLAUDE.md
        update_claude_md(project_root, skillbook)

        print(f"Removed: {target.content}")

    except Exception as e:
        print(f"Error removing insight: {e}")


def cmd_clear(args):
    """Clear all ACE learned strategies."""
    if not args.confirm:
        print("This will delete all learned strategies for this project.")
        print("Run with --confirm to proceed: ace-learn clear --confirm")
        return

    try:
        project_root = get_project_context(args)
        ace_dir = project_root / ".ace"
        skillbook_path = ace_dir / "skillbook.json"
    except NotInProjectError as e:
        print(str(e), file=sys.stderr)
        return

    try:
        skillbook = Skillbook()
        ace_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(skillbook_path, skillbook.dumps())

        # Update CLAUDE.md with empty skillbook
        update_claude_md(project_root, skillbook)

        print(f"All insights cleared for project: {project_root}")
        print("ACE will start fresh.")

    except Exception as e:
        print(f"Error clearing insights: {e}")


def cmd_setup(_args):
    """Install ACE slash commands into Claude Code."""
    commands_dir = Path(__file__).parent / "commands"
    target_dir = Path.home() / ".claude" / "commands"

    if not commands_dir.exists():
        print("Error: Command templates not found in package")
        return 1

    target_dir.mkdir(parents=True, exist_ok=True)

    installed = []
    for cmd_file in commands_dir.glob("*.md"):
        target_path = target_dir / cmd_file.name
        shutil.copy2(cmd_file, target_path)
        cmd_name = cmd_file.stem
        installed.append(cmd_name)

    if installed:
        print("Installed ACE slash commands:\n")
        for cmd in sorted(installed):
            print(f"  /{cmd}")
        print(f"\nCommands installed to: {target_dir}")
        print("\nYou can now use these commands in Claude Code.")
    else:
        print("No command files found to install.")
        return 1

    return 0


def _get_installed_commands() -> list:
    """Check which ACE slash commands are installed."""
    commands_dir = Path.home() / ".claude" / "commands"
    if not commands_dir.exists():
        return []

    ace_commands = [
        "ace-learn",
        "ace-learn-lines",
        "ace-doctor",
        "ace-insights",
        "ace-remove",
        "ace-clear",
    ]

    installed = []
    for cmd in ace_commands:
        if (commands_dir / f"{cmd}.md").exists():
            installed.append(cmd)

    return installed


def cmd_doctor(_args):
    """Verify ACE prerequisites and configuration."""
    import subprocess

    print("ACE Doctor - Checking prerequisites and configuration\n")
    all_ok = True

    # 1. Check Claude CLI
    print("1. Claude CLI...")
    claude_path = shutil.which("claude")
    if claude_path:
        print(f"   ✓ Found at: {claude_path}")
        try:
            result = subprocess.run(
                [claude_path, "--print", "-p", "ping"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                print("   ✓ CLI responds to ping")
            else:
                print(f"   ✗ CLI ping failed: {result.stderr[:100]}")
                all_ok = False
        except subprocess.TimeoutExpired:
            print("   ✗ CLI timed out")
            all_ok = False
        except Exception as e:
            print(f"   ✗ CLI test failed: {e}")
            all_ok = False
    else:
        print("   ✗ Claude CLI not found in PATH")
        print("     Install with: npm install -g @anthropic-ai/claude-code")
        all_ok = False

    # 1b. Check patched CLI
    print("\n   Patched CLI...")
    try:
        from .prompt_patcher import get_patch_info

        patch_info = get_patch_info()
        if patch_info["patched_cli_exists"]:
            status = "fresh" if patch_info["is_fresh"] else "stale"
            version = patch_info.get("patched_version") or "unknown"
            print(f"   ✓ Patched CLI available (v{version}, {status})")
            print(f"     {patch_info['patched_cli_path']}")
        else:
            print("   - Patched CLI not yet created")
            if patch_info["source_cli_path"]:
                print("     Will be created on first ace-learn run")
            else:
                print("     ✗ Cannot create - source cli.js not found")
    except Exception as e:
        print(f"   - Could not check patch status: {e}")

    # 2. Check transcript location
    print("\n2. Transcript location...")
    transcript = find_latest_transcript()
    if transcript:
        print(f"   ✓ Latest transcript: {transcript}")
        session_id = _extract_session_id(transcript)
        lines = _count_transcript_lines(transcript)
        print(f"   ✓ Session: {session_id} ({lines} lines)")
    else:
        print("   - No transcripts found yet (use Claude Code first)")

    # 3. Check output locations
    print("\n3. Output locations...")
    try:
        cwd = Path.cwd()
        project_root = find_project_root(cwd)
        if project_root:
            print(f"   Project root: {project_root}")
            claude_md = project_root / "CLAUDE.md"
            ace_dir = project_root / ".ace"
            skillbook_path = ace_dir / "skillbook.json"
            print(f"   CLAUDE.md: {claude_md}")
            if claude_md.exists():
                content = claude_md.read_text()
                if ACE_START_MARKER in content:
                    print("   ✓ CLAUDE.md has ACE section")
                else:
                    print("   - CLAUDE.md exists (ACE section will be added)")
            else:
                print("   - CLAUDE.md will be created")
            print(f"   Skillbook: {skillbook_path}")
            if skillbook_path.exists():
                print("   ✓ Existing skillbook found")
            else:
                print("   - No skillbook yet (will be created)")
        else:
            print(f"   No project root found from: {cwd}")
            print(f"   Will use home directory: {Path.home()}")
    except Exception as e:
        print(f"   Error detecting project: {e}")

    # 4. Check TOON availability
    print("\n4. TOON compression...")
    try:
        from toon import encode

        print("   ✓ python-toon installed")
    except ImportError:
        print("   ⚠ python-toon not installed (will use JSON fallback)")
        print("     Install with: pip install python-toon")

    # 5. Check slash commands
    print("\n5. Slash commands...")
    installed_cmds = _get_installed_commands()
    expected_cmds = [
        "ace-learn",
        "ace-learn-lines",
        "ace-doctor",
        "ace-insights",
        "ace-remove",
        "ace-clear",
    ]
    if len(installed_cmds) == len(expected_cmds):
        print(f"   ✓ All {len(installed_cmds)} slash commands installed")
    elif installed_cmds:
        missing = set(expected_cmds) - set(installed_cmds)
        print(f"   ⚠ {len(installed_cmds)}/{len(expected_cmds)} commands installed")
        print(f"     Missing: {', '.join(sorted(missing))}")
        print("     Run: ace-learn setup")
    else:
        print("   - No slash commands installed")
        print("     Run: ace-learn setup")

    # Summary
    print("\n" + "=" * 50)
    if all_ok:
        print("✓ All checks passed! ACE is ready to learn.")
        print("\nTo learn from your latest session, run: ace-learn")
    else:
        print("✗ Some checks failed. Please fix the issues above.")

    return 0 if all_ok else 1


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    """CLI entry point for ace-learn."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ACE learning for Claude Code - updates CLAUDE.md with learned strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ace-learn                  Learn from latest transcript, update CLAUDE.md
  ace-learn --lines 500      Learn from last 500 lines only
  ace-learn setup            Install /ace-* slash commands into Claude Code
  ace-learn doctor           Verify prerequisites
  ace-learn insights         Show learned strategies
  ace-learn remove <id>      Remove a specific insight
  ace-learn clear --confirm  Clear all insights

Learnings are stored in:
  - CLAUDE.md (TOON-compressed, for Claude Code to read)
  - .ace/skillbook.json (full skillbook for persistence)
""",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Setup command
    subparsers.add_parser("setup", help="Install ACE slash commands into Claude Code")

    # Doctor command
    subparsers.add_parser("doctor", help="Verify prerequisites and configuration")

    # Insight management commands
    insights_parser = subparsers.add_parser("insights", help="Show learned strategies")
    insights_parser.add_argument(
        "--project", "-P", help="Project root directory (default: auto-detect)"
    )

    remove_parser = subparsers.add_parser("remove", help="Remove a specific insight")
    remove_parser.add_argument("id", help="Insight ID or keyword to match")
    remove_parser.add_argument(
        "--project", "-P", help="Project root directory (default: auto-detect)"
    )

    clear_parser = subparsers.add_parser("clear", help="Clear all insights")
    clear_parser.add_argument(
        "--confirm", action="store_true", help="Confirm clearing all insights"
    )
    clear_parser.add_argument(
        "--project", "-P", help="Project root directory (default: auto-detect)"
    )

    # Main learning flags
    parser.add_argument(
        "--lines",
        "-l",
        type=int,
        default=None,
        help="Learn from last N lines only (default: full transcript)",
    )
    parser.add_argument(
        "--project", "-P", help="Project root directory (default: auto-detect)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.command == "setup":
        sys.exit(cmd_setup(args))
    elif args.command == "doctor":
        sys.exit(cmd_doctor(args))
    elif args.command == "insights":
        cmd_insights(args)
    elif args.command == "remove":
        cmd_remove(args)
    elif args.command == "clear":
        cmd_clear(args)
    else:
        # Default: run learning
        cmd_learn(args)


if __name__ == "__main__":
    main()
