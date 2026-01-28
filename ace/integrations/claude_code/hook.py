"""
Claude Code Hook Integration for ACE framework.

This module enables ACE learning from Claude Code sessions via hooks.
When configured as a Stop hook, it parses the session transcript and
updates a skill file that Claude automatically picks up.

Usage:
    1. Configure hook in ~/.claude/settings.json:
       {
         "hooks": {
           "Stop": [{
             "matcher": "*",
             "hooks": [{
               "type": "command",
               "command": "ace-learn"
             }]
           }]
         }
       }

    2. The hook receives transcript_path via stdin JSON
    3. ACE learns from the execution trace
    4. Updates ~/.claude/skills/ace-learnings/SKILL.md
"""

import json
import sys
import logging
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
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

from ...skillbook import Skillbook, Skill
from ...roles import Reflector, SkillManager, AgentOutput, ReflectorOutput
from ...prompts_v2_1 import PromptManager
from .cli_client import CLIClient

logger = logging.getLogger(__name__)


# ============================================================================
# Project Root Detection
# ============================================================================

# Markers to identify project root (checked in order)
# NOTE: .claude is NOT included because ~/.claude exists in home directory,
# which would incorrectly make $HOME the project root for non-project paths.
# Use .ace-root as explicit marker for monorepo root control.
DEFAULT_MARKERS = [
    ".ace-root",  # Explicit ACE project root marker (highest priority for monorepos)
    ".git",  # Version control
    ".hg",
    ".svn",
    "pyproject.toml",  # Python modern
    "package.json",  # Node.js
    "Cargo.toml",  # Rust
    "go.mod",  # Go
]


# Default timeout for async hook (seconds) - generous for LLM calls
ACE_HOOK_TIMEOUT = 300


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
    """
    Find project root by walking up from start directory.

    Priority order:
    1. ACE_PROJECT_DIR environment variable (if set and exists)
    2. Marker-based detection (markers are checked in priority order)

    The .ace-root marker has highest priority, allowing monorepo roots
    to be explicitly marked and take precedence over nested .git directories.

    Args:
        start: Directory to start searching from
        markers: List of file/directory names that indicate project root
                 (ordered by priority, highest first)

    Returns:
        Path to project root, or None if not found
    """
    # Check environment override first
    if env_dir := os.environ.get("ACE_PROJECT_DIR"):
        env_path = Path(env_dir).expanduser().resolve()
        if env_path.exists() and env_path.is_dir():
            logger.debug(f"Using ACE_PROJECT_DIR: {env_path}")
            return env_path
        else:
            logger.warning(f"ACE_PROJECT_DIR set but invalid: {env_dir}")

    markers = markers or DEFAULT_MARKERS
    start_resolved = start.resolve()

    # Check markers in priority order - higher priority markers
    # are checked across all parent directories first
    for marker in markers:
        current = start_resolved
        while True:
            if (current / marker).exists():
                return current
            if current.parent == current:  # Reached filesystem root
                break
            current = current.parent

    return None


# ============================================================================
# Transcript Parser
# ============================================================================


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

    def to_execution_trace(self) -> str:
        """Convert to execution trace format for Reflector."""
        parts = []
        step_num = 0

        for turn in self.turns:
            if turn.user_prompt:
                # Truncate very long prompts
                prompt_preview = turn.user_prompt[:200]
                if len(turn.user_prompt) > 200:
                    prompt_preview += "..."
                parts.append(f"[User] {prompt_preview}")

            if turn.assistant_text:
                parts.append(f"[Reasoning] {turn.assistant_text[:300]}")

            for tool in turn.tool_calls:
                step_num += 1
                status = "✗" if tool.is_error else "✓"

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

                # Include error info if failed
                if tool.is_error and tool.result:
                    error_preview = tool.result[:100]
                    parts.append(f"    Error: {error_preview}")

        return "\n".join(parts) if parts else "(No trace captured)"

    def get_feedback(self) -> str:
        """Generate feedback string for Reflector."""
        total = self.total_tool_calls
        failed = self.failed_tool_calls
        success_rate = ((total - failed) / total * 100) if total > 0 else 100

        feedback = (
            f"Session completed: {total} tool calls, {success_rate:.0f}% success rate"
        )
        if failed > 0:
            feedback += f" ({failed} failures)"
        return feedback


class TranscriptParser:
    """Parse Claude Code JSONL transcript files."""

    def parse(self, transcript_path: str) -> ParsedTranscript:
        """
        Parse a Claude Code transcript JSONL file.

        Args:
            transcript_path: Path to the .jsonl transcript file

        Returns:
            ParsedTranscript with structured conversation data
        """
        path = Path(transcript_path)
        if not path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")

        entries = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
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


# ============================================================================
# Skill File Generator
# ============================================================================


def get_project_skill_dir(cwd: str) -> Path:
    """
    Get the project-level skill directory for a given working directory.

    Finds the project root by walking up from cwd, then returns
    the skill directory path within that project.

    Args:
        cwd: Current working directory to start search from

    Returns:
        Path to skill directory: {project_root}/.claude/skills/ace-learnings/

    Raises:
        NotInProjectError: If no project root markers found
    """
    project_root = find_project_root(Path(cwd))
    if project_root is None:
        raise NotInProjectError(cwd)
    return project_root / ".claude" / "skills" / "ace-learnings"


class SkillGenerator:
    """Generate Claude Code skill files from ACE skillbook with progressive disclosure."""

    MIN_SKILLS_FOR_CATEGORY = 3  # Only split sections with 3+ skills

    def __init__(self, skill_dir: Path):
        """
        Initialize the skill generator.

        Args:
            skill_dir: Directory to write skill files to (required, use get_project_skill_dir())
        """
        self.skill_dir = skill_dir

    def _group_by_section(self, skills: List[Skill]) -> Dict[str, List[Skill]]:
        """Group skills by section, sorted by effectiveness within each."""
        sections: Dict[str, List[Skill]] = {}
        for skill in skills:
            sections.setdefault(skill.section, []).append(skill)
        # Sort each section by effectiveness
        for section in sections:
            sections[section] = sorted(
                sections[section],
                key=lambda b: (b.helpful - b.harmful, b.helpful),
                reverse=True,
            )
        return sections

    def _frontmatter(self, sections: List[str]) -> str:
        """Generate dynamic frontmatter based on actual sections.

        Note: location is auto-determined by Claude Code based on filesystem path,
        so we don't include it in frontmatter.
        """
        if sections:
            # Take up to 6 sections for the description
            keywords = ", ".join(s.replace("_", " ") for s in sorted(sections)[:6])
            desc = f"Project-specific coding patterns covering {keywords}. Use when writing code to follow established conventions."
        else:
            desc = "Project-specific patterns learned from coding sessions. Use when writing code to follow established conventions."
        return f"""---
name: ace-learnings
description: {desc}
---"""

    def _intro(self) -> str:
        return """# ACE Learned Strategies

These strategies have been automatically learned from coding sessions.
Apply relevant strategies based on the current task."""

    def _empty_skill(self) -> str:
        return f"""{self._frontmatter([])}

# ACE Learned Strategies

No strategies learned yet. Strategies will appear here as you use Claude Code.

{self._footer()}"""

    def _top_strategies(self, skills: List[Skill]) -> str:
        lines = ["## Top Strategies (by effectiveness)"]
        for i, s in enumerate(skills, 1):
            score = f"({s.helpful}↑ {s.harmful}↓)"
            lines.append(f"{i}. {s.content} {score}")
        return "\n".join(lines)

    def _section_inline(self, section: str, skills: List[Skill]) -> str:
        """Render a section inline (for small sections in main SKILL.md)."""
        title = section.replace("_", " ").title()
        lines = [f"## {title}"]
        for s in skills:
            score = f"({s.helpful}↑ {s.harmful}↓)"
            lines.append(f"- {s.content} {score}")
        return "\n".join(lines)

    def _category_index(self, large_sections: Dict[str, List[Skill]]) -> str:
        """Generate index of category files for progressive disclosure."""
        if not large_sections:
            return ""
        lines = ["## Categories"]
        lines.append("For detailed strategies, read the relevant category file:")
        for section in sorted(large_sections.keys()):
            skills = large_sections[section]
            title = section.replace("_", " ").title()
            filename = section.replace("_", "-") + ".md"
            lines.append(
                f"- **{title}**: `categories/{filename}` ({len(skills)} strategies)"
            )
        return "\n".join(lines)

    def _antipatterns(self, skills: List[Skill]) -> str:
        lines = ["## Antipatterns (avoid these)"]
        for s in skills:
            score = f"({s.harmful} failures)"
            lines.append(f"- ⚠️ {s.content} {score}")
        return "\n".join(lines)

    def _footer(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"""---
*Auto-generated by ACE at {timestamp}*"""

    def generate_main(
        self,
        sorted_skills: List[Skill],
        large_sections: Dict[str, List[Skill]],
        small_sections: Dict[str, List[Skill]],
    ) -> str:
        """Generate main SKILL.md with top strategies and category index."""
        all_sections = list(large_sections.keys()) + list(small_sections.keys())
        parts = [self._frontmatter(all_sections)]
        parts.append(self._intro())

        # Top 10 strategies (always shown)
        top_skills = sorted_skills[:10]
        if top_skills:
            parts.append(self._top_strategies(top_skills))

        # Category index for large sections (progressive disclosure)
        if large_sections:
            parts.append(self._category_index(large_sections))

        # Small sections inline (not worth splitting)
        for section in sorted(small_sections.keys()):
            parts.append(self._section_inline(section, small_sections[section]))

        # Antipatterns
        antipatterns = [s for s in sorted_skills if s.harmful > s.helpful]
        if antipatterns:
            parts.append(self._antipatterns(antipatterns[:5]))

        parts.append(self._footer())
        return "\n\n".join(parts)

    def generate_category(self, section: str, skills: List[Skill]) -> str:
        """Generate a category file for a specific section."""
        title = section.replace("_", " ").title()
        lines = [f"# {title} Strategies"]
        lines.append("")
        for s in skills:
            score = f"({s.helpful}↑ {s.harmful}↓)"
            lines.append(f"- {s.content} {score}")
        lines.append("")
        lines.append(self._footer())
        return "\n".join(lines)

    def save(self, skillbook: Skillbook) -> Path:
        """
        Save skill files with progressive disclosure.

        Creates:
        - SKILL.md: Top 10 + category index + small sections
        - categories/*.md: Detailed strategies for large sections

        Args:
            skillbook: ACE Skillbook to generate skill from

        Returns:
            Path to saved SKILL.md file
        """
        self.skill_dir.mkdir(parents=True, exist_ok=True)

        skills = skillbook.skills()
        if not skills:
            content = self._empty_skill()
            skill_path = self.skill_dir / "SKILL.md"
            skill_path.write_text(content, encoding="utf-8")
            return skill_path

        # Sort all skills by effectiveness
        sorted_skills = sorted(
            skills, key=lambda s: (s.helpful - s.harmful, s.helpful), reverse=True
        )

        # Group by section
        sections = self._group_by_section(skills)

        # Split into large (get own file) and small (stay inline)
        large_sections = {
            s: sk
            for s, sk in sections.items()
            if len(sk) >= self.MIN_SKILLS_FOR_CATEGORY
        }
        small_sections = {
            s: sk
            for s, sk in sections.items()
            if len(sk) < self.MIN_SKILLS_FOR_CATEGORY
        }

        # Generate and save main SKILL.md
        main_content = self.generate_main(sorted_skills, large_sections, small_sections)
        skill_path = self.skill_dir / "SKILL.md"
        skill_path.write_text(main_content, encoding="utf-8")
        logger.info(f"Saved skill file to {skill_path}")

        # Generate category files for large sections
        if large_sections:
            categories_dir = self.skill_dir / "categories"
            categories_dir.mkdir(exist_ok=True)
            for section, section_skills in large_sections.items():
                filename = section.replace("_", "-") + ".md"
                cat_content = self.generate_category(section, section_skills)
                cat_path = categories_dir / filename
                cat_path.write_text(cat_content, encoding="utf-8")
                logger.info(f"Saved category file to {cat_path}")

        return skill_path

    # Keep old generate() for backwards compatibility
    def generate(self, skillbook: Skillbook) -> str:
        """Generate SKILL.md content (legacy method, use save() for full feature)."""
        skills = skillbook.skills()
        if not skills:
            return self._empty_skill()

        sorted_skills = sorted(
            skills, key=lambda s: (s.helpful - s.harmful, s.helpful), reverse=True
        )
        sections = self._group_by_section(skills)

        # For legacy generate(), put everything inline
        return self.generate_main(sorted_skills, {}, sections)


# ============================================================================
# Skillbook Persistence (concurrent-safe)
# ============================================================================

_SKILLBOOK_EPOCH_FILENAME = ".ace-skillbook-epoch"
_SKILLBOOK_LOCK_FILENAME = ".ace-skillbook.lock"


def _skillbook_epoch_path(skill_dir: Path) -> Path:
    return skill_dir / _SKILLBOOK_EPOCH_FILENAME


def _ensure_skillbook_epoch(skill_dir: Path) -> str:
    """
    Get the current skillbook epoch.

    The epoch is a per-skill-dir marker used to prevent in-flight daemon tasks
    from writing stale skillbooks after `ace-learn clear`.
    """
    epoch_path = _skillbook_epoch_path(skill_dir)
    if epoch_path.exists():
        epoch = epoch_path.read_text(encoding="utf-8").strip()
        if epoch:
            return epoch
    # Default epoch when no clear has occurred yet.
    return "0"


def _bump_skillbook_epoch(skill_dir: Path) -> str:
    """Generate and persist a new epoch value."""
    epoch = uuid.uuid4().hex
    epoch_path = _skillbook_epoch_path(skill_dir)
    epoch_path.parent.mkdir(parents=True, exist_ok=True)
    epoch_path.write_text(epoch, encoding="utf-8")
    return epoch


@contextmanager
def _skillbook_lock(skill_dir: Path):
    """
    Inter-process lock for skillbook.json + SKILL.md writes.

    Best-effort cross-platform locking:
      - POSIX: fcntl.flock
      - Windows: falls back to no-op (still safe via epoch, but less robust)
    """
    lock_path = skill_dir / _SKILLBOOK_LOCK_FILENAME
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as f:
        if os.name == "posix":
            try:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
        yield


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


# ============================================================================
# Main Hook Learner
# ============================================================================


class ACEHookLearner:
    """
    Main class for learning from Claude Code sessions via hooks.

    Uses Claude CLI subscription (no API keys required).

    Usage:
        learner = ACEHookLearner(cwd="/path/to/project")
        learner.learn_from_transcript("/path/to/transcript.jsonl")
    """

    def __init__(
        self,
        cwd: str,
        skillbook_path: Optional[Path] = None,
        skill_dir: Optional[Path] = None,
        ace_model: str = "anthropic/claude-sonnet-4-5-20250929",
        ace_llm: Optional[Any] = None,
        use_cli: bool = True,  # Deprecated - always True
    ):
        """
        Initialize the hook learner.

        ACE now uses subscription-only mode via Claude CLI. No API keys required.

        Args:
            cwd: Working directory for skill storage
            skillbook_path: Where to store the persistent skillbook
            skill_dir: Where to write skill files (explicit path, bypasses auto-detection)
            ace_model: Deprecated - ignored (CLI only)
            ace_llm: Custom LLM client (optional, for testing)
            use_cli: Deprecated - always uses CLI

        Environment variables:
            ACE_CLI_PATH: Path to custom claude CLI or cli.js file
            ACE_PROJECT_DIR: Override project root detection
        """
        self.cwd = cwd

        # Determine skill directory
        if skill_dir:
            # Explicit skill_dir passed - use it directly
            self.skill_dir = skill_dir
        else:
            # Try project detection, fall back to global
            try:
                self.skill_dir = get_project_skill_dir(cwd)
            except NotInProjectError:
                self.skill_dir = Path.home() / ".claude" / "skills" / "ace-learnings-global"
                logger.info(f"No project root found, using global: {self.skill_dir}")

        self.skill_generator = SkillGenerator(self.skill_dir)
        self.skillbook_path = skillbook_path or (self.skill_dir / "skillbook.json")
        self.transcript_parser = TranscriptParser()
        self._skillbook_epoch = _ensure_skillbook_epoch(self.skill_dir)

        # Load or create skillbook
        if self.skillbook_path.exists():
            self.skillbook = Skillbook.load_from_file(str(self.skillbook_path))
            logger.info(f"Loaded skillbook with {len(self.skillbook.skills())} skills")
        else:
            self.skillbook = Skillbook()
            logger.info("Created new skillbook")

        # Create ACE components - subscription-only via CLI
        if ace_llm:
            self.ace_llm = ace_llm
        else:
            # Use CLI with optional custom path from environment
            cli_path = os.environ.get("ACE_CLI_PATH")
            logger.info("Using Claude Code CLI for learning (subscription mode)")
            self.ace_llm = CLIClient(cli_path=cli_path)

        prompt_mgr = PromptManager()
        self.reflector = Reflector(
            self.ace_llm, prompt_template=prompt_mgr.get_reflector_prompt()
        )
        self.skill_manager = SkillManager(
            self.ace_llm, prompt_template=prompt_mgr.get_skill_manager_prompt()
        )

    def _persist_skillbook_update(self, update) -> bool:
        """
        Persist an UpdateBatch safely.

        Prevents two classes of bugs:
          1) In-flight daemon jobs resurrecting skills after `ace-learn clear`
          2) Concurrent daemon jobs overwriting each other's updates
        """
        with _skillbook_lock(self.skill_dir):
            current_epoch = _ensure_skillbook_epoch(self.skill_dir)
            if current_epoch != self._skillbook_epoch:
                logger.warning(
                    "Skillbook was cleared during this learning run; skipping save"
                )
                return False

            if self.skillbook_path.exists():
                skillbook = Skillbook.load_from_file(str(self.skillbook_path))
            else:
                skillbook = Skillbook()

            skillbook.apply_update(update)

            self.skillbook_path.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(self.skillbook_path, skillbook.dumps())
            self.skill_generator.save(skillbook)

            # Keep in-memory copy consistent for logging/inspection.
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
        """Run Reflector with retry on transient failures."""
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
        """Run SkillManager with retry on transient failures."""
        return self.skill_manager.update_skills(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context=f"Claude Code session in {cwd}",
            progress=progress,
        )

    @classmethod
    def learn_from_hook_input(
        cls,
        hook_input: Dict[str, Any],
        ace_model: str = "anthropic/claude-sonnet-4-5-20250929",
        use_cli: bool = True,  # Default to CLI subscription
    ) -> bool:
        """
        Process hook input and learn from the session.

        This method now uses transcript-first root detection:
        1. Validates transcript_path exists
        2. Parses transcript to extract effective_cwd
        3. Resolves project_root from effective_cwd
        4. Falls back to global skill dir if no project root found

        Args:
            hook_input: Parsed hook input containing transcript_path and cwd
            ace_model: Model for ACE Reflector/SkillManager (ignored - CLI only)
            use_cli: Deprecated - always uses CLI subscription

        Returns:
            True if learning succeeded, False otherwise
        """
        transcript_path = hook_input.get("transcript_path")
        hook_cwd = hook_input.get("cwd")

        if not transcript_path:
            logger.error("No transcript_path in hook input")
            return False

        # Validate transcript exists before proceeding
        if not Path(transcript_path).exists():
            logger.error(f"Transcript file not found: {transcript_path}")
            return False

        # Parse transcript to get effective cwd (transcript cwd is more accurate)
        parser = TranscriptParser()
        try:
            transcript = parser.parse(transcript_path)
        except Exception as e:
            logger.error(f"Failed to parse transcript: {e}")
            return False

        # Use transcript cwd if available, fall back to hook cwd
        effective_cwd = transcript.cwd or hook_cwd
        if not effective_cwd:
            logger.error("No cwd in transcript or hook input")
            return False

        logger.info(f"Effective cwd: {effective_cwd}")

        # Resolve project root and skill directory
        project_root = find_project_root(Path(effective_cwd))
        if project_root:
            skill_dir = project_root / ".claude" / "skills" / "ace-learnings"
            logger.info(f"Using project skill dir: {skill_dir}")
        else:
            # Global fallback for non-project directories
            skill_dir = Path.home() / ".claude" / "skills" / "ace-learnings-global"
            logger.info(f"No project root found, using global skill dir: {skill_dir}")

        # Create learner with explicit paths
        learner = cls(
            cwd=effective_cwd,
            skill_dir=skill_dir,
            ace_model=ace_model,
            use_cli=True,  # Always use CLI
        )
        # Pass already-parsed transcript to avoid double parsing
        return learner.learn_from_transcript(transcript_path, parsed_transcript=transcript)

    def learn_from_transcript(
        self,
        transcript_path: str,
        parsed_transcript: Optional[ParsedTranscript] = None,
    ) -> bool:
        """
        Learn from a transcript file directly.

        Args:
            transcript_path: Path to Claude Code transcript JSONL
            parsed_transcript: Optional pre-parsed transcript to avoid double parsing

        Returns:
            True if learning succeeded
        """
        try:
            # Use pre-parsed transcript if provided, otherwise parse
            if parsed_transcript is not None:
                transcript = parsed_transcript
                logger.info(f"Using pre-parsed transcript: {transcript.total_tool_calls} tool calls")
            else:
                transcript = self.transcript_parser.parse(transcript_path)
                logger.info(f"Parsed transcript: {transcript.total_tool_calls} tool calls")

            # Skip trivial sessions (less than 3 tool calls)
            MIN_TOOL_CALLS = 3
            if transcript.total_tool_calls < MIN_TOOL_CALLS:
                logger.info(
                    f"Skipping trivial session ({transcript.total_tool_calls} tool calls, "
                    f"minimum {MIN_TOOL_CALLS})"
                )
                return True

            # Get last user prompt as the "task"
            task = "Claude Code session"
            for turn in reversed(transcript.turns):
                if turn.user_prompt:
                    task = turn.user_prompt[:200]
                    break

            # Create AgentOutput for Reflector
            agent_output = AgentOutput(
                reasoning=transcript.to_execution_trace(),
                final_answer=(
                    transcript.turns[-1].assistant_text if transcript.turns else ""
                ),
                skill_ids=[],
                raw={
                    "total_tools": transcript.total_tool_calls,
                    "failed_tools": transcript.failed_tool_calls,
                },
            )

            # Run Reflector with retry
            logger.info("Running Reflector...")
            reflection = self._run_reflector_with_retry(
                task=task,
                agent_output=agent_output,
                feedback=transcript.get_feedback(),
            )

            # Run SkillManager with retry
            logger.info("Running SkillManager...")
            skill_manager_output = self._run_skill_manager_with_retry(
                reflection=reflection,
                cwd=transcript.cwd,
                progress=f"{transcript.successful_tool_calls}/{transcript.total_tool_calls} successful",
            )

            # Persist update (epoch-aware + locked to avoid clobbering)
            if self._persist_skillbook_update(skill_manager_output.update):
                logger.info(f"Skillbook now has {len(self.skillbook.skills())} skills")

            return True

        except Exception as e:
            logger.error(f"Learning failed: {e}", exc_info=True)
            return False


# ============================================================================
# CLI Entry Point
# ============================================================================


def setup_hook():
    """Configure Claude Code to use ACE learning hook with async execution.

    Uses Claude Code's native async hook support for background learning.
    """
    settings_path = Path.home() / ".claude" / "settings.json"

    # Check that claude CLI is available
    claude_path = shutil.which("claude")
    if not claude_path:
        print("Warning: Claude CLI not found in PATH")
        print("Install with: npm install -g @anthropic-ai/claude-code")
        print()
        print("Continuing with setup anyway...")
        print()

    # Check ace-learn is available
    ace_learn_path = shutil.which("ace-learn")
    if not ace_learn_path:
        print("Warning: ace-learn not found in PATH")
        print("Install with: pip install ace-framework")
        print()

    # Load existing settings or create new
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            settings = {}
    else:
        settings = {}

    # Add/update hook config using native async hooks
    if "hooks" not in settings:
        settings["hooks"] = {}

    settings["hooks"]["Stop"] = [
        {
            "matcher": "*",
            "hooks": [
                {
                    "type": "command",
                    "command": "ace-learn",
                    "async": True,
                    "timeout": ACE_HOOK_TIMEOUT,
                }
            ],
        }
    ]

    # Write back
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2))

    # Create slash commands for enable/disable
    _create_slash_commands()

    # Patch Claude CLI for minimal system prompt
    from ace.integrations.claude_code.prompt_patcher import patch_cli

    print("\nPatching Claude CLI for minimal token overhead...")
    patched_path = patch_cli()
    if patched_path:
        print(f"✓ Patched CLI created: {patched_path}")
        print("  Token savings: ~2,800 tokens per learning call")
    else:
        print("⚠ Could not patch CLI (will use standard claude with full system prompt)")

    print("\n✓ Claude Code async hook configured!")
    print()
    print("ACE will learn from your sessions in the background.")
    print()
    print("Data locations (per-project):")
    print("  Skill file:  <project>/.claude/skills/ace-learnings/SKILL.md")
    print("  Skillbook:   <project>/.claude/skills/ace-learnings/skillbook.json")
    print()
    print("Note: Skills are stored per-project. Run from within a project directory.")
    print("      To control project root in monorepos, create a .ace-root file.")
    print(f"Settings saved to: {settings_path}")


def enable_hook():
    """Enable ACE learning hook in Claude Code settings.

    Uses Claude Code's native async hook support for background learning.
    """
    settings_path = Path.home() / ".claude" / "settings.json"

    # Load existing settings
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            settings = {}
    else:
        settings = {}

    # Add hook config using native async hooks
    if "hooks" not in settings:
        settings["hooks"] = {}

    settings["hooks"]["Stop"] = [
        {
            "matcher": "*",
            "hooks": [
                {
                    "type": "command",
                    "command": "ace-learn",
                    "async": True,
                    "timeout": ACE_HOOK_TIMEOUT,
                }
            ],
        }
    ]

    # Write back
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2))

    print("ACE learning enabled with async hook")


def disable_hook():
    """Disable ACE learning hook in Claude Code settings."""
    settings_path = Path.home() / ".claude" / "settings.json"

    if not settings_path.exists():
        print("ACE learning was not configured")
        return

    try:
        settings = json.loads(settings_path.read_text())
    except json.JSONDecodeError:
        print("ACE learning was not configured")
        return

    # Remove the Stop hook
    if "hooks" in settings and "Stop" in settings["hooks"]:
        del settings["hooks"]["Stop"]
        # Clean up empty hooks dict
        if not settings["hooks"]:
            del settings["hooks"]

    settings_path.write_text(json.dumps(settings, indent=2))
    print("ACE learning disabled")


def get_project_context(args) -> Path:
    """
    Get project root with priority: flag > env > auto-detect.

    Args:
        args: Parsed argparse arguments (may have .project attribute)

    Returns:
        Path to project root

    Raises:
        NotInProjectError: If no project root can be found
    """
    # 1. Explicit --project flag
    if hasattr(args, "project") and args.project:
        return Path(args.project).resolve()

    # 2. Environment variable (for CI/automation)
    if env_dir := os.environ.get("ACE_PROJECT_DIR"):
        return Path(env_dir).resolve()

    # 3. Auto-detect from shell cwd
    root = find_project_root(Path.cwd())
    if root is None:
        raise NotInProjectError(str(Path.cwd()))
    return root


def show_insights(args):
    """Show current ACE learned strategies."""
    try:
        project_root = get_project_context(args)
        skill_dir = get_project_skill_dir(str(project_root))
        skillbook_path = skill_dir / "skillbook.json"
    except NotInProjectError as e:
        print(str(e), file=sys.stderr)
        return

    if not skillbook_path.exists():
        print("No insights yet. ACE will learn from your Claude Code sessions.")
        return

    try:
        from ...skillbook import Skillbook

        skillbook = Skillbook.load_from_file(str(skillbook_path))
        skills = skillbook.skills()

        if not skills:
            print("No insights yet. ACE will learn from your Claude Code sessions.")
            return

        print(f"ACE Learned Strategies ({len(skills)} total)")
        print(f"Project: {project_root}\n")

        # Group by section
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


def remove_insight(args):
    """Remove a specific insight by ID."""
    try:
        project_root = get_project_context(args)
        skill_dir = get_project_skill_dir(str(project_root))
        skillbook_path = skill_dir / "skillbook.json"
    except NotInProjectError as e:
        print(str(e), file=sys.stderr)
        return

    if not skillbook_path.exists():
        print(f"No skillbook found for project: {project_root}")
        return

    try:
        from ...skillbook import Skillbook

        # First pass: find the target skill (read-only, no lock needed)
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

        # Remove under lock to prevent concurrent write corruption
        with _skillbook_lock(skill_dir):
            # Reload under lock to get latest state
            skillbook = Skillbook.load_from_file(str(skillbook_path))
            skillbook.remove_skill(target.id)
            _atomic_write_text(skillbook_path, skillbook.dumps())

            # Regenerate skill file
            generator = SkillGenerator(skill_dir)
            generator.save(skillbook)

        print(f"Removed: {target.content}")

    except Exception as e:
        print(f"Error removing insight: {e}")


def clear_insights(args):
    """Clear all ACE learned strategies."""
    if not args.confirm:
        print("This will delete all learned strategies for this project.")
        print("Run with --confirm to proceed: ace-learn clear --confirm")
        return

    try:
        project_root = get_project_context(args)
        skill_dir = get_project_skill_dir(str(project_root))
        skillbook_path = skill_dir / "skillbook.json"
        skill_path = skill_dir / "SKILL.md"
    except NotInProjectError as e:
        print(str(e), file=sys.stderr)
        return

    try:
        import shutil
        import glob
        from ...skillbook import Skillbook

        # Clear skillbook under lock and bump epoch so in-flight async jobs can't
        # resurrect cleared skills by writing stale state back to disk.
        with _skillbook_lock(skill_dir):
            _bump_skillbook_epoch(skill_dir)
            skillbook = Skillbook()
            _atomic_write_text(skillbook_path, skillbook.dumps())

            # Remove categories directory if it exists
            categories_dir = skill_dir / "categories"
            if categories_dir.exists():
                shutil.rmtree(categories_dir)

            # Regenerate empty skill file
            generator = SkillGenerator(skill_dir)
            generator.save(skillbook)

        print(f"All insights cleared for project: {project_root}")
        print("ACE will start fresh.")

    except Exception as e:
        print(f"Error clearing insights: {e}")


def _create_slash_commands():
    """Create slash commands for enabling/disabling ACE learning."""
    commands_dir = Path.home() / ".claude" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)

    # /ace-on command
    ace_on_content = """Enable ACE learning for Claude Code sessions.

Run this command to enable ACE learning:
```bash
ace-learn enable
```

After enabling, ACE will learn from your coding sessions and build a skillbook of strategies.
"""
    (commands_dir / "ace-on.md").write_text(ace_on_content)

    # /ace-off command
    ace_off_content = """Disable ACE learning for Claude Code sessions.

Run this command to disable ACE learning:
```bash
ace-learn disable
```

This stops ACE from learning from your sessions. Your existing skillbook is preserved.
"""
    (commands_dir / "ace-off.md").write_text(ace_off_content)

    # /ace-insights command
    ace_insights_content = """Show ACE learned strategies.

Run this command to see all learned insights:
```bash
ace-learn insights
```

Display the output to show the user their current skillbook of strategies.
"""
    (commands_dir / "ace-insights.md").write_text(ace_insights_content)

    # /ace-remove command
    ace_remove_content = """Remove an ACE learned strategy.

First, show current insights:
```bash
ace-learn insights
```

Then ask the user which insight to remove (by ID or keyword).

Remove it with:
```bash
ace-learn remove "<id-or-keyword>"
```

Confirm the removal to the user.
"""
    (commands_dir / "ace-remove.md").write_text(ace_remove_content)

    # /ace-clear command
    ace_clear_content = """Clear all ACE learned strategies.

IMPORTANT: Ask the user to confirm they want to delete all insights before proceeding.

If confirmed, run:
```bash
ace-learn clear --confirm
```

This will reset the skillbook and start fresh.
"""
    (commands_dir / "ace-clear.md").write_text(ace_clear_content)

    # /skill-retro command (from skill_improvement module)
    try:
        from ..claude_code_skill_improvement import create_slash_commands as create_skill_retro_command
        create_skill_retro_command()
    except ImportError:
        pass  # Module not available, skip


def doctor_check(args):
    """Verify ACE prerequisites and configuration.

    Checks:
    1. Claude CLI is available and works
    2. Node.js is available (if using patched CLI)
    3. Hook configuration
    4. Skill output location
    """
    import subprocess

    print("ACE Doctor - Checking prerequisites and configuration\n")
    all_ok = True

    # 1. Check Claude CLI
    print("1. Claude CLI...")
    claude_path = shutil.which("claude")
    if claude_path:
        print(f"   ✓ Found at: {claude_path}")
        # Test that it works
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

    # 2. Check for patched CLI (CLI Resolution)
    print("\n2. CLI Resolution...")
    patched_cli = Path.home() / ".ace" / "claude-learner" / "cli.js"
    if patched_cli.exists():
        print(f"   ✓ Using patched CLI: {patched_cli}")
        print("   ✓ Token savings: ~2,800 tokens per learning call")
        # Check node is available
        node_path = shutil.which("node")
        if node_path:
            print(f"   ✓ Node.js found at: {node_path}")
        else:
            print("   ✗ Node.js not found (required for patched CLI)")
            all_ok = False
    else:
        print("   ⚠ Using standard CLI (full system prompt)")
        print("   - Run 'ace-learn patch' to reduce token overhead")

    # 3. Check hook configuration
    print("\n3. Hook configuration...")
    settings_path = Path.home() / ".claude" / "settings.json"
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
            stop_hooks = settings.get("hooks", {}).get("Stop", [])
            if stop_hooks:
                for hook_config in stop_hooks:
                    for hook in hook_config.get("hooks", []):
                        cmd = hook.get("command", "")
                        if "ace" in cmd.lower():
                            is_async = hook.get("async", False)
                            timeout = hook.get("timeout", "default")
                            print(f"   ✓ ACE hook configured: {cmd}")
                            if is_async:
                                print(f"   ✓ Async mode enabled (timeout: {timeout}s)")
                            else:
                                print("   ⚠ Not using async mode - consider 'ace-learn setup'")
                            break
            else:
                print("   ✗ No Stop hook configured")
                print("     Run: ace-learn setup")
                all_ok = False
        except Exception as e:
            print(f"   ✗ Error reading settings: {e}")
            all_ok = False
    else:
        print("   ✗ No settings.json found")
        print("     Run: ace-learn setup")
        all_ok = False

    # 4. Check skill output location
    print("\n4. Skill output location...")
    try:
        cwd = Path.cwd()
        project_root = find_project_root(cwd)
        if project_root:
            skill_dir = project_root / ".claude" / "skills" / "ace-learnings"
            print(f"   Project root: {project_root}")
            print(f"   Skills will go to: {skill_dir}")
            if skill_dir.exists():
                skill_count = len(list(skill_dir.glob("*.md")))
                skillbook = skill_dir / "skillbook.json"
                if skillbook.exists():
                    print(f"   ✓ Existing skillbook found")
                else:
                    print(f"   - No skillbook yet (will be created)")
        else:
            global_dir = Path.home() / ".claude" / "skills" / "ace-learnings-global"
            print(f"   No project root found from: {cwd}")
            print(f"   Skills will go to (global): {global_dir}")
    except Exception as e:
        print(f"   Error detecting project: {e}")

    # Summary
    print("\n" + "=" * 50)
    if all_ok:
        print("✓ All checks passed! ACE is ready to learn.")
    else:
        print("✗ Some checks failed. Please fix the issues above.")

    return 0 if all_ok else 1


def cmd_patch(args):
    """Force create/recreate patched Claude CLI."""
    from ace.integrations.claude_code.prompt_patcher import patch_cli, find_claude_cli_js

    source = find_claude_cli_js()
    if not source:
        print("ERROR: Could not find Claude Code installation")
        return 1

    print(f"Source: {source}")
    patched = patch_cli(force=True)
    if patched:
        print(f"✓ Patched CLI created: {patched}")
        return 0
    else:
        print("ERROR: Patching failed")
        return 1


def cmd_unpatch(args):
    """Remove patched CLI."""
    patched = Path.home() / ".ace" / "claude-learner" / "cli.js"
    if patched.exists():
        patched.unlink()
        print(f"✓ Removed: {patched}")
        print("  ACE will now use standard claude CLI")
    else:
        print("No patched CLI found")
    return 0


def run_learning(args):
    """Run the learning process (called from async hook or manually).

    Parses stdin JSON from Claude Code hook and learns from the session.
    """
    # STEP 1: Parse stdin
    hook_input = None
    cwd = None
    transcript_path = None

    if args.transcript:
        # Manual mode with explicit transcript file
        cwd = getattr(args, "project", None) or os.getcwd()
        transcript_path = args.transcript
    else:
        # Hook mode: read stdin first (critical - must happen before spawn)
        if sys.stdin.isatty():
            print("error: no stdin input (expected hook JSON)", file=sys.stderr)
            print("hint: use -t <transcript> for manual learning", file=sys.stderr)
            sys.exit(1)
        try:
            hook_input = json.load(sys.stdin)
        except json.JSONDecodeError as e:
            print(f"error: invalid JSON from stdin: {e}", file=sys.stderr)
            sys.exit(1)

        cwd = hook_input.get("cwd")
        transcript_path = hook_input.get("transcript_path")

        if not cwd:
            print("error: missing 'cwd' in hook input", file=sys.stderr)
            sys.exit(1)
        if not transcript_path:
            print("error: missing 'transcript_path' in hook input", file=sys.stderr)
            sys.exit(1)

    # STEP 2: Run learning
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        if hook_input:
            success = ACEHookLearner.learn_from_hook_input(
                hook_input, ace_model=args.model
            )
        else:
            learner = ACEHookLearner(cwd=cwd, ace_model=args.model)
            success = learner.learn_from_transcript(transcript_path)
    except NotInProjectError as e:
        logger.error(str(e))
        print(str(e), file=sys.stderr)
        sys.exit(1)

    sys.exit(0 if success else 1)


def main():
    """CLI entry point for ace-learn."""
    import argparse

    parser = argparse.ArgumentParser(
        description="ACE learning for Claude Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ace-learn setup              Configure Claude Code hook (run once)
  ace-learn doctor             Verify prerequisites and configuration
  ace-learn enable             Enable ACE learning
  ace-learn disable            Disable ACE learning
  ace-learn insights           Show learned strategies
  ace-learn remove <id>        Remove a specific insight
  ace-learn clear --confirm    Clear all insights
  ace-learn                    Learn from stdin (called by hook)
  ace-learn -t transcript.jsonl   Learn from specific transcript
  ace-learn -P /path/to/project   Override project root detection

Skills are stored per-project at: <project>/.claude/skills/ace-learnings/
Global fallback: ~/.claude/skills/ace-learnings-global/
""",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Setup command
    subparsers.add_parser("setup", help="Configure Claude Code to use ACE learning")

    # Enable/disable commands
    subparsers.add_parser("enable", help="Enable ACE learning hook")
    subparsers.add_parser("disable", help="Disable ACE learning hook")

    # Doctor command
    subparsers.add_parser("doctor", help="Verify prerequisites and configuration")

    # Patch/unpatch commands
    subparsers.add_parser("patch", help="Create/recreate patched Claude CLI")
    subparsers.add_parser("unpatch", help="Remove patched CLI, use standard claude")

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

    # Learning options (work without subcommand for backwards compat)
    parser.add_argument(
        "--transcript", "-t", help="Path to transcript file (if not using stdin)"
    )
    parser.add_argument(
        "--project", "-P", help="Project root directory (default: auto-detect from cwd)"
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Model for ACE learning",
        default="anthropic/claude-sonnet-4-5-20250929",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.command == "setup":
        setup_hook()
    elif args.command == "enable":
        enable_hook()
    elif args.command == "disable":
        disable_hook()
    elif args.command == "doctor":
        sys.exit(doctor_check(args))
    elif args.command == "patch":
        sys.exit(cmd_patch(args))
    elif args.command == "unpatch":
        sys.exit(cmd_unpatch(args))
    elif args.command == "insights":
        show_insights(args)
    elif args.command == "remove":
        remove_insight(args)
    elif args.command == "clear":
        clear_insights(args)
    else:
        # Default: run learning (backwards compat with hook calling ace-learn)
        run_learning(args)


if __name__ == "__main__":
    main()
