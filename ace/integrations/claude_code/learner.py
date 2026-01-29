"""
Claude Code ACE Learning - Simple transcript-based learning.

This module enables ACE learning from Claude Code sessions by reading
existing transcript files directly. No hooks or complex infrastructure.

Usage:
    1. Use Claude Code normally
    2. Run /ace-learn (or `ace-learn`) to learn from the session
    3. Skills are updated in .claude/skills/ace-learnings/SKILL.md

Commands:
    ace-learn              # Learn from full latest transcript
    ace-learn --lines 500  # Learn from last 500 lines only (optional)
    ace-learn insights     # Show learned strategies
    ace-learn remove <id>  # Remove specific insight
    ace-learn clear --confirm  # Clear all insights
    ace-learn doctor       # Check prerequisites
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
from typing import List, Optional, Dict, Any
from datetime import datetime

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

from ...skillbook import Skillbook, Skill
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


def find_latest_transcript() -> Optional[Path]:
    """
    Find the latest Claude Code transcript file.

    Searches ~/.claude/projects/**/*.jsonl for the most recently modified
    transcript file.

    Returns:
        Path to the latest transcript, or None if not found
    """
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


def toon_transcript(transcript_path: Path, start_line: int = 0) -> str:
    """
    Read transcript .jsonl and convert to TOON format for Reflector.

    Args:
        transcript_path: Path to the .jsonl transcript
        start_line: Start reading from this line (0-indexed)

    Returns:
        TOON-encoded transcript entries
    """
    try:
        from toon import encode
    except ImportError:
        # Fallback to compact JSON if TOON not available
        logger.warning("TOON not installed, using compact JSON")
        entries = []
        with transcript_path.open() as f:
            for i, line in enumerate(f):
                if i >= start_line and line.strip():
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return json.dumps(entries, separators=(",", ":"))

    entries = []
    with transcript_path.open() as f:
        for i, line in enumerate(f):
            if i >= start_line and line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return encode(entries, {"delimiter": "\t"})


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
                if entry.get("type") == "user":
                    message = entry.get("message", {})
                    for block in message.get("content", []):
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
                    for block in message.get("content", []):
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
# Skill File Generator
# ============================================================================


def get_project_skill_dir(cwd: str) -> Path:
    """Get the project-level skill directory for a given working directory."""
    project_root = find_project_root(Path(cwd))
    if project_root is None:
        raise NotInProjectError(cwd)
    return project_root / ".claude" / "skills" / "ace-learnings"


class SkillGenerator:
    """Generate Claude Code skill files from ACE skillbook with progressive disclosure."""

    MIN_SKILLS_FOR_CATEGORY = 3

    def __init__(self, skill_dir: Path):
        self.skill_dir = skill_dir

    def _group_by_section(self, skills: List[Skill]) -> Dict[str, List[Skill]]:
        sections: Dict[str, List[Skill]] = {}
        for skill in skills:
            sections.setdefault(skill.section, []).append(skill)
        for section in sections:
            sections[section] = sorted(
                sections[section],
                key=lambda b: (b.helpful - b.harmful, b.helpful),
                reverse=True,
            )
        return sections

    def _frontmatter(self, sections: List[str]) -> str:
        if sections:
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
        title = section.replace("_", " ").title()
        lines = [f"## {title}"]
        for s in skills:
            score = f"({s.helpful}↑ {s.harmful}↓)"
            lines.append(f"- {s.content} {score}")
        return "\n".join(lines)

    def _category_index(self, large_sections: Dict[str, List[Skill]]) -> str:
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
        all_sections = list(large_sections.keys()) + list(small_sections.keys())
        parts = [self._frontmatter(all_sections)]
        parts.append(self._intro())

        top_skills = sorted_skills[:10]
        if top_skills:
            parts.append(self._top_strategies(top_skills))

        if large_sections:
            parts.append(self._category_index(large_sections))

        for section in sorted(small_sections.keys()):
            parts.append(self._section_inline(section, small_sections[section]))

        antipatterns = [s for s in sorted_skills if s.harmful > s.helpful]
        if antipatterns:
            parts.append(self._antipatterns(antipatterns[:5]))

        parts.append(self._footer())
        return "\n\n".join(parts)

    def generate_category(self, section: str, skills: List[Skill]) -> str:
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
        self.skill_dir.mkdir(parents=True, exist_ok=True)

        skills = skillbook.skills()
        if not skills:
            content = self._empty_skill()
            skill_path = self.skill_dir / "SKILL.md"
            skill_path.write_text(content, encoding="utf-8")
            return skill_path

        sorted_skills = sorted(
            skills, key=lambda s: (s.helpful - s.harmful, s.helpful), reverse=True
        )

        sections = self._group_by_section(skills)
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

        main_content = self.generate_main(sorted_skills, large_sections, small_sections)
        skill_path = self.skill_dir / "SKILL.md"
        skill_path.write_text(main_content, encoding="utf-8")
        logger.info(f"Saved skill file to {skill_path}")

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

    def generate(self, skillbook: Skillbook) -> str:
        """Legacy method for backwards compatibility."""
        skills = skillbook.skills()
        if not skills:
            return self._empty_skill()

        sorted_skills = sorted(
            skills, key=lambda s: (s.helpful - s.harmful, s.helpful), reverse=True
        )
        sections = self._group_by_section(skills)
        return self.generate_main(sorted_skills, {}, sections)


# ============================================================================
# Skillbook Persistence (concurrent-safe)
# ============================================================================

_SKILLBOOK_EPOCH_FILENAME = ".ace-skillbook-epoch"
_SKILLBOOK_LOCK_FILENAME = ".ace-skillbook.lock"


def _skillbook_epoch_path(skill_dir: Path) -> Path:
    return skill_dir / _SKILLBOOK_EPOCH_FILENAME


def _ensure_skillbook_epoch(skill_dir: Path) -> str:
    epoch_path = _skillbook_epoch_path(skill_dir)
    if epoch_path.exists():
        epoch = epoch_path.read_text(encoding="utf-8").strip()
        if epoch:
            return epoch
    return "0"


def _bump_skillbook_epoch(skill_dir: Path) -> str:
    epoch = uuid.uuid4().hex
    epoch_path = _skillbook_epoch_path(skill_dir)
    epoch_path.parent.mkdir(parents=True, exist_ok=True)
    epoch_path.write_text(epoch, encoding="utf-8")
    return epoch


@contextmanager
def _skillbook_lock(skill_dir: Path):
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
    context transfer to the Reflector.

    Usage:
        learner = ACELearner(cwd="/path/to/project")
        learner.learn_from_transcript("/path/to/transcript.jsonl")
    """

    def __init__(
        self,
        cwd: str,
        skillbook_path: Optional[Path] = None,
        skill_dir: Optional[Path] = None,
        ace_llm: Optional[Any] = None,
    ):
        """
        Initialize the learner.

        Args:
            cwd: Working directory for skill storage
            skillbook_path: Where to store the persistent skillbook
            skill_dir: Where to write skill files
            ace_llm: Custom LLM client (optional, for testing)
        """
        self.cwd = cwd

        if skill_dir:
            self.skill_dir = skill_dir
        else:
            try:
                self.skill_dir = get_project_skill_dir(cwd)
            except NotInProjectError:
                self.skill_dir = (
                    Path.home() / ".claude" / "skills" / "ace-learnings-global"
                )
                logger.info(f"No project root found, using global: {self.skill_dir}")

        self.skill_generator = SkillGenerator(self.skill_dir)
        self.skillbook_path = skillbook_path or (self.skill_dir / "skillbook.json")
        self._skillbook_epoch = _ensure_skillbook_epoch(self.skill_dir)

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
        """Persist an UpdateBatch safely."""
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
    # Find latest transcript
    transcript_path = find_latest_transcript()
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

    # Determine skill directory
    try:
        if hasattr(args, "project") and args.project:
            project_root = Path(args.project).resolve()
            skill_dir = project_root / ".claude" / "skills" / "ace-learnings"
        else:
            project_root = find_project_root(Path(cwd))
            if project_root:
                skill_dir = project_root / ".claude" / "skills" / "ace-learnings"
            else:
                skill_dir = Path.home() / ".claude" / "skills" / "ace-learnings-global"
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
        learner = ACELearner(cwd=cwd, skill_dir=skill_dir)
        success = learner.learn_from_transcript(transcript_path, start_line=start_line)

        if success:
            print("\n✓ Learning complete!")
            print(f"Skills updated: {skill_dir / 'SKILL.md'}")
        else:
            print("\n✗ Learning failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Learning failed: {e}", exc_info=True)
        print(f"\n✗ Learning failed: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_insights(args):
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
        skill_dir = get_project_skill_dir(str(project_root))
        skillbook_path = skill_dir / "skillbook.json"
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

        with _skillbook_lock(skill_dir):
            skillbook = Skillbook.load_from_file(str(skillbook_path))
            skillbook.remove_skill(target.id)
            _atomic_write_text(skillbook_path, skillbook.dumps())

            generator = SkillGenerator(skill_dir)
            generator.save(skillbook)

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
        skill_dir = get_project_skill_dir(str(project_root))
        skillbook_path = skill_dir / "skillbook.json"
    except NotInProjectError as e:
        print(str(e), file=sys.stderr)
        return

    try:
        with _skillbook_lock(skill_dir):
            _bump_skillbook_epoch(skill_dir)
            skillbook = Skillbook()
            _atomic_write_text(skillbook_path, skillbook.dumps())

            categories_dir = skill_dir / "categories"
            if categories_dir.exists():
                shutil.rmtree(categories_dir)

            generator = SkillGenerator(skill_dir)
            generator.save(skillbook)

        print(f"All insights cleared for project: {project_root}")
        print("ACE will start fresh.")

    except Exception as e:
        print(f"Error clearing insights: {e}")


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

    # 3. Check skill output location
    print("\n3. Skill output location...")
    try:
        cwd = Path.cwd()
        project_root = find_project_root(cwd)
        if project_root:
            skill_dir = project_root / ".claude" / "skills" / "ace-learnings"
            print(f"   Project root: {project_root}")
            print(f"   Skills will go to: {skill_dir}")
            if skill_dir.exists():
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

    # 4. Check TOON availability
    print("\n4. TOON compression...")
    try:
        from toon import encode

        print("   ✓ python-toon installed")
    except ImportError:
        print("   ⚠ python-toon not installed (will use JSON fallback)")
        print("     Install with: pip install python-toon")

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
        description="ACE learning for Claude Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  ace-learn                  Learn from full latest transcript
  ace-learn --lines 500      Learn from last 500 lines only
  ace-learn doctor           Verify prerequisites
  ace-learn insights         Show learned strategies
  ace-learn remove <id>      Remove a specific insight
  ace-learn clear --confirm  Clear all insights

Skills are stored per-project at: <project>/.claude/skills/ace-learnings/
Global fallback: ~/.claude/skills/ace-learnings-global/
""",
    )

    subparsers = parser.add_subparsers(dest="command")

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

    if args.command == "doctor":
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
