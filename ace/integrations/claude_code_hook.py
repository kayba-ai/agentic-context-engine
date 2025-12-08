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

from ..playbook import Playbook, Bullet
from ..roles import Reflector, Curator, GeneratorOutput, ReflectorOutput
from ..llm_providers import LiteLLMClient
from ..prompts_v2_1 import PromptManager

logger = logging.getLogger(__name__)


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


class SkillGenerator:
    """Generate Claude Code skill files from ACE playbook."""

    DEFAULT_SKILL_DIR = Path.home() / ".claude" / "skills" / "ace-learnings"

    def __init__(self, skill_dir: Optional[Path] = None):
        self.skill_dir = skill_dir or self.DEFAULT_SKILL_DIR

    def generate(self, playbook: Playbook) -> str:
        """
        Generate SKILL.md content from playbook.

        Args:
            playbook: ACE Playbook with learned strategies

        Returns:
            Markdown content for SKILL.md
        """
        bullets = playbook.bullets()
        if not bullets:
            return self._empty_skill()

        # Sort bullets by effectiveness (helpful - harmful)
        sorted_bullets = sorted(
            bullets, key=lambda b: (b.helpful - b.harmful, b.helpful), reverse=True
        )

        # Group by section
        sections: Dict[str, List[Bullet]] = {}
        for bullet in sorted_bullets:
            sections.setdefault(bullet.section, []).append(bullet)

        # Build skill content
        parts = [self._frontmatter()]
        parts.append(self._intro())

        # Top strategies (cross-section, top 10 by effectiveness)
        top_bullets = sorted_bullets[:10]
        if top_bullets:
            parts.append(self._top_strategies(top_bullets))

        # Section-specific strategies
        for section, section_bullets in sorted(sections.items()):
            parts.append(self._section(section, section_bullets[:10]))

        # Antipatterns (high harmful count)
        antipatterns = [b for b in bullets if b.harmful > b.helpful]
        if antipatterns:
            parts.append(self._antipatterns(antipatterns[:5]))

        parts.append(self._footer())

        return "\n\n".join(parts)

    def _frontmatter(self) -> str:
        return """---
name: ace-learnings
description: Learned coding strategies from past sessions. Apply when writing code, debugging, testing, navigating codebases, or making architectural decisions. Contains proven patterns and common pitfalls to avoid.
---"""

    def _intro(self) -> str:
        return """# ACE Learned Strategies

These strategies have been automatically learned from coding sessions.
Apply relevant strategies based on the current task."""

    def _empty_skill(self) -> str:
        return f"""{self._frontmatter()}

# ACE Learned Strategies

No strategies learned yet. Strategies will appear here as you use Claude Code.

{self._footer()}"""

    def _top_strategies(self, bullets: List[Bullet]) -> str:
        lines = ["## Top Strategies (by effectiveness)"]
        for i, b in enumerate(bullets, 1):
            score = f"({b.helpful}↑ {b.harmful}↓)"
            lines.append(f"{i}. {b.content} {score}")
        return "\n".join(lines)

    def _section(self, section: str, bullets: List[Bullet]) -> str:
        # Convert section name to title case
        title = section.replace("_", " ").title()
        lines = [f"## {title}"]
        for b in bullets:
            score = f"({b.helpful}↑ {b.harmful}↓)"
            lines.append(f"- {b.content} {score}")
        return "\n".join(lines)

    def _antipatterns(self, bullets: List[Bullet]) -> str:
        lines = ["## Antipatterns (avoid these)"]
        for b in bullets:
            score = f"({b.harmful} failures)"
            lines.append(f"- ⚠️ {b.content} {score}")
        return "\n".join(lines)

    def _footer(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        return f"""---
*Auto-generated by ACE at {timestamp}*"""

    def save(self, playbook: Playbook) -> Path:
        """
        Save skill file to disk.

        Args:
            playbook: ACE Playbook to generate skill from

        Returns:
            Path to saved SKILL.md file
        """
        self.skill_dir.mkdir(parents=True, exist_ok=True)

        content = self.generate(playbook)
        skill_path = self.skill_dir / "SKILL.md"

        skill_path.write_text(content, encoding="utf-8")
        logger.info(f"Saved skill file to {skill_path}")

        return skill_path


# ============================================================================
# Main Hook Learner
# ============================================================================


class ACEHookLearner:
    """
    Main class for learning from Claude Code sessions via hooks.

    Usage:
        learner = ACEHookLearner()
        learner.learn_from_hook()  # Reads stdin, processes, updates skill
    """

    # Store playbook in same directory as SKILL.md for portability
    DEFAULT_SKILL_DIR = SkillGenerator.DEFAULT_SKILL_DIR
    DEFAULT_PLAYBOOK_PATH = DEFAULT_SKILL_DIR / "playbook.json"

    def __init__(
        self,
        playbook_path: Optional[Path] = None,
        skill_dir: Optional[Path] = None,
        ace_model: str = "anthropic/claude-sonnet-4-5-20250929",
        ace_llm: Optional[LiteLLMClient] = None,
    ):
        """
        Initialize the hook learner.

        Args:
            playbook_path: Where to store the persistent playbook
            skill_dir: Where to write the skill file
            ace_model: Model for ACE Reflector/Curator
            ace_llm: Custom LLM client (overrides ace_model)
        """
        # Use skill_dir for both skill and playbook if provided
        if skill_dir:
            self.skill_generator = SkillGenerator(skill_dir)
            self.playbook_path = playbook_path or (skill_dir / "playbook.json")
        else:
            self.skill_generator = SkillGenerator()
            self.playbook_path = playbook_path or self.DEFAULT_PLAYBOOK_PATH
        self.transcript_parser = TranscriptParser()

        # Load or create playbook
        if self.playbook_path.exists():
            self.playbook = Playbook.load_from_file(str(self.playbook_path))
            logger.info(f"Loaded playbook with {len(self.playbook.bullets())} bullets")
        else:
            self.playbook = Playbook()
            logger.info("Created new playbook")

        # Create ACE components with v2.1 prompts for better effectiveness
        self.ace_llm = ace_llm or LiteLLMClient(model=ace_model, max_tokens=2048)
        prompt_mgr = PromptManager()
        self.reflector = Reflector(
            self.ace_llm, prompt_template=prompt_mgr.get_reflector_prompt()
        )
        self.curator = Curator(
            self.ace_llm, prompt_template=prompt_mgr.get_curator_prompt()
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _run_reflector_with_retry(
        self, task: str, generator_output: GeneratorOutput, feedback: str
    ):
        """Run Reflector with retry on transient failures."""
        return self.reflector.reflect(
            question=task,
            generator_output=generator_output,
            playbook=self.playbook,
            ground_truth=None,
            feedback=feedback,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _run_curator_with_retry(self, reflection, cwd: str, progress: str):
        """Run Curator with retry on transient failures."""
        return self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=f"Claude Code session in {cwd}",
            progress=progress,
        )

    def learn_from_hook(self) -> bool:
        """
        Process hook input from stdin and learn.

        Expected stdin JSON:
        {
            "session_id": "...",
            "transcript_path": "/path/to/transcript.jsonl",
            "cwd": "...",
            "hook_event_name": "Stop"
        }

        Returns:
            True if learning succeeded, False otherwise
        """
        try:
            # Read hook input from stdin
            hook_input = json.load(sys.stdin)
            transcript_path = hook_input.get("transcript_path")

            if not transcript_path:
                logger.error("No transcript_path in hook input")
                return False

            return self.learn_from_transcript(transcript_path)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse hook input: {e}")
            return False
        except Exception as e:
            logger.error(f"Hook learning failed: {e}")
            return False

    def learn_from_transcript(self, transcript_path: str) -> bool:
        """
        Learn from a transcript file directly.

        Args:
            transcript_path: Path to Claude Code transcript JSONL

        Returns:
            True if learning succeeded
        """
        try:
            # Parse transcript
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

            # Create GeneratorOutput for Reflector
            generator_output = GeneratorOutput(
                reasoning=transcript.to_execution_trace(),
                final_answer=(
                    transcript.turns[-1].assistant_text if transcript.turns else ""
                ),
                bullet_ids=[],
                raw={
                    "total_tools": transcript.total_tool_calls,
                    "failed_tools": transcript.failed_tool_calls,
                },
            )

            # Run Reflector with retry
            logger.info("Running Reflector...")
            reflection = self._run_reflector_with_retry(
                task=task,
                generator_output=generator_output,
                feedback=transcript.get_feedback(),
            )

            # Run Curator with retry
            logger.info("Running Curator...")
            curator_output = self._run_curator_with_retry(
                reflection=reflection,
                cwd=transcript.cwd,
                progress=f"{transcript.successful_tool_calls}/{transcript.total_tool_calls} successful",
            )

            # Update playbook
            self.playbook.apply_delta(curator_output.delta)
            logger.info(f"Playbook now has {len(self.playbook.bullets())} bullets")

            # Save playbook
            self.playbook_path.parent.mkdir(parents=True, exist_ok=True)
            self.playbook.save_to_file(str(self.playbook_path))

            # Update skill file
            self.skill_generator.save(self.playbook)

            return True

        except Exception as e:
            logger.error(f"Learning failed: {e}", exc_info=True)
            return False


# ============================================================================
# CLI Entry Point
# ============================================================================


def setup_hook():
    """Configure Claude Code to use ACE learning hook."""
    settings_path = Path.home() / ".claude" / "settings.json"

    # Load existing settings or create new
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            settings = {}
    else:
        settings = {}

    # Add/update hook config (merge, don't overwrite)
    if "hooks" not in settings:
        settings["hooks"] = {}

    settings["hooks"]["Stop"] = [
        {"matcher": "*", "hooks": [{"type": "command", "command": "ace-learn"}]}
    ]

    # Write back
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2))

    # Create .env template if it doesn't exist
    env_path = Path.home() / ".ace" / ".env"
    if not env_path.exists():
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text(
            "# ACE Framework Configuration\n# Add your Anthropic API key here\nANTHROPIC_API_KEY=your-key-here\n"
        )

    skill_dir = ACEHookLearner.DEFAULT_SKILL_DIR
    print("✓ Claude Code hook configured!")
    print()
    print("Next steps:")
    print(f"  1. Add your API key to: {env_path}")
    print("  2. Start using Claude Code - it will learn from your sessions!")
    print()
    print("Data locations:")
    print(f"  Skill file:  {skill_dir / 'SKILL.md'}")
    print(f"  Playbook:    {skill_dir / 'playbook.json'}")
    print()
    print(f"Settings saved to: {settings_path}")

    # Create slash commands for enable/disable
    _create_slash_commands()


def enable_hook():
    """Enable ACE learning hook in Claude Code settings."""
    settings_path = Path.home() / ".claude" / "settings.json"

    # Load existing settings
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            settings = {}
    else:
        settings = {}

    # Add hook config
    if "hooks" not in settings:
        settings["hooks"] = {}

    settings["hooks"]["Stop"] = [
        {"matcher": "*", "hooks": [{"type": "command", "command": "ace-learn"}]}
    ]

    # Write back
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2))

    print("ACE learning enabled")


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


def show_insights():
    """Show current ACE learned strategies."""
    playbook_path = ACEHookLearner.DEFAULT_PLAYBOOK_PATH

    if not playbook_path.exists():
        print("No insights yet. ACE will learn from your Claude Code sessions.")
        return

    try:
        from ..playbook import Playbook

        playbook = Playbook.load_from_file(str(playbook_path))
        bullets = playbook.bullets()

        if not bullets:
            print("No insights yet. ACE will learn from your Claude Code sessions.")
            return

        print(f"ACE Learned Strategies ({len(bullets)} total):\n")

        # Group by section
        sections: dict = {}
        for bullet in bullets:
            section = bullet.section
            if section not in sections:
                sections[section] = []
            sections[section].append(bullet)

        for section, section_bullets in sorted(sections.items()):
            print(f"## {section.replace('_', ' ').title()}")
            for b in section_bullets:
                score = f"({b.helpful}↑ {b.harmful}↓)"
                print(f"  [{b.id}] {b.content} {score}")
            print()

    except Exception as e:
        print(f"Error reading playbook: {e}")


def remove_insight(insight_id: str):
    """Remove a specific insight by ID."""
    playbook_path = ACEHookLearner.DEFAULT_PLAYBOOK_PATH

    if not playbook_path.exists():
        print("No playbook found.")
        return

    try:
        from ..playbook import Playbook

        playbook = Playbook.load_from_file(str(playbook_path))

        # Find bullet by ID or partial match
        bullets = playbook.bullets()
        target = None
        for b in bullets:
            if (
                b.id == insight_id
                or insight_id in b.id
                or insight_id.lower() in b.content.lower()
            ):
                target = b
                break

        if not target:
            print(f"No insight found matching '{insight_id}'")
            print("Use 'ace-learn insights' to see available insights.")
            return

        # Remove the bullet
        playbook.remove_bullet(target.id)
        playbook.save_to_file(str(playbook_path))

        # Regenerate skill file
        skill_dir = ACEHookLearner.DEFAULT_SKILL_DIR
        generator = SkillGenerator(skill_dir)
        generator.save(playbook)

        print(f"Removed: {target.content}")

    except Exception as e:
        print(f"Error removing insight: {e}")


def clear_insights(confirm: bool = False):
    """Clear all ACE learned strategies."""
    if not confirm:
        print("This will delete all learned strategies.")
        print("Run with --confirm to proceed: ace-learn clear --confirm")
        return

    playbook_path = ACEHookLearner.DEFAULT_PLAYBOOK_PATH
    skill_path = ACEHookLearner.DEFAULT_SKILL_DIR / "SKILL.md"

    try:
        from ..playbook import Playbook

        # Create empty playbook
        playbook = Playbook()
        playbook.save_to_file(str(playbook_path))

        # Regenerate empty skill file
        skill_dir = ACEHookLearner.DEFAULT_SKILL_DIR
        generator = SkillGenerator(skill_dir)
        generator.save(playbook)

        print("All insights cleared. ACE will start fresh.")

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

After enabling, ACE will learn from your coding sessions and build a playbook of strategies.
"""
    (commands_dir / "ace-on.md").write_text(ace_on_content)

    # /ace-off command
    ace_off_content = """Disable ACE learning for Claude Code sessions.

Run this command to disable ACE learning:
```bash
ace-learn disable
```

This stops ACE from learning from your sessions. Your existing playbook is preserved.
"""
    (commands_dir / "ace-off.md").write_text(ace_off_content)

    # /ace-insights command
    ace_insights_content = """Show ACE learned strategies.

Run this command to see all learned insights:
```bash
ace-learn insights
```

Display the output to show the user their current playbook of strategies.
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

This will reset the playbook and start fresh.
"""
    (commands_dir / "ace-clear.md").write_text(ace_clear_content)


def run_learning(args):
    """Run the learning process (called from hook or manually)."""
    # Fork to background unless --sync is specified
    # This allows Claude Code to exit immediately without waiting for learning
    if not args.sync and not args.transcript:
        # Read stdin before forking (stdin won't be available in child)
        try:
            stdin_data = sys.stdin.read()
        except Exception:
            stdin_data = ""

        # Fork to background
        pid = os.fork()
        if pid > 0:
            # Parent exits immediately - Claude Code continues
            sys.exit(0)

        # Child continues with learning in background
        # Detach from terminal
        os.setsid()

        # Redirect stdin to the data we captured
        sys.stdin = __import__("io").StringIO(stdin_data)

    # Setup logging (to file in background mode)
    log_file = None
    if not args.sync and not args.transcript:
        log_dir = Path.home() / ".ace" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"ace-learn-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        filename=str(log_file) if log_file else None,
    )

    # Create learner
    learner = ACEHookLearner(
        playbook_path=Path(args.playbook),
        skill_dir=Path(args.skill_dir),
        ace_model=args.model,
    )

    # Learn from transcript or hook input
    if args.transcript:
        success = learner.learn_from_transcript(args.transcript)
    else:
        success = learner.learn_from_hook()

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
  ace-learn enable             Enable ACE learning
  ace-learn disable            Disable ACE learning
  ace-learn insights           Show learned strategies
  ace-learn remove <id>        Remove a specific insight
  ace-learn clear --confirm    Clear all insights
  ace-learn                    Learn from stdin (called by hook)
  ace-learn -t transcript.jsonl   Learn from specific transcript
""",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Setup command
    subparsers.add_parser("setup", help="Configure Claude Code to use ACE learning")

    # Enable/disable commands
    subparsers.add_parser("enable", help="Enable ACE learning hook")
    subparsers.add_parser("disable", help="Disable ACE learning hook")

    # Insight management commands
    subparsers.add_parser("insights", help="Show learned strategies")
    remove_parser = subparsers.add_parser("remove", help="Remove a specific insight")
    remove_parser.add_argument("id", help="Insight ID or keyword to match")
    clear_parser = subparsers.add_parser("clear", help="Clear all insights")
    clear_parser.add_argument(
        "--confirm", action="store_true", help="Confirm clearing all insights"
    )

    # Learning options (work without subcommand for backwards compat)
    parser.add_argument(
        "--transcript", "-t", help="Path to transcript file (if not using stdin)"
    )
    parser.add_argument(
        "--playbook",
        "-p",
        help="Path to playbook file",
        default=str(ACEHookLearner.DEFAULT_PLAYBOOK_PATH),
    )
    parser.add_argument(
        "--skill-dir",
        "-s",
        help="Directory for skill output",
        default=str(SkillGenerator.DEFAULT_SKILL_DIR),
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
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Run synchronously (default: fork to background)",
    )

    args = parser.parse_args()

    if args.command == "setup":
        setup_hook()
    elif args.command == "enable":
        enable_hook()
    elif args.command == "disable":
        disable_hook()
    elif args.command == "insights":
        show_insights()
    elif args.command == "remove":
        remove_insight(args.id)
    elif args.command == "clear":
        clear_insights(confirm=args.confirm)
    else:
        # Default: run learning (backwards compat with hook calling ace-learn)
        run_learning(args)


if __name__ == "__main__":
    main()
