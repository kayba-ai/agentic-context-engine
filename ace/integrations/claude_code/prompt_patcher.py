"""
ACE System Prompt Patcher - Internal utility.

Patches Claude Code's cli.js to replace the massive system prompt with a minimal
one for ACE learning. This prevents tool_use attempts in --print mode and reduces
token overhead significantly.

This module is used internally by CLIClient to auto-patch when possible.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# Storage locations for patched CLI
ACE_DIR = Path.home() / ".ace"
ACE_CLAUDE_DIR = ACE_DIR / "claude-learner"
ACE_CLI_JS = ACE_CLAUDE_DIR / "cli.js"
ACE_BACKUP = ACE_CLAUDE_DIR / "cli.js.original"
ACE_PATCH_LOCK = ACE_CLAUDE_DIR / ".patch.lock"
ACE_SOURCE_VERSION = ACE_CLAUDE_DIR / ".source-version"

# Anchor text to locate the main system prompt template literal
MAIN_SYSTEM_PROMPT_ANCHORS = [
    "You are an interactive CLI tool that helps users",
]

# Minimal system prompt for ACE learning runs (Reflector/SkillManager)
ACE_MINIMAL_SYSTEM_PROMPT = """
You are an ACE Learning Analyzer.

CRITICAL:
- Do NOT use any tools.
- Follow the user prompt exactly.
- If asked for JSON, output ONLY valid JSON with no surrounding text.
"""

# Additional targeted replacements for hardening
ACE_REPLACEMENTS = [
    (
        "IMPORTANT: Assist with authorized security testing, defensive security, CTF challenges, and educational contexts. Refuse requests for destructive techniques, DoS attacks, mass targeting, supply chain compromise, or detection evasion for malicious purposes.",
        "IMPORTANT: You are in ANALYSIS MODE. Do NOT use any tools. Just analyze text and output JSON.",
    ),
    (
        "You are Claude Code, Anthropic's official CLI for Claude.",
        "You are an ACE Learning Analyzer. Output JSON only.",
    ),
    (
        "# Tone and style\n- Only use emojis if the user explicitly requests it.",
        "# ACE ANALYSIS MODE\n- Output ONLY valid JSON\n- Do NOT use any tools\n- Analyze the provided text and extract insights",
    ),
]


class PatchError(Exception):
    """Raised when patching fails."""


@contextmanager
def _file_lock(lock_path: Path):
    """Acquire an exclusive file lock for concurrent safety."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w", encoding="utf-8") as f:
        if os.name == "posix":
            try:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
        yield


def find_claude_cli_js() -> Optional[Path]:
    """
    Find Claude Code's cli.js file.

    Searches common installation paths and also tries to find via `which claude`.

    Returns:
        Path to cli.js, or None if not found
    """
    home = Path.home()

    # Common installation paths
    search_paths = [
        # Local Claude installation
        home
        / ".claude"
        / "local"
        / "node_modules"
        / "@anthropic-ai"
        / "claude-code"
        / "cli.js",
        # Homebrew (macOS)
        Path("/opt/homebrew/lib/node_modules/@anthropic-ai/claude-code/cli.js"),
        # Global npm
        home
        / ".npm-global"
        / "lib"
        / "node_modules"
        / "@anthropic-ai"
        / "claude-code"
        / "cli.js",
        # System locations
        Path("/usr/local/lib/node_modules/@anthropic-ai/claude-code/cli.js"),
        home
        / ".local"
        / "lib"
        / "node_modules"
        / "@anthropic-ai"
        / "claude-code"
        / "cli.js",
    ]

    # Also try to find via `which claude`
    try:
        result = subprocess.run(
            ["which", "claude"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            claude_path = Path(result.stdout.strip())
            # Resolve symlinks
            claude_path = claude_path.resolve()
            # If it's a JS file, use it directly
            if claude_path.suffix == ".js":
                search_paths.insert(0, claude_path)
            else:
                # It might be a symlink to the package, check parent
                cli_js = claude_path.parent / "cli.js"
                if cli_js.exists():
                    search_paths.insert(0, cli_js)
                # Or it might be in node_modules relative path
                potential = (
                    claude_path.parent.parent
                    / "lib"
                    / "node_modules"
                    / "@anthropic-ai"
                    / "claude-code"
                    / "cli.js"
                )
                if potential.exists():
                    search_paths.insert(0, potential)
    except Exception:
        pass

    for path in search_paths:
        if path.exists():
            logger.debug(f"Found cli.js at: {path}")
            return path

    return None


def extract_version(content: str) -> Optional[str]:
    """Extract Claude Code version from cli.js content."""
    matches = re.findall(r'\bVERSION:"(\d+\.\d+\.\d+)"', content)
    if matches:
        from collections import Counter

        return Counter(matches).most_common(1)[0][0]
    return None


def _skip_string(content: str, start: int, quote: str) -> int:
    """Skip over a JS string literal starting at `start` (on the opening quote)."""
    i = start + 1
    while i < len(content):
        ch = content[i]
        if ch == "\\":
            i += 2
            continue
        if ch == quote:
            return i + 1
        i += 1
    raise PatchError("Unterminated string literal while parsing template expression")


def _skip_line_comment(content: str, start: int) -> int:
    i = start + 2
    while i < len(content) and content[i] != "\n":
        i += 1
    return i


def _skip_block_comment(content: str, start: int) -> int:
    end = content.find("*/", start + 2)
    if end == -1:
        raise PatchError("Unterminated block comment while parsing template expression")
    return end + 2


def _skip_template_literal_in_expression(content: str, start: int) -> int:
    """Skip over a template literal occurring inside a `${...}` expression."""
    i = start + 1
    while i < len(content):
        ch = content[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "`":
            return i + 1
        if ch == "$" and i + 1 < len(content) and content[i + 1] == "{":
            end_brace = _find_matching_brace(content, i + 2)
            i = end_brace + 1
            continue
        i += 1
    raise PatchError("Unterminated template literal inside ${...} expression")


def _find_matching_brace(content: str, start: int) -> int:
    """
    Find the matching closing brace for a `${ ... }` expression.

    `start` must point to the first character *inside* the expression.
    Returns the index of the matching `}`.
    """
    depth = 1
    i = start
    while i < len(content):
        ch = content[i]

        if ch == "\\":
            i += 2
            continue

        if ch in ("'", '"'):
            i = _skip_string(content, i, ch)
            continue

        if ch == "`":
            i = _skip_template_literal_in_expression(content, i)
            continue

        if ch == "/" and i + 1 < len(content):
            nxt = content[i + 1]
            if nxt == "/":
                i = _skip_line_comment(content, i)
                continue
            if nxt == "*":
                i = _skip_block_comment(content, i)
                continue

        if ch == "{":
            depth += 1
            i += 1
            continue

        if ch == "}":
            depth -= 1
            if depth == 0:
                return i
            i += 1
            continue

        i += 1

    raise PatchError("Unterminated ${...} expression while finding template end")


def _find_template_literal_end(content: str, start: int) -> int:
    """
    Find the closing backtick for a template literal.

    `start` must point to the first character *inside* the template literal.
    Returns the index of the closing backtick.
    """
    i = start
    while i < len(content):
        ch = content[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "`":
            return i
        if ch == "$" and i + 1 < len(content) and content[i + 1] == "{":
            end_brace = _find_matching_brace(content, i + 2)
            i = end_brace + 1
            continue
        i += 1
    raise PatchError("Unterminated template literal while locating main system prompt")


def find_main_system_prompt_template(content: str) -> Optional[Tuple[int, int]]:
    """
    Locate the content bounds of Claude Code's main system prompt template literal.

    Returns:
        (start, end) where content[start:end] is the template payload to replace.
        The backticks themselves are NOT included.
    """
    for anchor in MAIN_SYSTEM_PROMPT_ANCHORS:
        search_pos = 0
        while True:
            anchor_idx = content.find(anchor, search_pos)
            if anchor_idx == -1:
                break

            # Look for the nearest `return[`` before this anchor
            window_start = max(0, anchor_idx - 50_000)
            window = content[window_start:anchor_idx]
            matches = list(re.finditer(r"return\s*\[\s*`", window))
            if matches:
                start = window_start + matches[-1].end()
                try:
                    end_backtick = _find_template_literal_end(content, start)
                except PatchError:
                    end_backtick = -1

                if end_backtick != -1 and start <= anchor_idx <= end_backtick:
                    return (start, end_backtick)

            search_pos = anchor_idx + 1

    return None


def _escape_for_template_literal(text: str) -> str:
    """Escape text for use inside a JS template literal."""
    return text.replace("`", "\\`")


def patch_main_system_prompt_template(
    content: str, new_prompt: str
) -> Tuple[str, bool]:
    """Replace the main system prompt template with a new prompt."""
    bounds = find_main_system_prompt_template(content)
    if not bounds:
        return content, False

    start, end = bounds
    replacement = _escape_for_template_literal(new_prompt)

    if content[start:end] == replacement:
        return content, True

    patched = content[:start] + replacement + content[end:]
    return patched, True


def patch_cli_js(original_path: Path, output_path: Path) -> bool:
    """
    Patch cli.js for ACE learning.

    Primary strategy: Replace the entire main system prompt template literal.
    Fallback strategy: Targeted string replacements for known prompt fragments.

    Args:
        original_path: Path to original cli.js
        output_path: Path where patched version will be written

    Returns:
        True if patching succeeded
    """
    logger.info(f"Patching cli.js from: {original_path}")
    content = original_path.read_text(encoding="utf-8")
    original_size = len(content)

    version = extract_version(content)
    logger.info(f"Claude Code version: {version or 'unknown'}")
    logger.debug(f"Original size: {original_size:,} bytes")

    patched_content = content
    patches_applied: List[str] = []

    # 1) Replace the full main system prompt (big token savings)
    patched_content, ok = patch_main_system_prompt_template(
        patched_content, ACE_MINIMAL_SYSTEM_PROMPT
    )
    if ok:
        patches_applied.append("main-system-prompt")
        logger.debug("Applied main system prompt patch")

    # 2) Apply targeted replacements as additional hardening
    replacements_made = 0
    for old_text, new_text in ACE_REPLACEMENTS:
        if old_text in patched_content:
            patched_content = patched_content.replace(old_text, new_text)
            replacements_made += 1
        else:
            # Try with different escape patterns
            alt_old = old_text.replace("\\'", "'").replace('\\"', '"')
            if alt_old in patched_content:
                patched_content = patched_content.replace(alt_old, new_text)
                replacements_made += 1

    if replacements_made:
        patches_applied.append(
            f"targeted-fragments:{replacements_made}/{len(ACE_REPLACEMENTS)}"
        )
        logger.debug(f"Applied {replacements_made} targeted replacements")

    if patched_content == content:
        logger.warning("No changes could be applied to cli.js")
        return False

    logger.info(f"Applied patches: {', '.join(patches_applied)}")

    # Calculate size difference
    new_size = len(patched_content)
    diff = original_size - new_size
    logger.debug(f"New size: {new_size:,} bytes ({diff:+,} bytes)")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write patched content
    output_path.write_text(patched_content, encoding="utf-8")
    logger.info(f"Wrote patched cli.js to: {output_path}")

    # Store source version for freshness check
    if version:
        ACE_SOURCE_VERSION.write_text(version, encoding="utf-8")

    return True


def _is_patch_fresh(source_path: Path) -> bool:
    """
    Check if the patched CLI is fresh (same source version).

    Returns True if patched CLI exists and matches source version.
    """
    if not ACE_CLI_JS.exists():
        return False

    if not ACE_SOURCE_VERSION.exists():
        return False

    try:
        stored_version = ACE_SOURCE_VERSION.read_text(encoding="utf-8").strip()
        source_content = source_path.read_text(encoding="utf-8")
        current_version = extract_version(source_content)

        if current_version and stored_version == current_version:
            return True

        logger.info(
            f"Source version changed: {stored_version} -> {current_version or 'unknown'}"
        )
        return False
    except Exception as e:
        logger.debug(f"Error checking patch freshness: {e}")
        return False


def get_or_create_patched_cli(force: bool = False) -> Optional[Path]:
    """
    Get path to patched CLI, creating it if necessary.

    This is the main entry point for CLIClient. It handles:
    - Concurrent-safe patching via file lock
    - Freshness check (re-patch if source version changed)
    - Graceful fallback if patching fails

    Args:
        force: If True, re-patch even if patched file exists

    Returns:
        Path to patched cli.js, or None if patching failed/unavailable
    """
    with _file_lock(ACE_PATCH_LOCK):
        source = find_claude_cli_js()
        if not source:
            logger.debug("Could not find Claude Code cli.js to patch")
            return None

        # Check if we have a fresh patch
        if not force and ACE_CLI_JS.exists() and _is_patch_fresh(source):
            logger.debug(f"Using existing patched CLI: {ACE_CLI_JS}")
            return ACE_CLI_JS

        # Create backup of original (for reference)
        if not ACE_BACKUP.exists():
            ACE_BACKUP.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, ACE_BACKUP)
            logger.debug(f"Backed up original to: {ACE_BACKUP}")

        # Patch
        try:
            success = patch_cli_js(source, ACE_CLI_JS)
            if success:
                return ACE_CLI_JS
        except Exception as e:
            logger.warning(f"Failed to patch cli.js: {e}")

        return None


def is_patched_cli_available() -> bool:
    """Check if a patched CLI is available (without creating one)."""
    return ACE_CLI_JS.exists()


def get_patch_info() -> dict:
    """
    Get information about the current patch status.

    Returns:
        Dict with patch status information for diagnostics
    """
    info = {
        "patched_cli_exists": ACE_CLI_JS.exists(),
        "patched_cli_path": str(ACE_CLI_JS) if ACE_CLI_JS.exists() else None,
        "source_cli_path": None,
        "source_version": None,
        "patched_version": None,
        "is_fresh": False,
    }

    source = find_claude_cli_js()
    if source:
        info["source_cli_path"] = str(source)
        try:
            content = source.read_text(encoding="utf-8")
            info["source_version"] = extract_version(content)
        except Exception:
            pass

    if ACE_SOURCE_VERSION.exists():
        try:
            info["patched_version"] = ACE_SOURCE_VERSION.read_text(
                encoding="utf-8"
            ).strip()
        except Exception:
            pass

    if source and info["patched_cli_exists"]:
        info["is_fresh"] = _is_patch_fresh(source)

    return info
