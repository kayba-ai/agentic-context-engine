#!/usr/bin/env python3
"""
ACE System Prompt Patcher

Inspired by tweakcc, this script patches Claude Code's cli.js to replace
the full system prompt with a minimal one for ACE learning.

The mechanism:
1. Find Claude Code's cli.js file
2. Locate the main system prompt template literal
3. Replace it with minimal ACE-focused content
4. Save to a separate location (doesn't modify original)

Usage:
    python ace_prompt_patcher.py find     # Find Claude Code installation
    python ace_prompt_patcher.py patch    # Create patched copy for ACE
    python ace_prompt_patcher.py restore  # Restore from backup
    python ace_prompt_patcher.py test     # Test the patched version
"""

import os
import re
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

# Where to store the patched version
ACE_DIR = Path.home() / ".ace"
ACE_CLAUDE_DIR = ACE_DIR / "claude-learner"
ACE_CLI_JS = ACE_CLAUDE_DIR / "cli.js"
ACE_BACKUP = ACE_CLAUDE_DIR / "cli.js.original"

# Anchor text inside Claude Code's *main* system prompt template literal.
# Used to locate the exact template-literal payload to replace, without
# needing versioned prompt data (unlike tweakcc).
MAIN_SYSTEM_PROMPT_ANCHORS = [
    "You are an interactive CLI tool that helps users",
]

# Minimal system prompt for ACE learning runs (Reflector/SkillManager).
# This is deliberately short to reduce token overhead and to prevent tool use
# in `claude --print` mode.
ACE_MINIMAL_SYSTEM_PROMPT = (
    "\n"
    "You are an ACE Learning Analyzer.\n"
    "\n"
    "CRITICAL:\n"
    "- Do NOT use any tools.\n"
    "- Follow the user prompt exactly.\n"
    "- If asked for JSON, output ONLY valid JSON with no surrounding text.\n"
)


class PatchError(Exception):
    """Raised when patching fails."""


# Minimal system prompt pieces for ACE learning
# We replace specific constant strings in cli.js
ACE_REPLACEMENTS = [
    # Replace the security policy with a minimal one
    (
        "IMPORTANT: Assist with authorized security testing, defensive security, CTF challenges, and educational contexts. Refuse requests for destructive techniques, DoS attacks, mass targeting, supply chain compromise, or detection evasion for malicious purposes.",
        "IMPORTANT: You are in ANALYSIS MODE. Do NOT use any tools. Just analyze text and output JSON."
    ),
    # Replace the main instruction
    (
        "You are Claude Code, Anthropic's official CLI for Claude.",
        "You are an ACE Learning Analyzer. Output JSON only."
    ),
    # Add a marker at the start of tone/style section
    (
        "# Tone and style\n- Only use emojis if the user explicitly requests it.",
        "# ACE ANALYSIS MODE\n- Output ONLY valid JSON\n- Do NOT use any tools\n- Analyze the provided text and extract insights"
    ),
]


def find_claude_cli_js() -> Optional[Path]:
    """Find Claude Code's cli.js file."""
    home = Path.home()

    # Common installation paths (from tweakcc)
    search_paths = [
        # Local Claude installation
        home / ".claude" / "local" / "node_modules" / "@anthropic-ai" / "claude-code" / "cli.js",
        # Homebrew (macOS)
        Path("/opt/homebrew/lib/node_modules/@anthropic-ai/claude-code/cli.js"),
        # Global npm
        home / ".npm-global" / "lib" / "node_modules" / "@anthropic-ai" / "claude-code" / "cli.js",
        # Various other locations
        Path("/usr/local/lib/node_modules/@anthropic-ai/claude-code/cli.js"),
        home / ".local" / "lib" / "node_modules" / "@anthropic-ai" / "claude-code" / "cli.js",
    ]

    # Also try to find via `which claude`
    try:
        import subprocess
        result = subprocess.run(["which", "claude"], capture_output=True, text=True)
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
                potential = claude_path.parent.parent / "lib" / "node_modules" / "@anthropic-ai" / "claude-code" / "cli.js"
                if potential.exists():
                    search_paths.insert(0, potential)
    except Exception:
        pass

    for path in search_paths:
        if path.exists():
            return path

    return None


def extract_version(content: str) -> Optional[str]:
    """Extract Claude Code version from cli.js content."""
    matches = re.findall(r'\bVERSION:"(\d+\.\d+\.\d+)"', content)
    if matches:
        # Return most common version
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

    `start` must point to the first character *inside* the expression (right after `{`).
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

    `start` must point to the first character *inside* the template literal (right after the opening `).
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
    Locate the *content* bounds (start, end) of Claude Code's main system prompt template literal.

    Returns:
        (start, end) where content[start:end] is the template payload to replace.
        (The backticks themselves are NOT included.)
    """
    # Some anchor strings appear multiple times in cli.js. Prefer an anchor
    # occurrence that is actually inside a `return[``...``]` template literal.
    for anchor in MAIN_SYSTEM_PROMPT_ANCHORS:
        search_pos = 0
        while True:
            anchor_idx = content.find(anchor, search_pos)
            if anchor_idx == -1:
                break

            # Look for the nearest `return[`` before this anchor occurrence.
            # Limit search to a local window for performance.
            window_start = max(0, anchor_idx - 50_000)
            window = content[window_start:anchor_idx]
            matches = list(re.finditer(r"return\s*\[\s*`", window))
            if matches:
                start = window_start + matches[-1].end()  # after opening backtick
                try:
                    end_backtick = _find_template_literal_end(content, start)
                except PatchError:
                    end_backtick = -1

                if end_backtick != -1 and start <= anchor_idx <= end_backtick:
                    return (start, end_backtick)

            search_pos = anchor_idx + 1

    return None


def _escape_for_template_literal(text: str) -> str:
    # Avoid accidentally terminating the template literal.
    return text.replace("`", "\\`")


def patch_main_system_prompt_template(content: str, new_prompt: str) -> Tuple[str, bool]:
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

    Primary strategy:
      - Replace the entire main system prompt template literal with a minimal one.

    Fallback strategy:
      - Targeted string replacements for a few known prompt fragments.
    """
    print(f"Reading: {original_path}")
    content = original_path.read_text(encoding='utf-8')
    original_size = len(content)

    version = extract_version(content)
    print(f"Claude Code version: {version or 'unknown'}")
    print(f"Original size: {original_size:,} bytes")

    patched_content = content
    patches_applied: List[str] = []

    # 1) Replace the full main system prompt (big token savings, also hides tools).
    patched_content, ok = patch_main_system_prompt_template(
        patched_content, ACE_MINIMAL_SYSTEM_PROMPT
    )
    if ok:
        patches_applied.append("main-system-prompt")

    # 2) Apply targeted replacements as additional hardening.
    replacements_made = 0
    for old_text, new_text in ACE_REPLACEMENTS:
        if old_text in patched_content:
            patched_content = patched_content.replace(old_text, new_text)
            replacements_made += 1
            print(f"  ✓ Replaced: '{old_text[:50]}...'")
        else:
            # Try with different escape patterns (cli.js may escape quotes differently)
            alt_old = old_text.replace("\\'", "'").replace('\\"', '"')
            if alt_old in patched_content:
                patched_content = patched_content.replace(alt_old, new_text)
                replacements_made += 1
                print(f"  ✓ Replaced (alt): '{old_text[:50]}...'")
            else:
                print(f"  ✗ Not found: '{old_text[:50]}...'")

    if replacements_made:
        patches_applied.append(
            f"targeted-fragments:{replacements_made}/{len(ACE_REPLACEMENTS)}"
        )

    if patched_content == content:
        print("\nERROR: No changes could be applied.")
        print(
            "The Claude Code cli.js format may have changed, or it may already be patched."
        )
        return False

    if patches_applied:
        print(f"\nApplied patches: {', '.join(patches_applied)}")

    # Calculate size difference
    new_size = len(patched_content)
    diff = original_size - new_size
    print(f"New size: {new_size:,} bytes ({diff:+,} bytes)")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write patched content
    output_path.write_text(patched_content, encoding='utf-8')
    print(f"Wrote patched cli.js to: {output_path}")

    return True


def create_wrapper_script(cli_js_path: Path) -> Path:
    """Create a shell script that runs the patched cli.js."""
    wrapper_path = cli_js_path.parent / "claude-ace"

    wrapper_content = f'''#!/usr/bin/env node
// ACE-patched Claude Code wrapper
require('{cli_js_path}');
'''

    wrapper_path.write_text(wrapper_content)
    wrapper_path.chmod(0o755)
    print(f"Created wrapper script: {wrapper_path}")

    return wrapper_path


def cmd_find():
    """Find Claude Code installation."""
    print("Searching for Claude Code installation...")

    cli_js = find_claude_cli_js()
    if cli_js:
        print(f"\nFound: {cli_js}")
        content = cli_js.read_text(encoding='utf-8')
        version = extract_version(content)
        print(f"Version: {version or 'unknown'}")
        print(f"Size: {cli_js.stat().st_size:,} bytes")

        # Check if we can locate the main system prompt template
        bounds = find_main_system_prompt_template(content)
        if bounds:
            start, end = bounds
            print(f"Main system prompt found: {end - start:,} chars")
        else:
            print("WARNING: Could not locate main system prompt template")
    else:
        print("\nCould not find Claude Code installation.")
        print("Make sure Claude Code is installed via npm or the native installer.")


def cmd_patch():
    """Create a patched copy for ACE learning."""
    cli_js = find_claude_cli_js()
    if not cli_js:
        print("ERROR: Could not find Claude Code installation.")
        print("Install with: npm install -g @anthropic-ai/claude-code")
        sys.exit(1)

    print(f"Original: {cli_js}")
    print(f"Target: {ACE_CLI_JS}")

    # Create backup of original (for reference)
    if not ACE_BACKUP.exists():
        ACE_BACKUP.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cli_js, ACE_BACKUP)
        print(f"Backed up original to: {ACE_BACKUP}")

    # Patch
    if patch_cli_js(cli_js, ACE_CLI_JS):
        print("\nPatch successful!")
        print(f"\nTo use the patched version for ACE learning:")
        print(f"  node {ACE_CLI_JS} --print -p 'your prompt'")
    else:
        print("\nPatch failed!")
        sys.exit(1)


def cmd_restore():
    """Restore from backup (remove patched version)."""
    if ACE_CLI_JS.exists():
        ACE_CLI_JS.unlink()
        print(f"Removed: {ACE_CLI_JS}")

    if ACE_BACKUP.exists():
        print(f"Backup preserved at: {ACE_BACKUP}")

    print("Restored. The original Claude Code installation is unchanged.")


def cmd_test():
    """Test the patched cli.js with a simple prompt."""
    if not ACE_CLI_JS.exists():
        print("ERROR: Patched cli.js not found. Run 'patch' first.")
        sys.exit(1)

    print(f"Testing patched cli.js at: {ACE_CLI_JS}")
    print("Running: node cli.js --print -p 'Say hello'")
    print("-" * 50)

    try:
        result = subprocess.run(
            ["node", str(ACE_CLI_JS), "--print", "-p", "Say hello in JSON format"],
            capture_output=True,
            text=True,
            timeout=30
        )
        print("STDOUT:")
        print(result.stdout[:500] if result.stdout else "(empty)")
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr[:500])
        print("-" * 50)
        print(f"Exit code: {result.returncode}")
    except subprocess.TimeoutExpired:
        print("ERROR: Command timed out after 30 seconds")
    except Exception as e:
        print(f"ERROR: {e}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "find":
        cmd_find()
    elif cmd == "patch":
        cmd_patch()
    elif cmd == "restore":
        cmd_restore()
    elif cmd == "test":
        cmd_test()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


def patch_cli(force: bool = False) -> Optional[Path]:
    """
    Convenience function to patch Claude CLI.

    Args:
        force: If True, re-patch even if patched file exists

    Returns:
        Path to patched CLI, or None if patching failed
    """
    if not force and ACE_CLI_JS.exists():
        return ACE_CLI_JS

    source = find_claude_cli_js()
    if not source:
        return None

    success = patch_cli_js(source, ACE_CLI_JS)
    return ACE_CLI_JS if success else None


if __name__ == "__main__":
    main()
