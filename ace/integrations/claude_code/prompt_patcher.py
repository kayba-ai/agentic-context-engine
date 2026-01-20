#!/usr/bin/env python3
"""
ACE System Prompt Patcher

Inspired by tweakcc, this script patches Claude Code's cli.js to replace
the full system prompt with a minimal one for ACE learning.

The mechanism:
1. Find Claude Code's cli.js file
2. Locate the system prompt assembly function
3. Replace key pieces with minimal ACE-focused content
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


def find_system_prompt_pattern(content: str) -> Optional[Tuple[int, int, str]]:
    """
    Find the main system prompt in cli.js.
    Returns (start, end, matched_content) or None.

    The system prompt typically starts with something like:
    "You are Claude Code, Anthropic's official CLI..."
    """
    # Look for the main system prompt pattern
    # This is a simplified approach - we look for known markers
    patterns = [
        # Main system prompt marker
        r'You are Claude Code, Anthropic\'s official CLI for Claude',
        r'You are Claude Code, Anthropic\\\'s official CLI for Claude',
        # Alternative patterns
        r'IMPORTANT: Assist with authorized security testing',
    ]

    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            # Find the string boundaries
            # The prompt is typically in a template literal or string
            start = match.start()

            # Walk backwards to find the start of the string
            string_start = start
            while string_start > 0:
                char = content[string_start - 1]
                if char in ['`', '"', "'"]:
                    # Check if it's escaped
                    if string_start > 1 and content[string_start - 2] == '\\':
                        string_start -= 1
                        continue
                    string_start -= 1
                    break
                string_start -= 1

            # Find the delimiter used
            delimiter = content[string_start] if string_start < len(content) else '`'

            # Walk forward to find the end of the string
            string_end = match.end()
            depth = 1 if delimiter == '`' else 0

            while string_end < len(content):
                char = content[string_end]

                if delimiter == '`':
                    # Template literal - need to handle ${...}
                    if char == '$' and string_end + 1 < len(content) and content[string_end + 1] == '{':
                        depth += 1
                        string_end += 2
                        continue
                    elif char == '}' and depth > 1:
                        depth -= 1
                        string_end += 1
                        continue
                    elif char == '`' and depth == 1:
                        # Check if escaped
                        if content[string_end - 1] != '\\':
                            string_end += 1
                            break
                else:
                    # Regular string
                    if char == delimiter and content[string_end - 1] != '\\':
                        string_end += 1
                        break

                string_end += 1

            matched = content[string_start:string_end]
            return (string_start, string_end, matched)

    return None


def patch_cli_js(original_path: Path, output_path: Path) -> bool:
    """
    Patch cli.js using targeted string replacements.

    This approach finds specific known strings in the system prompt
    and replaces them with ACE-focused alternatives, rather than
    trying to replace the entire prompt.
    """
    print(f"Reading: {original_path}")
    content = original_path.read_text(encoding='utf-8')
    original_size = len(content)

    version = extract_version(content)
    print(f"Claude Code version: {version or 'unknown'}")
    print(f"Original size: {original_size:,} bytes")

    # Apply each targeted replacement
    patched_content = content
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

    if replacements_made == 0:
        print("\nERROR: No replacements could be made.")
        print("The cli.js format may have changed.")
        return False

    print(f"\nApplied {replacements_made}/{len(ACE_REPLACEMENTS)} replacements")

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

        # Check if we can find the system prompt
        result = find_system_prompt_pattern(content)
        if result:
            print(f"System prompt found: {len(result[2]):,} chars")
        else:
            print("WARNING: Could not locate system prompt pattern")
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


if __name__ == "__main__":
    main()
