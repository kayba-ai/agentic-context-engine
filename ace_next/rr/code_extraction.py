"""Pure functions for extracting Python code from LLM responses."""

from __future__ import annotations

import re
from typing import Optional


def extract_code(response: str) -> Optional[str]:
    """Extract Python code from an LLM response.

    Uses a layered extraction approach with fallback chain:
    1. Fenced blocks: ```python, ~~~python, bare ```
    2. Indented blocks: 4-space or tab-indented code
    3. FINAL() extraction: extract just the FINAL(...) call
    """
    # Layer 1: Fenced blocks
    matches = extract_fenced_blocks(response)
    if matches:
        first_block = matches[0].strip()
        # Explicit batch request: combine all blocks
        if first_block.startswith("# BATCH"):
            return "\n\n".join(m.strip() for m in matches)
        return first_block

    # Layer 2: Indented blocks
    indented = extract_indented_block(response)
    if indented and looks_like_python(indented):
        return indented

    # Layer 3: FINAL() extraction
    final_call = extract_final_call(response)
    if final_call:
        return final_call

    return None


def extract_fenced_blocks(response: str) -> list[str]:
    """Extract all fenced code blocks from a response.

    Tries multiple fence styles in order:
    1. ```python ... ```
    2. ~~~python ... ~~~
    3. ``` ... ``` (validates as Python)
    """
    # Try ```python blocks first
    pattern = r"```python\s*(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches

    # Try ~~~python blocks
    pattern = r"~~~python\s*(.*?)~~~"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches

    # Try bare ``` blocks, but validate they look like Python
    pattern = r"```\s*(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        python_matches = [m for m in matches if looks_like_python(m)]
        if python_matches:
            return python_matches

    return []


def extract_indented_block(response: str) -> Optional[str]:
    """Extract code from an indented block (4 spaces or tab).

    Finds contiguous indented lines and returns them with indentation removed.
    """
    lines = response.split("\n")
    code_lines: list[str] = []
    in_code = False

    for line in lines:
        if line.startswith("    ") or line.startswith("\t"):
            in_code = True
            if line.startswith("    "):
                code_lines.append(line[4:])
            else:
                code_lines.append(line[1:])
        elif in_code:
            if line.strip():
                break
            code_lines.append("")

    if code_lines:
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()
        return "\n".join(code_lines)

    return None


def extract_final_call(response: str) -> Optional[str]:
    """Extract a FINAL(...) call from response text.

    Last-resort extraction when no code blocks are found.
    Extracts a complete FINAL() call with balanced parentheses.
    """
    match = re.search(r"FINAL\s*\(", response)
    if not match:
        return None

    start = match.start()
    depth = 0
    in_string: str | None = None
    escape_next = False

    for i, char in enumerate(response[match.end() - 1 :], start=match.end() - 1):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char in "\"'" and not in_string:
            in_string = char
        elif char == in_string:
            in_string = None
        elif not in_string:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return response[start : i + 1]

    return None


def looks_like_python(code: str) -> bool:
    """Check if code looks like valid Python."""
    indicators = [
        "def ",
        "import ",
        "print(",
        "FINAL(",
        "for ",
        "if ",
        "while ",
        "class ",
        "return ",
        "= ",
        "==",
        "+=",
        "try:",
        "except",
        "with ",
    ]
    return any(ind in code for ind in indicators)
