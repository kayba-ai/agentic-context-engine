"""Claude Code CLI client for subscription-based LLM access."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import shutil
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from ace.llm import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class CLIClientError(Exception):
    """Error from Claude Code CLI execution."""

    pass


def _resolve_cli_path(cli_path: Optional[str]) -> Path:
    """
    Resolve the Claude CLI path with priority order:

    1. Explicit parameter (cli_path)
    2. ACE_CLI_PATH environment variable
    3. Patched cli.js (auto-created if possible)
    4. System 'claude' binary from PATH

    The patched cli.js replaces Claude Code's massive system prompt with a
    minimal ACE-focused one. This prevents tool_use attempts in --print mode
    and significantly reduces token overhead.

    Returns:
        Path to the CLI executable

    Raises:
        FileNotFoundError: If no CLI can be found
    """
    # 1. Explicit parameter
    if cli_path:
        path = Path(cli_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"CLI not found at: {cli_path}")
        return path

    # 2. Environment override
    if env_path := os.environ.get("ACE_CLI_PATH"):
        path = Path(env_path).expanduser()
        if path.exists():
            logger.info(f"Using ACE_CLI_PATH: {path}")
            return path
        logger.warning(f"ACE_CLI_PATH set but not found: {env_path}")

    # 3. Try patched cli.js (auto-created if possible)
    try:
        from .prompt_patcher import get_or_create_patched_cli

        patched = get_or_create_patched_cli()
        if patched:
            logger.info(f"Using patched CLI: {patched}")
            return patched
    except Exception as e:
        logger.debug(f"Could not create patched CLI: {e}")

    # 4. Fallback: system claude binary
    claude_path = shutil.which("claude")
    if claude_path:
        logger.info("Using system claude (patching unavailable)")
        return Path(claude_path)

    raise FileNotFoundError(
        "Claude CLI not found. Checked:\n"
        "  - ACE_CLI_PATH environment variable\n"
        "  - Patched cli.js (~/.ace/claude-learner/cli.js)\n"
        "  - 'claude' in PATH\n\n"
        "Install with: npm install -g @anthropic-ai/claude-code"
    )


def _is_js_cli(cli_path: Path) -> bool:
    """Check if the CLI path is a JavaScript file requiring node."""
    return cli_path.suffix == ".js"


class CLIClient(LLMClient):
    """
    LLM client that uses Claude Code CLI for subscription-based access.

    This client runs `claude --print` with prompts passed via stdin,
    allowing ACE to use a Claude Code subscription instead of API calls.

    CLI Resolution Order:
    1. Explicit cli_path parameter
    2. ACE_CLI_PATH environment variable
    3. System 'claude' command from PATH

    Args:
        cli_path: Path to claude CLI. If None, auto-detects using resolution order.
        timeout: Command timeout in seconds (default: 120)
        max_retries: Maximum retries on failure (default: 3)

    Example:
        >>> from ace.integrations.claude_code.cli_client import CLIClient
        >>> from ace import Reflector, SkillManager
        >>>
        >>> llm = CLIClient()
        >>> reflector = Reflector(llm=llm)
        >>> skill_manager = SkillManager(llm=llm)
    """

    def __init__(
        self,
        cli_path: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize CLI client.

        Args:
            cli_path: Path to claude CLI executable or cli.js file.
                      If None, auto-detects using resolution order.
            timeout: Command timeout in seconds
            max_retries: Maximum retries on transient failures
        """
        super().__init__(model="claude-cli")

        self.timeout = timeout
        self.max_retries = max_retries
        self.cli_path = _resolve_cli_path(cli_path)
        self._is_js = _is_js_cli(self.cli_path)

        logger.info(f"Initialized CLIClient: {self.cli_path} " f"(js={self._is_js})")

    def _run_cli(self, prompt: str) -> str:
        """
        Execute Claude CLI and return the response.

        Args:
            prompt: The prompt to send

        Returns:
            Raw text response from CLI

        Raises:
            CLIClientError: If CLI execution fails
        """
        # Build command based on CLI type
        # Note: Prompt is passed via stdin (-p -) to avoid OS argument length limits
        if self._is_js:
            # Running a JS file directly with node
            cmd = ["node", str(self.cli_path), "--print", "-p", "-"]
        else:
            # Running the claude command
            cmd = [str(self.cli_path), "--print", "-p", "-"]

        # Strip API keys to enforce subscription-only mode
        env = os.environ.copy()
        for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"]:
            env.pop(key, None)

        last_error = None
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"CLI attempt {attempt + 1}/{self.max_retries}")

                result = subprocess.run(
                    cmd,
                    input=prompt,  # Pass via stdin to avoid argument length limits
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    env=env,
                )

                if result.returncode != 0:
                    error_msg = (
                        result.stderr.strip() or f"Exit code: {result.returncode}"
                    )
                    logger.warning(f"CLI returned error: {error_msg}")
                    logger.debug(
                        f"CLI stdout: {result.stdout[:500] if result.stdout else '(empty)'}"
                    )
                    logger.debug(
                        f"CLI stderr: {result.stderr[:500] if result.stderr else '(empty)'}"
                    )
                    last_error = CLIClientError(f"CLI error: {error_msg}")
                    continue

                response = result.stdout.strip()
                if not response:
                    logger.warning("CLI returned empty response")
                    last_error = CLIClientError("Empty response from CLI")
                    continue

                return response

            except subprocess.TimeoutExpired:
                logger.warning(f"CLI timed out after {self.timeout}s")
                last_error = CLIClientError(f"CLI timed out after {self.timeout}s")
            except Exception as e:
                logger.warning(f"CLI execution error: {e}")
                last_error = CLIClientError(f"Execution error: {e}")

        raise last_error or CLIClientError("CLI failed after max retries")

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate completion using Claude CLI.

        When using a patched CLI (default), the system prompt is replaced with
        a minimal ACE-focused prompt. When using the system claude binary,
        the 'system' parameter is prepended to the user prompt as a workaround.

        Args:
            prompt: Input prompt text
            system: System prompt (prepended if using unpatched CLI)
            **kwargs: Ignored (CLI doesn't support additional params)

        Returns:
            LLMResponse containing the generated text
        """
        # Combine system prompt into user prompt if provided
        # This is needed only when using the system CLI binary
        full_prompt = prompt
        if system and not self._is_js:
            full_prompt = f"{system}\n\n{prompt}"

        response_text = self._run_cli(full_prompt)

        return LLMResponse(
            text=response_text,
            raw={"cli_path": str(self.cli_path), "provider": "claude-cli"},
        )

    def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> T:
        """
        Completion with structured output - parse JSON and validate with Pydantic.

        The prompt should instruct the model to output valid JSON matching
        the response_model schema.

        Args:
            prompt: User prompt (should request JSON output)
            response_model: Pydantic model class to validate against
            system: Ignored (Claude Code has its own system prompt)
            **kwargs: Additional parameters (ignored)

        Returns:
            Instance of response_model with validated data

        Raises:
            json.JSONDecodeError: If response is not valid JSON
            ValidationError: If JSON doesn't match response_model schema
        """
        # Add JSON instruction to prompt
        json_prompt = self._add_json_instruction(prompt, response_model)

        # Get response
        response = self.complete(json_prompt, system=system, **kwargs)
        response_text = response.text

        # Extract JSON from response
        json_str = self._extract_json(response_text)

        # Parse and validate
        try:
            data = json.loads(json_str)
            return response_model.model_validate(data)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}...")
            raise

    def _add_json_instruction(
        self,
        prompt: str,
        response_model: Type[BaseModel],
    ) -> str:
        """Add JSON formatting instruction to prompt."""
        # Get schema for the model
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        instruction = f"""
IMPORTANT: You must respond with ONLY valid JSON matching this schema:
{schema_str}

Do not include any text before or after the JSON. Output ONLY the JSON object.

---

{prompt}
"""
        return instruction.strip()

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON object from response text.

        Handles cases where the model includes extra text around the JSON.

        Args:
            text: Raw response text

        Returns:
            Extracted JSON string
        """
        text = text.strip()

        # If it already looks like JSON, return as-is
        if text.startswith("{") and text.endswith("}"):
            return text

        # Try to find JSON object in the text
        # Look for outermost { }
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
            # Return from start to end of string
            return text[start:]

        return text[start : end + 1]
