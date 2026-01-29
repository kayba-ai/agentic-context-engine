"""Claude Code CLI client for ACE learning - uses subscription auth instead of API keys."""

from __future__ import annotations

import subprocess
import os
import json
import re
import shutil
import logging
from typing import Any, ClassVar, Optional, Protocol, Type, TypeVar, cast
from dataclasses import dataclass

from ..llm import LLMClient, LLMResponse


class PydanticModelProtocol(Protocol):
    """Protocol for Pydantic v2 model class methods used in structured output."""

    model_fields: ClassVar[dict[str, Any]]

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]: ...

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "PydanticModelProtocol": ...


# Type variable for Pydantic models
T = TypeVar("T", bound=PydanticModelProtocol)

logger = logging.getLogger(__name__)


# Check if claude CLI is available (handle Windows .cmd extension)
def _find_claude_cli() -> Optional[str]:
    """Find the claude CLI executable, handling Windows .cmd files."""
    # Try direct name first
    claude_path = shutil.which("claude")
    if claude_path:
        return claude_path

    # On Windows, try with .cmd extension
    if os.name == "nt":
        claude_path = shutil.which("claude.cmd")
        if claude_path:
            return claude_path

    return None


_CLAUDE_CLI_PATH = _find_claude_cli()
CLAUDE_CODE_CLI_AVAILABLE = _CLAUDE_CLI_PATH is not None


@dataclass
class ClaudeCodeLLMConfig:
    """Configuration for Claude Code CLI LLM client."""

    model: str = "claude-opus-4-5-20251101"  # Model selection hint (used in prompts)
    timeout: int = 300  # Timeout in seconds for CLI call
    max_tokens: int = 4096  # Max tokens to request
    working_dir: Optional[str] = None  # Working directory for claude CLI
    verbose: bool = False  # Enable verbose output

    # Compatibility with InstructorClient expectations (these are ignored by CLI)
    temperature: float = 0.0  # Ignored - CLI uses default
    top_p: Optional[float] = None  # Ignored - CLI uses default
    api_key: Optional[str] = None  # Ignored - uses subscription auth
    api_base: Optional[str] = None  # Ignored - uses CLI
    extra_headers: Optional[dict] = None  # Ignored - uses CLI
    ssl_verify: Optional[bool] = None  # Ignored - uses CLI


class ClaudeCodeLLMClient(LLMClient):
    """
    LLM client that uses Claude Code CLI for completions.

    This client uses the user's Claude Code subscription authentication
    instead of requiring ANTHROPIC_API_KEY or OPENAI_API_KEY.

    Key features:
    - Uses 'claude' CLI with --print flag for non-interactive operation
    - Filters out ANTHROPIC_API_KEY to force subscription auth
    - Suitable for ACE learning (Reflector/SkillManager) without API keys

    Example:
        >>> client = ClaudeCodeLLMClient()
        >>> response = client.complete("Analyze this session and suggest improvements")
        >>> print(response.text)

        >>> # With config
        >>> config = ClaudeCodeLLMConfig(timeout=600, working_dir="./project")
        >>> client = ClaudeCodeLLMClient(config=config)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        timeout: int = 300,
        max_tokens: int = 4096,
        working_dir: Optional[str] = None,
        config: Optional[ClaudeCodeLLMConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Claude Code CLI client.

        Args:
            model: Model identifier (hint, actual model depends on subscription)
            timeout: Timeout in seconds for CLI call
            max_tokens: Maximum tokens to generate
            working_dir: Working directory for claude CLI
            config: Complete configuration object (overrides other params)
            **kwargs: Additional parameters (ignored for CLI compatibility)
        """
        if not CLAUDE_CODE_CLI_AVAILABLE:
            raise RuntimeError(
                "Claude Code CLI not found. Install from: https://claude.ai/code\n"
                "Or ensure 'claude' is in your PATH."
            )

        # Use provided config or create from parameters
        if config:
            self.config = config
        else:
            self.config = ClaudeCodeLLMConfig(
                model=model or "claude-opus-4-5-20251101",
                timeout=timeout,
                max_tokens=max_tokens,
                working_dir=working_dir,
            )

        super().__init__(model=self.config.model)

        logger.info(
            f"ClaudeCodeLLMClient initialized (timeout={self.config.timeout}s, "
            f"working_dir={self.config.working_dir or 'current'})"
        )

    def complete(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        """
        Generate completion using Claude Code CLI.

        Args:
            prompt: Input prompt text
            system: Optional system message (prepended to prompt)
            **kwargs: Additional parameters (mostly ignored for CLI)

        Returns:
            LLMResponse containing the generated text and metadata
        """
        # Build full prompt with system message if provided
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"

        # Prepare environment - filter out ANTHROPIC_API_KEY to force subscription auth
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

        # Build command - use full path to handle Windows .cmd files
        if _CLAUDE_CLI_PATH is None:
            return LLMResponse(
                text="Error: Claude CLI not found in PATH",
                raw={"error": True, "not_found": True},
            )

        cmd = [
            _CLAUDE_CLI_PATH,
            "--print",  # Non-interactive, print output
            "--output-format",
            "text",  # Plain text output (split for Windows compatibility)
        ]

        # Determine working directory
        cwd = self.config.working_dir or os.getcwd()

        try:
            # On Windows, .cmd files need shell=True
            use_shell = os.name == "nt" and _CLAUDE_CLI_PATH.endswith(".cmd")

            # Set UTF-8 encoding for Windows to handle Unicode
            if os.name == "nt":
                env = env.copy() if env else os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"

            result = subprocess.run(
                cmd,
                input=full_prompt,
                text=True,
                capture_output=True,
                timeout=self.config.timeout,
                cwd=cwd,
                env=env,
                shell=use_shell,
                encoding="utf-8",
                errors="replace",  # Replace undecodable chars instead of failing
            )

            if result.returncode != 0:
                error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                logger.error(
                    f"Claude CLI failed (code {result.returncode}): {error_msg}"
                )
                return LLMResponse(
                    text=f"Error: Claude CLI failed with code {result.returncode}",
                    raw={
                        "error": True,
                        "returncode": result.returncode,
                        "stderr": error_msg,
                    },
                )

            # Extract output text
            output_text = result.stdout.strip()

            return LLMResponse(
                text=output_text,
                raw={
                    "model": "claude-code-cli",
                    "provider": "claude-code-subscription",
                    "returncode": result.returncode,
                },
            )

        except subprocess.TimeoutExpired:
            logger.error(f"Claude CLI timed out after {self.config.timeout}s")
            return LLMResponse(
                text=f"Error: Claude CLI timed out after {self.config.timeout}s",
                raw={"error": True, "timeout": True},
            )
        except Exception as e:
            logger.error(f"Claude CLI error: {e}")
            return LLMResponse(
                text=f"Error: {e}",
                raw={"error": True, "exception": str(e)},
            )

    def complete_json(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        """
        Generate completion expecting JSON output.

        Adds JSON formatting instructions to the prompt.

        Args:
            prompt: Input prompt text
            system: Optional system message
            **kwargs: Additional parameters

        Returns:
            LLMResponse with JSON text in the response
        """
        json_prompt = f"""{prompt}

IMPORTANT: Respond with valid JSON only. No markdown code blocks, no explanation text.
Just the raw JSON object."""

        return self.complete(json_prompt, system=system, **kwargs)

    def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> T:
        """
        Generate completion and parse into a Pydantic model.

        This method provides Instructor-like functionality using Claude Code CLI.
        It includes the JSON schema in the prompt and validates the response.

        Args:
            prompt: Input prompt text
            response_model: Pydantic model class to parse response into
            system: Optional system message
            max_retries: Number of retries on parse failure
            **kwargs: Additional parameters

        Returns:
            Instance of response_model populated with the response

        Raises:
            ValueError: If response cannot be parsed after retries
        """
        # Get JSON schema from Pydantic model
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        # Build structured prompt with schema
        structured_prompt = f"""{prompt}

## Required Output Format

You must respond with a JSON object matching this exact schema:

```json
{schema_str}
```

CRITICAL INSTRUCTIONS:
1. Output ONLY valid JSON - no markdown, no explanation, no extra text
2. Follow the schema exactly - all required fields must be present
3. Use the correct data types as specified in the schema
4. Start your response with {{ and end with }}"""

        last_error = None
        for attempt in range(max_retries):
            response = self.complete(structured_prompt, system=system, **kwargs)

            # Check for CLI errors
            if response.raw and response.raw.get("error"):
                last_error = f"CLI error: {response.text}"
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: {last_error}")
                continue

            # Try to extract and parse JSON
            try:
                json_text = self._extract_json(response.text)
                parsed_data = json.loads(json_text)
                result = cast(T, response_model.model_validate(parsed_data))
                logger.debug(
                    f"Successfully parsed {response_model.__name__} on attempt {attempt + 1}"
                )
                return result

            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: {last_error}")
                # Add hint about the error for retry
                structured_prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {last_error}. Please output valid JSON only."

            except Exception as e:
                last_error = f"Validation error: {e}"
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: {last_error}")
                structured_prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {last_error}. Please follow the schema exactly."

        raise ValueError(
            f"Failed to get valid {response_model.__name__} after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _extract_json(self, text: str) -> str:
        """
        Extract JSON from response text, handling markdown code blocks.

        Args:
            text: Raw response text

        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks if present
        text = text.strip()

        # Try to extract from ```json ... ``` blocks
        json_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_block_match:
            text = json_block_match.group(1).strip()

        # If text starts with { or [, assume it's JSON
        if text.startswith("{") or text.startswith("["):
            # Find the matching closing bracket
            bracket_count = 0
            in_string = False
            escape_next = False
            end_pos = 0

            for i, char in enumerate(text):
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
                if char in "{[":
                    bracket_count += 1
                elif char in "}]":
                    bracket_count -= 1
                    if bracket_count == 0:
                        end_pos = i + 1
                        break

            if end_pos > 0:
                text = text[:end_pos]

        return text.strip()


def is_claude_code_cli_available() -> bool:
    """Check if Claude Code CLI is available."""
    return CLAUDE_CODE_CLI_AVAILABLE


__all__ = [
    "ClaudeCodeLLMClient",
    "ClaudeCodeLLMConfig",
    "CLAUDE_CODE_CLI_AVAILABLE",
    "is_claude_code_cli_available",
]
