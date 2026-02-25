"""Claude Code CLI provider for ACE next.

Self-contained — no imports from ``ace/``.  Uses the user's Claude Code
subscription authentication (no API key required).  Satisfies
``LLMClientLike`` via ``complete()`` and ``complete_structured()``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, ClassVar, Optional, Protocol, Type, TypeVar, cast

from .litellm import LLMResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic model protocol (for complete_structured)
# ---------------------------------------------------------------------------


class _PydanticModelProtocol(Protocol):
    """Minimal interface for Pydantic v2 model classes."""

    model_fields: ClassVar[dict[str, Any]]

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]: ...

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> "_PydanticModelProtocol": ...


T = TypeVar("T", bound=_PydanticModelProtocol)


# ---------------------------------------------------------------------------
# CLI discovery
# ---------------------------------------------------------------------------


def _find_claude_cli() -> Optional[str]:
    """Find the ``claude`` CLI executable, handling Windows ``.cmd`` files."""
    path = shutil.which("claude")
    if path:
        return path
    if os.name == "nt":
        path = shutil.which("claude.cmd")
        if path:
            return path
    return None


_CLAUDE_CLI_PATH = _find_claude_cli()
CLAUDE_CODE_CLI_AVAILABLE = _CLAUDE_CLI_PATH is not None


# ---------------------------------------------------------------------------
# ClaudeCodeLLMConfig
# ---------------------------------------------------------------------------


@dataclass
class ClaudeCodeLLMConfig:
    """Configuration for the Claude Code CLI LLM client."""

    model: str = "claude-opus-4-5-20251101"
    timeout: int = 300
    max_tokens: int = 4096
    working_dir: Optional[str] = None
    verbose: bool = False

    # Compatibility with InstructorClient expectations (ignored by CLI)
    temperature: float = 0.0
    top_p: Optional[float] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    extra_headers: Optional[dict] = None
    ssl_verify: Optional[bool] = None


# ---------------------------------------------------------------------------
# ClaudeCodeLLMClient
# ---------------------------------------------------------------------------


class ClaudeCodeLLMClient:
    """LLM client that uses the Claude Code CLI for completions.

    Uses the user's Claude Code subscription authentication instead of
    requiring ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY``.

    Args:
        model: Model identifier (hint — actual model depends on subscription).
        timeout: Timeout in seconds for the CLI call.
        max_tokens: Maximum tokens to generate.
        working_dir: Working directory for the ``claude`` CLI.
        config: Complete configuration (overrides other params).

    Example::

        client = ClaudeCodeLLMClient()
        response = client.complete("Analyze this code")
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
        if not CLAUDE_CODE_CLI_AVAILABLE:
            raise RuntimeError(
                "Claude Code CLI not found. Install from: https://claude.ai/code\n"
                "Or ensure 'claude' is in your PATH."
            )

        self.config = config or ClaudeCodeLLMConfig(
            model=model or "claude-opus-4-5-20251101",
            timeout=timeout,
            max_tokens=max_tokens,
            working_dir=working_dir,
        )

        logger.info(
            "ClaudeCodeLLMClient initialized (timeout=%ds, working_dir=%s)",
            self.config.timeout,
            self.config.working_dir or "current",
        )

    # -- core completion -------------------------------------------------------

    def complete(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        """Generate a completion using the Claude Code CLI."""
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        # Filter out ANTHROPIC_API_KEY to force subscription auth
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

        if _CLAUDE_CLI_PATH is None:
            return LLMResponse(
                text="Error: Claude CLI not found in PATH",
                raw={"error": True, "not_found": True},
            )

        cmd = [_CLAUDE_CLI_PATH, "--print", "--output-format", "text"]
        cwd = self.config.working_dir or os.getcwd()

        try:
            use_shell = os.name == "nt" and _CLAUDE_CLI_PATH.endswith(".cmd")
            if os.name == "nt":
                env = env.copy()
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
                errors="replace",
            )

            if result.returncode != 0:
                error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                logger.error("Claude CLI failed (code %d): %s", result.returncode, error_msg)
                return LLMResponse(
                    text=f"Error: Claude CLI failed with code {result.returncode}",
                    raw={"error": True, "returncode": result.returncode, "stderr": error_msg},
                )

            return LLMResponse(
                text=result.stdout.strip(),
                raw={
                    "model": "claude-code-cli",
                    "provider": "claude-code-subscription",
                    "returncode": result.returncode,
                },
            )

        except subprocess.TimeoutExpired:
            logger.error("Claude CLI timed out after %ds", self.config.timeout)
            return LLMResponse(
                text=f"Error: Claude CLI timed out after {self.config.timeout}s",
                raw={"error": True, "timeout": True},
            )
        except Exception as e:
            logger.error("Claude CLI error: %s", e)
            return LLMResponse(text=f"Error: {e}", raw={"error": True, "exception": str(e)})

    # -- structured output -----------------------------------------------------

    def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system: Optional[str] = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> T:
        """Generate a completion and parse into a Pydantic model.

        Includes the JSON schema in the prompt and validates the response,
        retrying with error feedback on failure.
        """
        schema_str = json.dumps(response_model.model_json_schema(), indent=2)

        structured_prompt = (
            f"{prompt}\n\n"
            f"## Required Output Format\n\n"
            f"You must respond with a JSON object matching this exact schema:\n\n"
            f"```json\n{schema_str}\n```\n\n"
            f"CRITICAL INSTRUCTIONS:\n"
            f"1. Output ONLY valid JSON - no markdown, no explanation, no extra text\n"
            f"2. Follow the schema exactly - all required fields must be present\n"
            f"3. Use the correct data types as specified in the schema\n"
            f"4. Start your response with {{ and end with }}"
        )

        last_error: Optional[str] = None
        for attempt in range(max_retries):
            response = self.complete(structured_prompt, system=system, **kwargs)

            if response.raw and response.raw.get("error"):
                last_error = f"CLI error: {response.text}"
                logger.warning("Attempt %d/%d: %s", attempt + 1, max_retries, last_error)
                continue

            try:
                json_text = self._extract_json(response.text)
                parsed = json.loads(json_text)
                result = cast(T, response_model.model_validate(parsed))
                logger.debug(
                    "Successfully parsed %s on attempt %d",
                    response_model.__name__,
                    attempt + 1,
                )
                return result
            except json.JSONDecodeError as e:
                last_error = f"JSON parse error: {e}"
                logger.warning("Attempt %d/%d: %s", attempt + 1, max_retries, last_error)
                structured_prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {last_error}. Please output valid JSON only."
            except Exception as e:
                last_error = f"Validation error: {e}"
                logger.warning("Attempt %d/%d: %s", attempt + 1, max_retries, last_error)
                structured_prompt += f"\n\nPREVIOUS ATTEMPT FAILED: {last_error}. Please follow the schema exactly."

        raise ValueError(
            f"Failed to get valid {response_model.__name__} after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from response text, handling markdown code blocks."""
        text = text.strip()

        json_block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_block:
            text = json_block.group(1).strip()

        if text.startswith(("{", "[")):
            bracket_count = 0
            in_string = False
            escape_next = False
            end_pos = 0

            for i, ch in enumerate(text):
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\":
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch in "{[":
                    bracket_count += 1
                elif ch in "}]":
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
