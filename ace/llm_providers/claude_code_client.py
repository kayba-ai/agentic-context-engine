"""Claude Code CLI client for ACE learning - uses subscription auth instead of API keys."""

from __future__ import annotations

import subprocess
import os
import json
import shutil
import logging
from typing import Any, Optional
from dataclasses import dataclass

from ..llm import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

# Check if claude CLI is available
CLAUDE_CODE_CLI_AVAILABLE = shutil.which("claude") is not None


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

        # Build command
        cmd = [
            "claude",
            "--print",  # Non-interactive, print output
            "--output-format=text",  # Plain text output (simpler than JSON for parsing)
        ]

        # Determine working directory
        cwd = self.config.working_dir or os.getcwd()

        try:
            result = subprocess.run(
                cmd,
                input=full_prompt,
                text=True,
                capture_output=True,
                timeout=self.config.timeout,
                cwd=cwd,
                env=env,
            )

            if result.returncode != 0:
                error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                logger.error(f"Claude CLI failed (code {result.returncode}): {error_msg}")
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


def is_claude_code_cli_available() -> bool:
    """Check if Claude Code CLI is available."""
    return CLAUDE_CODE_CLI_AVAILABLE


__all__ = [
    "ClaudeCodeLLMClient",
    "ClaudeCodeLLMConfig",
    "CLAUDE_CODE_CLI_AVAILABLE",
    "is_claude_code_cli_available",
]
