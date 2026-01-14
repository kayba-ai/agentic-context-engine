"""Production LLM client implementations for ACE."""

from typing import Optional
from .litellm_client import LiteLLMClient, LiteLLMConfig

LangChainLiteLLMClient: Optional[type]
ClaudeCodeLLMClient: Optional[type]
ClaudeCodeLLMConfig: Optional[type]
CLAUDE_CODE_CLI_AVAILABLE: bool = False

try:
    from .langchain_client import LangChainLiteLLMClient as _LangChainLiteLLMClient

    LangChainLiteLLMClient = _LangChainLiteLLMClient  # type: ignore[assignment]
except ImportError:
    LangChainLiteLLMClient = None  # Optional dependency  # type: ignore[assignment]

# Claude Code CLI client (uses subscription auth instead of API keys)
try:
    from .claude_code_client import (
        ClaudeCodeLLMClient as _ClaudeCodeLLMClient,
        ClaudeCodeLLMConfig as _ClaudeCodeLLMConfig,
        CLAUDE_CODE_CLI_AVAILABLE as _CLAUDE_CODE_CLI_AVAILABLE,
    )

    ClaudeCodeLLMClient = _ClaudeCodeLLMClient  # type: ignore[assignment]
    ClaudeCodeLLMConfig = _ClaudeCodeLLMConfig  # type: ignore[assignment]
    CLAUDE_CODE_CLI_AVAILABLE = _CLAUDE_CODE_CLI_AVAILABLE
except ImportError:
    ClaudeCodeLLMClient = None  # type: ignore[assignment]
    ClaudeCodeLLMConfig = None  # type: ignore[assignment]
    CLAUDE_CODE_CLI_AVAILABLE = False

__all__ = [
    "LiteLLMClient",
    "LiteLLMConfig",
    "LangChainLiteLLMClient",
    "ClaudeCodeLLMClient",
    "ClaudeCodeLLMConfig",
    "CLAUDE_CODE_CLI_AVAILABLE",
]
