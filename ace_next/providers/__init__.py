"""ACE LLM providers — client wrappers for LLM APIs.

- ``LiteLLMClient`` / ``LiteLLMConfig`` — LiteLLM integration (100+ providers)
- ``InstructorClient`` / ``wrap_with_instructor`` — Instructor structured outputs
- ``LangChainLiteLLMClient`` — LangChain + LiteLLM (optional)
- ``ClaudeCodeLLMClient`` / ``ClaudeCodeLLMConfig`` — Claude Code CLI (optional)

Heavy dependencies (litellm, instructor, openai) are lazily imported so that
lightweight consumers (e.g. the CLI) don't pay the startup cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Config is lightweight — always available eagerly.
from .config import ACEModelConfig, ModelConfig, load_config, save_config

if TYPE_CHECKING:
    from .instructor import InstructorClient, wrap_with_instructor
    from .litellm import LiteLLMClient, LiteLLMConfig, LLMResponse
    from .registry import ValidationResult, search_models, validate_connection

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # LiteLLM
    "LiteLLMClient": ("ace_next.providers.litellm", "LiteLLMClient"),
    "LiteLLMConfig": ("ace_next.providers.litellm", "LiteLLMConfig"),
    "LLMResponse": ("ace_next.providers.litellm", "LLMResponse"),
    # Instructor
    "InstructorClient": ("ace_next.providers.instructor", "InstructorClient"),
    "wrap_with_instructor": ("ace_next.providers.instructor", "wrap_with_instructor"),
    # Registry
    "ValidationResult": ("ace_next.providers.registry", "ValidationResult"),
    "validate_connection": ("ace_next.providers.registry", "validate_connection"),
    "search_models": ("ace_next.providers.registry", "search_models"),
    # Optional: LangChain
    "LangChainLiteLLMClient": ("ace_next.providers.langchain", "LangChainLiteLLMClient"),
    # Optional: Claude Code
    "ClaudeCodeLLMClient": ("ace_next.providers.claude_code", "ClaudeCodeLLMClient"),
    "ClaudeCodeLLMConfig": ("ace_next.providers.claude_code", "ClaudeCodeLLMConfig"),
    "CLAUDE_CODE_CLI_AVAILABLE": ("ace_next.providers.claude_code", "CLAUDE_CODE_CLI_AVAILABLE"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        try:
            module = importlib.import_module(module_path)
        except ImportError:
            # Optional providers (langchain, claude_code) may not be installed.
            return None
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'ace_next.providers' has no attribute {name!r}")


__all__ = [
    # Config
    "ModelConfig",
    "ACEModelConfig",
    "load_config",
    "save_config",
    # Registry
    "ValidationResult",
    "validate_connection",
    "search_models",
    # LiteLLM
    "LiteLLMClient",
    "LiteLLMConfig",
    "LLMResponse",
    # Instructor
    "InstructorClient",
    "wrap_with_instructor",
    # LangChain (optional)
    "LangChainLiteLLMClient",
    # Claude Code CLI (optional)
    "ClaudeCodeLLMClient",
    "ClaudeCodeLLMConfig",
    "CLAUDE_CODE_CLI_AVAILABLE",
]
