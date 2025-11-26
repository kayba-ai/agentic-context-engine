"""Production LLM client implementations for ACE."""

from typing import Optional
from .litellm_client import LiteLLMClient, LiteLLMConfig

LangChainLiteLLMClient: Optional[type]

try:
    from .langchain_client import LangChainLiteLLMClient as _LangChainLiteLLMClient

    LangChainLiteLLMClient = _LangChainLiteLLMClient  # type: ignore[assignment]
except ImportError:
    LangChainLiteLLMClient = None  # Optional dependency  # type: ignore[assignment]

__all__ = [
    "LiteLLMClient",
    "LiteLLMConfig",
    "LangChainLiteLLMClient",
]
