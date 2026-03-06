"""LangChain LiteLLM provider for ACE next.

Self-contained — no imports from ``ace/``.  Wraps ``langchain-litellm``
(``ChatLiteLLM`` / ``ChatLiteLLMRouter``) to satisfy ``LLMClientLike``.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Iterator, Optional

from .litellm import LLMResponse

logger = logging.getLogger(__name__)

# langchain-litellm is optional --------------------------------------------------

try:
    from langchain_litellm import ChatLiteLLM, ChatLiteLLMRouter
    from litellm import Router as LiteLLMRouter

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatLiteLLM = None  # type: ignore[assignment,misc]
    ChatLiteLLMRouter = None  # type: ignore[assignment,misc]
    LiteLLMRouter = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# LangChainLiteLLMClient
# ---------------------------------------------------------------------------


class LangChainLiteLLMClient:
    """LangChain wrapper for LiteLLM integration.

    Provides integration with LangChain's ``ChatLiteLLM`` and
    ``ChatLiteLLMRouter``, enabling model routing, load balancing, and
    the broader LangChain ecosystem.

    Args:
        model: Model name (e.g. ``"gpt-4"``, ``"claude-3-sonnet"``).
        router: Optional ``litellm.Router`` for load balancing.
        temperature: Sampling temperature (0.0–1.0).
        max_tokens: Maximum tokens in response.
        **kwargs: Forwarded to ``ChatLiteLLM`` / ``ChatLiteLLMRouter``.

    Example::

        client = LangChainLiteLLMClient(model="gpt-4", temperature=0.0)
        response = client.complete("What is the capital of France?")
    """

    def __init__(
        self,
        model: str,
        router: Optional[Any] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-litellm is not installed. "
                "Install with: pip install langchain-litellm"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if router:
            logger.info(
                "Initializing LangChainLiteLLMClient with router for model: %s", model
            )
            self.llm = ChatLiteLLMRouter(
                router=router,
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            self.is_router = True
        else:
            logger.info("Initializing LangChainLiteLLMClient for model: %s", model)
            self.llm = ChatLiteLLM(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            self.is_router = False

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _filter_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Filter out ACE-specific parameters that shouldn't reach LangChain."""
        blocked = {"refinement_round", "max_refinement_rounds"}
        return {k: v for k, v in kwargs.items() if k not in blocked}

    def _extract_metadata(self, response: Any) -> dict[str, Any]:
        """Build metadata dict from a LangChain response message."""
        metadata: dict[str, Any] = {
            "model": response.response_metadata.get("model", self.model),
            "finish_reason": response.response_metadata.get("finish_reason"),
        }
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            metadata["usage"] = {
                "prompt_tokens": response.usage_metadata.get("input_tokens"),
                "completion_tokens": response.usage_metadata.get("output_tokens"),
                "total_tokens": response.usage_metadata.get("total_tokens"),
            }
        if self.is_router:
            metadata["router"] = True
            metadata["model_used"] = response.response_metadata.get(
                "model_name", self.model
            )
        return metadata

    # -- public API -------------------------------------------------------------

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Synchronous completion."""
        filtered = self._filter_kwargs(kwargs)
        try:
            response = self.llm.invoke(prompt, **filtered)
            return LLMResponse(
                text=response.content, raw=self._extract_metadata(response)
            )
        except Exception as e:
            logger.error("Error in LangChain completion: %s", e)
            raise

    async def acomplete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Async completion."""
        filtered = self._filter_kwargs(kwargs)
        try:
            response = await self.llm.ainvoke(prompt, **filtered)
            return LLMResponse(
                text=response.content, raw=self._extract_metadata(response)
            )
        except Exception as e:
            logger.error("Error in async LangChain completion: %s", e)
            raise

    def complete_with_stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Synchronous streaming completion."""
        filtered = self._filter_kwargs(kwargs)
        try:
            for chunk in self.llm.stream(prompt, **filtered):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error("Error in LangChain streaming: %s", e)
            raise

    async def acomplete_with_stream(
        self, prompt: str, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Async streaming completion."""
        filtered = self._filter_kwargs(kwargs)
        try:
            async for chunk in self.llm.astream(prompt, **filtered):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error("Error in async LangChain streaming: %s", e)
            raise


__all__ = [
    "LangChainLiteLLMClient",
    "LANGCHAIN_AVAILABLE",
]
