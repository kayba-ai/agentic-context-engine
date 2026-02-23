"""LLMClientLike — structural protocol for LLM clients."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMClientLike(Protocol):
    """Minimal interface that any LLM client must satisfy.

    Concrete implementations include ``LiteLLMClient``,
    ``DummyLLMClient``, or any object with ``complete`` and
    ``complete_structured`` methods.
    """

    def complete(self, prompt: str, **kwargs: Any) -> Any:
        """Return a text response for *prompt*."""
        ...

    def complete_structured(
        self,
        prompt: str,
        response_model: type[T],
        **kwargs: Any,
    ) -> T:
        """Return a validated Pydantic model instance for *prompt*."""
        ...
