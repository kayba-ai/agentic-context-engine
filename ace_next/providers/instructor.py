"""Instructor-enhanced LLM client for robust structured outputs.

Self-contained integration — no imports from ``ace/``.  Wraps a
``LiteLLMClient`` (or any compatible object) with Instructor's automatic
Pydantic validation and retry capabilities.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from .litellm import LiteLLMClient, LLMResponse

logger = logging.getLogger(__name__)

from litellm import completion

try:
    import instructor

    INSTRUCTOR_AVAILABLE = True
except ImportError:
    INSTRUCTOR_AVAILABLE = False

T = TypeVar("T", bound=BaseModel)


class InstructorClient:
    """Wrapper that adds Instructor's structured-output capabilities.

    Provides automatic validation, intelligent retry on validation failures,
    and feeds validation errors back to the LLM for correction.

    Args:
        llm: Base LLM client to wrap (typically ``LiteLLMClient``).
        mode: Instructor mode (default: ``MD_JSON`` for broad compatibility).
        max_retries: Maximum validation retries.

    Example::

        base_llm = LiteLLMClient(model="gpt-4")
        llm = InstructorClient(base_llm)
        agent = Agent(llm)  # Auto-validates outputs
    """

    def __init__(
        self,
        llm: Any,
        mode: Any = None,
        max_retries: int = 3,
    ) -> None:
        if not INSTRUCTOR_AVAILABLE:
            raise ImportError(
                "instructor is required for InstructorClient. "
                "Install it with: pip install ace-framework[instructor]"
            )

        self.llm = llm
        self.mode = mode if mode is not None else instructor.Mode.MD_JSON
        self.max_retries = max_retries

        # Patch LiteLLM completion function with Instructor
        self.client = instructor.from_litellm(completion, mode=self.mode)
        logger.info(
            f"Initialized InstructorClient with mode={self.mode}, max_retries={max_retries}"
        )

    def complete(
        self,
        prompt: str,
        response_model: Optional[Type[T]] = None,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Standard completion returning text (backward compatible).

        If *response_model* is provided, delegates to
        :meth:`complete_structured` and wraps the result as an
        ``LLMResponse``.
        """
        if response_model is not None:
            structured_output = self.complete_structured(
                prompt=prompt, response_model=response_model, system=system, **kwargs
            )
            return LLMResponse(
                text=structured_output.model_dump_json(indent=2),
                raw={"structured_output": structured_output.model_dump()},
            )
        return self.llm.complete(prompt, system=system, **kwargs)

    def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> T:
        """Completion with Instructor validation and retry.

        Core method: validates the LLM response against *response_model*
        (a Pydantic ``BaseModel`` subclass) and retries with error feedback
        on validation failure.
        """
        # Extract model name from wrapped client
        model_config = getattr(self.llm, "config", None)
        if model_config and hasattr(model_config, "model"):
            model_name = model_config.model
        else:
            model_name = kwargs.pop("model", "gpt-3.5-turbo")

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        call_params: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "response_model": response_model,
            "max_retries": self.max_retries,
        }

        # Forward LLM config parameters
        if model_config:
            call_params.update(
                {
                    "temperature": kwargs.get("temperature", model_config.temperature),
                    "max_tokens": kwargs.get("max_tokens", model_config.max_tokens),
                }
            )
            top_p_value = kwargs.get("top_p", model_config.top_p)
            if top_p_value is not None:
                call_params["top_p"] = top_p_value

            if model_config.api_key:
                call_params["api_key"] = model_config.api_key
            if model_config.api_base:
                call_params["api_base"] = model_config.api_base
            if model_config.extra_headers:
                call_params["extra_headers"] = model_config.extra_headers
            if model_config.ssl_verify is not None:
                call_params["ssl_verify"] = model_config.ssl_verify

            # Claude parameter resolution
            sampling_priority = getattr(
                model_config, "sampling_priority", "temperature"
            )
            call_params = LiteLLMClient._resolve_sampling_params(
                call_params, model_name, sampling_priority
            )

        # Forward remaining kwargs
        call_params.update({k: v for k, v in kwargs.items() if k not in call_params})

        try:
            response = self.client.chat.completions.create(**call_params)
            logger.debug(
                f"Instructor validation successful for {response_model.__name__}"
            )
            return response
        except Exception as e:
            logger.error(f"Instructor validation failed: {e}")
            raise


def wrap_with_instructor(
    llm: Any,
    mode: Any = None,
    max_retries: int = 3,
) -> InstructorClient:
    """Convenience wrapper to add Instructor capabilities to an LLM client.

    Example::

        from ace_next import LiteLLMClient, wrap_with_instructor, Agent

        base_llm = LiteLLMClient(model="gpt-4")
        llm = wrap_with_instructor(base_llm)
        agent = Agent(llm)  # Auto-validates
    """
    return InstructorClient(llm, mode=mode, max_retries=max_retries)
