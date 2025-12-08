"""Instructor-enhanced LLM client for robust structured outputs with automatic validation and retry."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

from ..llm import LLMClient, LLMResponse
from .litellm_client import LiteLLMClient

logger = logging.getLogger(__name__)

# Instructor is a core dependency - always available
import instructor
from litellm import completion

T = TypeVar("T", bound=BaseModel)


class InstructorClient:
    """
    Wrapper around LiteLLM that adds Instructor's structured output capabilities.

    Provides automatic validation, intelligent retry on validation failures,
    and feeds validation errors back to the LLM for correction.

    Args:
        llm: Base LLMClient to wrap (typically LiteLLMClient)
        mode: Instructor mode - 'json' (default), 'md_json', 'tools', etc.
        max_retries: Maximum validation retries (default: 3)

    Example:
        >>> from ace.llm_providers.litellm_client import LiteLLMClient
        >>> from ace.llm_providers.instructor_client import InstructorClient
        >>>
        >>> base_llm = LiteLLMClient(model="gpt-4")
        >>> instructor_llm = InstructorClient(base_llm)
        >>>
        >>> # Use with Agent/Reflector/SkillManager
        >>> from ace import Agent, AgentOutput
        >>> agent = Agent(instructor_llm)

    Features:
        - Automatic Pydantic validation
        - Intelligent retry with error feedback
        - Works with any LiteLLM-compatible model
    """

    def __init__(
        self,
        llm: LLMClient,
        mode=instructor.Mode.MD_JSON,
        max_retries: int = 3,
    ):
        """
        Initialize Instructor client wrapper.

        Args:
            llm: Base LLM client (must be LiteLLMClient or compatible)
            mode: Instructor mode for structured output (default: instructor.Mode.MD_JSON)
                   MD_JSON works with all models including local ones; use Mode.JSON for OpenAI structured outputs
            max_retries: Maximum validation retries
        """

        self.llm = llm
        self.mode = mode
        self.max_retries = max_retries

        # Patch LiteLLM completion function with Instructor
        self.client = instructor.from_litellm(completion, mode=mode)
        logger.info(
            f"Initialized InstructorClient with mode={mode}, max_retries={max_retries}"
        )

    def complete(
        self,
        prompt: str,
        response_model: Optional[Type[T]] = None,
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Standard completion that returns text (for backward compatibility).

        Args:
            prompt: User prompt
            response_model: Optional Pydantic model for structured output
            system: Optional system prompt
            **kwargs: Additional LLM parameters

        Returns:
            LLMResponse with text content
        """
        if response_model is not None:
            # Structured output requested - use complete_structured
            structured_output = self.complete_structured(
                prompt=prompt, response_model=response_model, system=system, **kwargs
            )
            # Convert to LLMResponse
            return LLMResponse(
                text=structured_output.model_dump_json(indent=2),
                raw={"structured_output": structured_output.model_dump()},
            )
        else:
            # Fall back to base LLM for unstructured output
            return self.llm.complete(prompt, system=system, **kwargs)

    def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system: Optional[str] = None,
        **kwargs: Any,
    ) -> T:
        """
        Completion with structured output validation using Instructor.

        This is the core method that leverages Instructor's capabilities:
        - Automatic Pydantic validation
        - Intelligent retry on validation failures
        - Feeds validation errors back to LLM

        Args:
            prompt: User prompt
            response_model: Pydantic model class to validate against
            system: Optional system prompt
            **kwargs: Additional LLM parameters (temperature, max_tokens, etc.)

        Returns:
            Instance of response_model with validated data

        Raises:
            ValidationError: If validation fails after max_retries

        Example:
            >>> from ace.roles import AgentOutput
            >>> output = instructor_llm.complete_structured(
            ...     prompt="What is 2+2?",
            ...     response_model=AgentOutput
            ... )
            >>> print(output.final_answer)
            4
        """
        # Extract LiteLLM-compatible parameters from our LLMClient
        # Note: This assumes we're wrapping a LiteLLMClient
        model = getattr(self.llm, "config", None)
        if model and hasattr(model, "model"):
            model_name = model.model
        else:
            model_name = kwargs.pop("model", "gpt-3.5-turbo")

        # Build messages
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Merge parameters
        call_params = {
            "model": model_name,
            "messages": messages,
            "response_model": response_model,
            "max_retries": self.max_retries,
        }

        # Add LLM parameters
        if hasattr(self.llm, "config"):
            config = self.llm.config
            call_params.update(
                {
                    "temperature": kwargs.get("temperature", config.temperature),
                    "max_tokens": kwargs.get("max_tokens", config.max_tokens),
                }
            )
            # Only add top_p if explicitly set (not None)
            top_p_value = kwargs.get("top_p", config.top_p)
            if top_p_value is not None:
                call_params["top_p"] = top_p_value

            # Forward authentication and HTTP settings (Issue #44)
            if config.api_key:
                call_params["api_key"] = config.api_key
            if config.api_base:
                call_params["api_base"] = config.api_base
            if config.extra_headers:
                call_params["extra_headers"] = config.extra_headers
            if config.ssl_verify is not None:
                call_params["ssl_verify"] = config.ssl_verify

            # Apply Claude parameter resolution to avoid temperature + top_p conflict
            sampling_priority = getattr(config, "sampling_priority", "temperature")
            call_params = LiteLLMClient._resolve_sampling_params(
                call_params, model_name, sampling_priority
            )

        # Add any additional kwargs
        call_params.update({k: v for k, v in kwargs.items() if k not in call_params})

        try:
            # Use Instructor's patched completion
            response = self.client.chat.completions.create(**call_params)
            logger.debug(
                f"Instructor validation successful for {response_model.__name__}"
            )
            return response

        except Exception as e:
            logger.error(f"Instructor validation failed: {e}")
            raise


def wrap_with_instructor(
    llm: LLMClient, mode=instructor.Mode.MD_JSON, max_retries: int = 3
) -> InstructorClient:
    """
    Convenience function to wrap an LLM client with Instructor capabilities.

    Args:
        llm: Base LLM client to wrap
        mode: Instructor mode (default: instructor.Mode.MD_JSON for broad compatibility)
        max_retries: Maximum validation retries

    Returns:
        InstructorClient with automatic Pydantic validation

    Example:
        >>> from ace.llm_providers.litellm_client import LiteLLMClient
        >>> from ace.llm_providers.instructor_client import wrap_with_instructor
        >>>
        >>> base_llm = LiteLLMClient(model="gpt-4")
        >>> llm = wrap_with_instructor(base_llm)
        >>> agent = Agent(llm)  # Auto-validates
    """
    return InstructorClient(llm, mode=mode, max_retries=max_retries)
