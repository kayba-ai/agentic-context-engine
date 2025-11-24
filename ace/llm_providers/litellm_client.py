"""LiteLLM client for unified access to 100+ LLM providers."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import asyncio
import logging

from ..llm import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

try:
    import litellm
    from litellm import completion, acompletion, Router

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logger.warning("LiteLLM not installed. Install with: pip install litellm")

# Try to import Opik for span association
try:
    from opik.opik_context import get_current_span_data

    OPIK_SPAN_AVAILABLE = True
except ImportError:
    get_current_span_data = None
    OPIK_SPAN_AVAILABLE = False


@dataclass
class LiteLLMConfig:
    """Configuration for LiteLLM client."""

    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 0.9
    timeout: int = 60
    max_retries: int = 3
    fallbacks: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    # Provider-specific settings
    azure_deployment: Optional[str] = None
    azure_api_key: Optional[str] = None
    azure_api_base: Optional[str] = None

    # Cost tracking
    track_cost: bool = True
    max_budget: Optional[float] = None

    # Debugging
    verbose: bool = False

    # Claude-specific parameter handling
    # Anthropic API limitation: temperature and top_p cannot both be specified
    sampling_priority: str = "temperature"  # "temperature" | "top_p" | "top_k"


class LiteLLMClient(LLMClient):
    """
    Production LLM client using LiteLLM for unified access to multiple providers.

    Supports:
    - OpenAI (GPT-3.5, GPT-4, etc.)
    - Anthropic (Claude-2, Claude-3, etc.)
    - Google (Gemini, PaLM)
    - Cohere
    - Azure OpenAI
    - AWS Bedrock
    - 100+ other providers

    Claude Parameter Handling:
    Due to Anthropic API limitations, temperature and top_p cannot both be specified
    for Claude models. This client automatically resolves parameter conflicts using
    priority-based resolution following industry best practices:

    - "temperature" (default): Temperature takes precedence, Anthropic-recommended
    - "top_p": Nucleus sampling takes precedence, for advanced scenarios
    - "top_k": Top-k sampling takes precedence, for advanced use only

    Example:
        >>> client = LiteLLMClient(model="gpt-4")
        >>> response = client.complete("What is the capital of France?")

        >>> # Using Claude with sampling priority
        >>> from ace.llm_providers.litellm_client import LiteLLMConfig
        >>> config = LiteLLMConfig(
        ...     model="claude-3-sonnet-20240229",
        ...     sampling_priority="temperature"
        ... )
        >>> client = LiteLLMClient(config=config)

        >>> # Using with fallbacks
        >>> client = LiteLLMClient(
        ...     model="gpt-4",
        ...     fallbacks=["claude-3-sonnet-20240229", "gpt-3.5-turbo"]
        ... )
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        fallbacks: Optional[List[str]] = None,
        sampling_priority: str = "temperature",
        config: Optional[LiteLLMConfig] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize LiteLLM client.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-sonnet-20240229")
            api_key: API key for the provider (can also use environment variables)
            api_base: Base URL for API (for custom endpoints)
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            fallbacks: List of fallback models if primary fails
            sampling_priority: Priority for Claude parameter resolution ("temperature", "top_p", "top_k")
            config: Complete configuration object (overrides other params)
            **kwargs: Additional provider-specific parameters
        """
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "LiteLLM is not installed. Install with: pip install litellm"
            )

        # Use provided config or create from parameters
        if config:
            self.config = config
            # Use model from config if not provided as parameter
            if model is None:
                model = config.model
        else:
            if model is None:
                raise ValueError(
                    "Either 'model' parameter or 'config' with model must be provided"
                )
            self.config = LiteLLMConfig(
                model=model,
                api_key=api_key,
                api_base=api_base,
                temperature=temperature,
                max_tokens=max_tokens,
                fallbacks=fallbacks,
                sampling_priority=sampling_priority,
                **kwargs,
            )

        super().__init__(model=model)

        # Set up API keys from environment if not provided
        self._setup_api_keys()

        # Configure LiteLLM settings
        if self.config.verbose:
            litellm.set_verbose = True

        # Set up router if fallbacks are configured
        self.router: Optional[Any] = None  # Router type from litellm
        if self.config.fallbacks:
            self._setup_router()

        # Set up Opik integration for automatic token tracking
        self._setup_opik_integration()

    def _setup_api_keys(self) -> None:
        """Set up API keys from config or environment variables."""
        if not self.config.api_key:
            # Try to get API key from environment based on model provider
            if (
                "gpt" in self.config.model.lower()
                or "openai" in self.config.model.lower()
            ):
                self.config.api_key = os.getenv("OPENAI_API_KEY")
            elif (
                "claude" in self.config.model.lower()
                or "anthropic" in self.config.model.lower()
            ):
                self.config.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif "cohere" in self.config.model.lower():
                self.config.api_key = os.getenv("COHERE_API_KEY")
            elif "gemini" in self.config.model.lower():
                self.config.api_key = os.getenv("GOOGLE_API_KEY")

    def _setup_router(self) -> None:
        """Set up router for load balancing and fallbacks."""
        model_list = []

        # Add primary model
        model_list.append(
            {
                "model_name": self.config.model,
                "litellm_params": {
                    "model": self.config.model,
                    "api_key": self.config.api_key,
                    "api_base": self.config.api_base,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
            }
        )

        # Add fallback models
        for fallback_model in self.config.fallbacks or []:
            model_list.append(
                {
                    "model_name": fallback_model,
                    "litellm_params": {
                        "model": fallback_model,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                    },
                }
            )

        fallback_list = (
            [{self.config.model: self.config.fallbacks}]
            if self.config.fallbacks
            else []
        )
        self.router = Router(
            model_list=model_list,
            fallbacks=fallback_list,
            num_retries=self.config.max_retries,
            timeout=self.config.timeout,
        )

    def _setup_opik_integration(self) -> None:
        """Set up Opik integration for automatic token and cost tracking."""
        # Check if explicitly disabled
        if os.environ.get("OPIK_DISABLED", "").lower() in ("true", "1", "yes"):
            logger.debug(
                "Opik integration disabled via OPIK_DISABLED environment variable"
            )
            return

        try:
            # Import observability module
            from ..observability import get_integration

            # Get or create global Opik integration
            opik_integration = get_integration()

            # Set up LiteLLM callback for automatic tracking
            if opik_integration.setup_litellm_callback():
                logger.debug(
                    "Opik LiteLLM integration configured for automatic token tracking"
                )
            else:
                logger.debug(
                    "Opik LiteLLM integration not available or already configured"
                )

        except ImportError:
            # Observability module not available, continue without it
            logger.debug(
                "Opik observability not available, continuing without automatic tracking"
            )
        except Exception as e:
            # Non-critical error, log but don't interrupt initialization
            logger.debug(f"Failed to setup Opik integration (non-critical): {e}")

    @staticmethod
    def _resolve_sampling_params(
        params: Dict[str, Any], model: str, sampling_priority: str = "temperature"
    ) -> Dict[str, Any]:
        """
        Single source of truth for parameter conflict resolution.

        Anthropic API limitation: temperature and top_p cannot both be specified.
        This follows industry best practices with clear priority rules.

        Priority order (configurable):
        1. temperature (default, Anthropic-recommended for most cases)
        2. top_p (nucleus sampling, advanced scenarios)
        3. top_k (advanced use only)

        Args:
            params: Parameters dictionary to resolve
            model: Model name to check if Claude
            sampling_priority: 'temperature' (default), 'top_p', or 'top_k'

        Returns:
            Clean parameters dict with conflicts resolved

        Raises:
            ValueError: If sampling_priority is invalid
        """
        # Only apply to Claude models
        if "claude" not in model.lower():
            return params

        if sampling_priority not in ["temperature", "top_p", "top_k"]:
            raise ValueError(
                f"Invalid sampling_priority: {sampling_priority}. Must be one of: temperature, top_p, top_k"
            )

        resolved = params.copy()

        # Check which sampling params are present and not None
        has_temperature = (
            "temperature" in resolved and resolved["temperature"] is not None
        )
        has_top_p = "top_p" in resolved and resolved["top_p"] is not None
        has_top_k = "top_k" in resolved and resolved["top_k"] is not None

        # Remove None parameters early
        if "temperature" in resolved and resolved["temperature"] is None:
            resolved.pop("temperature")
        if "top_p" in resolved and resolved["top_p"] is None:
            resolved.pop("top_p")
        if "top_k" in resolved and resolved["top_k"] is None:
            resolved.pop("top_k")

        # Apply priority-based resolution
        if (
            sampling_priority == "temperature"
            and has_temperature
            and resolved["temperature"] > 0
        ):
            # Non-zero temperature takes precedence - remove others
            resolved.pop("top_p", None)
            resolved.pop("top_k", None)
            if has_top_p or has_top_k:
                logger.info(
                    f"Claude model {model}: Using temperature={resolved['temperature']}, ignoring other sampling params"
                )

        elif sampling_priority == "top_p" and has_top_p:
            # top_p takes precedence - remove others
            resolved.pop("temperature", None)
            resolved.pop("top_k", None)
            if has_temperature or has_top_k:
                logger.info(
                    f"Claude model {model}: Using top_p={resolved['top_p']}, ignoring other sampling params"
                )

        elif sampling_priority == "top_k" and has_top_k:
            # top_k takes precedence - remove others
            resolved.pop("temperature", None)
            resolved.pop("top_p", None)
            if has_temperature or has_top_p:
                logger.info(
                    f"Claude model {model}: Using top_k={resolved['top_k']}, ignoring other sampling params"
                )

        else:
            # Fallback: use default priority (temperature > top_p > top_k)
            # Special case: if temperature is 0, allow top_p to take precedence
            if has_temperature and resolved["temperature"] > 0:
                # Non-zero temperature takes precedence over everything
                resolved.pop("top_p", None)
                resolved.pop("top_k", None)
            elif has_top_p:
                # top_p takes precedence over temperature=0 and top_k
                resolved.pop("temperature", None)
                resolved.pop("top_k", None)
            elif has_temperature:
                # Only temperature (including 0), remove others
                resolved.pop("top_p", None)
                resolved.pop("top_k", None)
            # If only top_k is set, keep it

        return resolved

    def complete(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        """
        Generate completion for the given prompt.

        Args:
            prompt: Input prompt text
            system: Optional system message to prepend to the conversation
            **kwargs: Additional parameters to pass to the model

        Returns:
            LLMResponse containing the generated text and metadata
        """
        # Prepare messages in chat format (most models now use this)
        messages = []

        # Add system message if provided
        if system:
            messages.append({"role": "system", "content": system})

        # Add user message
        messages.append({"role": "user", "content": prompt})

        # Merge config with runtime kwargs
        merged_params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "timeout": kwargs.get("timeout", self.config.timeout),
            "num_retries": kwargs.get("num_retries", self.config.max_retries),
            "drop_params": True,  # Automatically drop unsupported parameters
        }

        # Add optional sampling parameters if provided
        if kwargs.get("top_p") is not None or self.config.top_p is not None:
            merged_params["top_p"] = kwargs.get("top_p", self.config.top_p)
        if kwargs.get("top_k") is not None:
            merged_params["top_k"] = kwargs.get("top_k")

        # Apply single-point parameter resolution for Claude models
        call_params = self._resolve_sampling_params(
            merged_params, self.config.model, self.config.sampling_priority
        )

        # Add API credentials
        if self.config.api_key:
            call_params["api_key"] = self.config.api_key
        if self.config.api_base:
            call_params["api_base"] = self.config.api_base

        # Add Opik span association for role-level token aggregation
        if OPIK_SPAN_AVAILABLE and get_current_span_data:
            try:
                current_span_data = get_current_span_data()
                if current_span_data:
                    # Add metadata to associate this LLM call with the current Opik span
                    if "metadata" not in call_params:
                        call_params["metadata"] = {}
                    if "opik" not in call_params["metadata"]:
                        call_params["metadata"]["opik"] = {}
                    call_params["metadata"]["opik"][
                        "current_span_data"
                    ] = current_span_data
                    logger.debug(
                        f"Associated LLM call with Opik span: {current_span_data.get('name', 'unknown')}"
                    )
            except Exception as e:
                # Non-critical: continue without span association
                logger.debug(f"Failed to associate LLM call with Opik span: {e}")

        # Add remaining kwargs (excluding ACE-specific and already-handled parameters)
        ace_specific_params = {
            "refinement_round",
            "max_refinement_rounds",
            "stream_thinking",
        }
        handled_params = {
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "timeout",
            "num_retries",
        }
        call_params.update(
            {
                k: v
                for k, v in kwargs.items()
                if k not in call_params
                and k not in ace_specific_params
                and k not in handled_params
            }
        )

        try:
            # Use router if available, otherwise direct completion
            if self.router:
                response = self.router.completion(**call_params)
            else:
                response = completion(**call_params)

            # Extract text from response
            text = response.choices[0].message.content

            # Build metadata
            metadata = {
                "model": response.model,
                "usage": response.usage.model_dump() if response.usage else None,
                "cost": (
                    response._hidden_params.get("response_cost", None)
                    if hasattr(response, "_hidden_params")
                    else None
                ),
                "provider": self._get_provider_from_model(response.model),
            }

            return LLMResponse(text=text, raw=metadata)

        except Exception as e:
            logger.error(f"Error in LiteLLM completion: {e}")
            raise

    async def acomplete(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        """
        Async version of complete.

        Args:
            prompt: Input prompt text
            system: Optional system message to prepend to the conversation
            **kwargs: Additional parameters to pass to the model

        Returns:
            LLMResponse containing the generated text and metadata
        """
        # Prepare messages in chat format
        messages = []

        # Add system message if provided
        if system:
            messages.append({"role": "system", "content": system})

        # Add user message
        messages.append({"role": "user", "content": prompt})

        # Merge config with runtime kwargs
        merged_params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "timeout": kwargs.get("timeout", self.config.timeout),
            "num_retries": kwargs.get("num_retries", self.config.max_retries),
            "drop_params": True,  # Automatically drop unsupported parameters
        }

        # Add optional sampling parameters if provided
        if kwargs.get("top_p") is not None or self.config.top_p is not None:
            merged_params["top_p"] = kwargs.get("top_p", self.config.top_p)
        if kwargs.get("top_k") is not None:
            merged_params["top_k"] = kwargs.get("top_k")

        # Apply single-point parameter resolution for Claude models
        call_params = self._resolve_sampling_params(
            merged_params, self.config.model, self.config.sampling_priority
        )

        # Add API credentials
        if self.config.api_key:
            call_params["api_key"] = self.config.api_key
        if self.config.api_base:
            call_params["api_base"] = self.config.api_base

        # Add Opik span association for role-level token aggregation
        if OPIK_SPAN_AVAILABLE and get_current_span_data:
            try:
                current_span_data = get_current_span_data()
                if current_span_data:
                    # Add metadata to associate this LLM call with the current Opik span
                    if "metadata" not in call_params:
                        call_params["metadata"] = {}
                    if "opik" not in call_params["metadata"]:
                        call_params["metadata"]["opik"] = {}
                    call_params["metadata"]["opik"][
                        "current_span_data"
                    ] = current_span_data
                    logger.debug(
                        f"Associated LLM call with Opik span: {current_span_data.get('name', 'unknown')}"
                    )
            except Exception as e:
                # Non-critical: continue without span association
                logger.debug(f"Failed to associate LLM call with Opik span: {e}")

        # Add remaining kwargs (excluding ACE-specific and already-handled parameters)
        ace_specific_params = {
            "refinement_round",
            "max_refinement_rounds",
            "stream_thinking",
        }
        handled_params = {
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "timeout",
            "num_retries",
        }
        call_params.update(
            {
                k: v
                for k, v in kwargs.items()
                if k not in call_params
                and k not in ace_specific_params
                and k not in handled_params
            }
        )

        try:
            # Use router if available, otherwise direct completion
            if self.router:
                response = await self.router.acompletion(**call_params)
            else:
                response = await acompletion(**call_params)

            # Extract text from response
            text = response.choices[0].message.content

            # Build metadata
            metadata = {
                "model": response.model,
                "usage": response.usage.model_dump() if response.usage else None,
                "cost": (
                    response._hidden_params.get("response_cost", None)
                    if hasattr(response, "_hidden_params")
                    else None
                ),
                "provider": self._get_provider_from_model(response.model),
            }

            return LLMResponse(text=text, raw=metadata)

        except Exception as e:
            logger.error(f"Error in LiteLLM async completion: {e}")
            raise

    def complete_with_stream(self, prompt: str, **kwargs: Any):
        """
        Generate completion with streaming support.

        Args:
            prompt: Input prompt text
            **kwargs: Additional parameters to pass to the model

        Yields:
            Chunks of generated text
        """
        messages = [{"role": "user", "content": prompt}]

        call_params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": True,
        }

        if self.config.api_key:
            call_params["api_key"] = self.config.api_key

        try:
            response = completion(**call_params)

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error in LiteLLM streaming: {e}")
            raise

    def _get_provider_from_model(self, model: str) -> str:
        """Infer provider from model name."""
        model_lower = model.lower()

        if "gpt" in model_lower or "openai" in model_lower:
            return "openai"
        elif "claude" in model_lower or "anthropic" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower or "palm" in model_lower:
            return "google"
        elif "command" in model_lower or "cohere" in model_lower:
            return "cohere"
        elif "llama" in model_lower or "mistral" in model_lower:
            return "meta"
        else:
            return "unknown"

    @classmethod
    def list_models(cls) -> List[str]:
        """List all supported models."""
        if not LITELLM_AVAILABLE:
            return []

        # This returns a list of common models
        # In practice, LiteLLM supports 100+ models
        return [
            # OpenAI
            "gpt-4",
            "gpt-4-turbo-preview",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            # Anthropic
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2",
            # Google
            "gemini-pro",
            "gemini-pro-vision",
            "palm-2",
            # Cohere
            "command",
            "command-light",
            "command-nightly",
            # Meta
            "llama-2-70b",
            "llama-2-13b",
            "llama-2-7b",
            # Mistral
            "mistral-7b",
            "mistral-medium",
            "mixtral-8x7b",
            # Note: Many more models are supported
            # See: https://docs.litellm.ai/docs/providers
        ]
