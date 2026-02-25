"""LiteLLM client for unified access to 100+ LLM providers.

Self-contained integration — no imports from ``ace/``.  Satisfies the
``LLMClientLike`` protocol used by Agent, Reflector, and SkillManager.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)

try:
    import litellm
    from litellm import Router, acompletion, completion

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    logger.warning("LiteLLM not installed. Install with: pip install litellm")

T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# LLMResponse — lightweight container for LLM outputs
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Container for LLM outputs."""

    text: str
    raw: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# LiteLLMConfig
# ---------------------------------------------------------------------------


@dataclass
class LiteLLMConfig:
    """Configuration for LiteLLM client."""

    model: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: Optional[float] = None
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
    sampling_priority: str = "temperature"  # "temperature" | "top_p" | "top_k"

    # HTTP/SSL settings
    extra_headers: Optional[Dict[str, str]] = None
    ssl_verify: Optional[Union[bool, str]] = None

    # Model-specific parameters (reasoning_effort, budget_tokens, etc.)
    extra_params: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# LiteLLMClient
# ---------------------------------------------------------------------------


class LiteLLMClient:
    """Production LLM client using LiteLLM for unified access to multiple providers.

    Supports OpenAI, Anthropic, Google, Cohere, Azure OpenAI, AWS Bedrock,
    and 100+ other providers.

    Claude Parameter Handling:
        Due to Anthropic API limitations, temperature and top_p cannot both be
        specified for Claude models.  This client automatically resolves
        conflicts using priority-based resolution.

    Example::

        client = LiteLLMClient(model="gpt-4")
        response = client.complete("What is the capital of France?")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        fallbacks: Optional[List[str]] = None,
        sampling_priority: str = "temperature",
        config: Optional[LiteLLMConfig] = None,
        **kwargs: Any,
    ) -> None:
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "LiteLLM is not installed. Install with: pip install litellm"
            )

        if config:
            self.config = config
            if model is None:
                model = config.model
        else:
            if model is None:
                raise ValueError(
                    "Either 'model' parameter or 'config' with model must be provided"
                )
            config_fields = {
                "api_version",
                "top_p",
                "timeout",
                "max_retries",
                "metadata",
                "azure_deployment",
                "azure_api_key",
                "azure_api_base",
                "track_cost",
                "max_budget",
                "verbose",
                "extra_headers",
                "ssl_verify",
            }
            config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
            extra_params = {k: v for k, v in kwargs.items() if k not in config_fields}

            self.config = LiteLLMConfig(
                model=model,
                api_key=api_key,
                api_base=api_base,
                temperature=temperature,
                max_tokens=max_tokens,
                fallbacks=fallbacks,
                sampling_priority=sampling_priority,
                extra_params=extra_params if extra_params else None,
                **config_kwargs,
            )

        self.model = model

        if self.config.verbose:
            litellm.set_verbose = True

        self.router: Optional[Any] = None
        if self.config.fallbacks:
            self._setup_router()

    # -- Router ---------------------------------------------------------------

    def _setup_router(self) -> None:
        """Set up router for load balancing and fallbacks."""
        model_list = [
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
        ]
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

    # -- Claude parameter resolution -----------------------------------------

    @staticmethod
    def _resolve_sampling_params(
        params: Dict[str, Any], model: str, sampling_priority: str = "temperature"
    ) -> Dict[str, Any]:
        """Resolve Claude sampling-parameter conflicts.

        Anthropic API limitation: temperature and top_p cannot both be specified.
        """
        if "claude" not in model.lower():
            return params

        if sampling_priority not in ["temperature", "top_p", "top_k"]:
            raise ValueError(
                f"Invalid sampling_priority: {sampling_priority}. "
                "Must be one of: temperature, top_p, top_k"
            )

        resolved = params.copy()

        has_temperature = (
            "temperature" in resolved and resolved["temperature"] is not None
        )
        has_top_p = "top_p" in resolved and resolved["top_p"] is not None
        has_top_k = "top_k" in resolved and resolved["top_k"] is not None

        # Remove None parameters early
        for key in ("temperature", "top_p", "top_k"):
            if key in resolved and resolved[key] is None:
                resolved.pop(key)

        if (
            sampling_priority == "temperature"
            and has_temperature
            and resolved["temperature"] > 0
        ):
            resolved.pop("top_p", None)
            resolved.pop("top_k", None)
            if has_top_p or has_top_k:
                logger.info(
                    f"Claude model {model}: Using temperature={resolved['temperature']}, "
                    "ignoring other sampling params"
                )
        elif sampling_priority == "top_p" and has_top_p:
            resolved.pop("temperature", None)
            resolved.pop("top_k", None)
            if has_temperature or has_top_k:
                logger.info(
                    f"Claude model {model}: Using top_p={resolved['top_p']}, "
                    "ignoring other sampling params"
                )
        elif sampling_priority == "top_k" and has_top_k:
            resolved.pop("temperature", None)
            resolved.pop("top_p", None)
            if has_temperature or has_top_p:
                logger.info(
                    f"Claude model {model}: Using top_k={resolved['top_k']}, "
                    "ignoring other sampling params"
                )
        else:
            # Fallback: default priority (temperature > top_p > top_k)
            if has_temperature and resolved["temperature"] > 0:
                resolved.pop("top_p", None)
                resolved.pop("top_k", None)
            elif has_top_p:
                resolved.pop("temperature", None)
                resolved.pop("top_k", None)
            elif has_temperature:
                resolved.pop("top_p", None)
                resolved.pop("top_k", None)

        return resolved

    # -- Call building --------------------------------------------------------

    def _build_call_params(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> Dict[str, Any]:
        """Build the parameter dict for a litellm.completion() call."""
        merged_params: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "timeout": kwargs.get("timeout", self.config.timeout),
            "num_retries": kwargs.get("num_retries", self.config.max_retries),
            "drop_params": True,
        }

        if kwargs.get("top_p") is not None or self.config.top_p is not None:
            merged_params["top_p"] = kwargs.get("top_p", self.config.top_p)
        if kwargs.get("top_k") is not None:
            merged_params["top_k"] = kwargs.get("top_k")

        call_params = self._resolve_sampling_params(
            merged_params, self.config.model, self.config.sampling_priority
        )

        if self.config.api_key:
            call_params["api_key"] = self.config.api_key
        if self.config.api_base:
            call_params["api_base"] = self.config.api_base

        if self.config.extra_headers:
            call_params["extra_headers"] = self.config.extra_headers
        if self.config.ssl_verify is not None:
            call_params["ssl_verify"] = self.config.ssl_verify

        if self.config.extra_params:
            call_params.update(self.config.extra_params)

        # Forward remaining kwargs (excluding ACE-specific and handled params)
        ace_specific = {
            "refinement_round",
            "max_refinement_rounds",
            "stream_thinking",
        }
        handled = {
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
                if k not in call_params and k not in ace_specific and k not in handled
            }
        )

        return call_params

    # -- Completion methods ---------------------------------------------------

    def _call_completion(self, call_params: Dict[str, Any]) -> LLMResponse:
        """Execute litellm.completion() and wrap the result."""
        if self.router:
            response = self.router.completion(**call_params)
        else:
            response = completion(**call_params)

        text = response.choices[0].message.content or ""
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

    def complete(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        """Generate completion for the given prompt."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        call_params = self._build_call_params(messages, **kwargs)
        try:
            return self._call_completion(call_params)
        except Exception as e:
            logger.error(f"Error in LiteLLM completion: {e}")
            raise

    def complete_messages(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        """Multi-turn completion preserving structured message context."""
        call_params = self._build_call_params(messages, **kwargs)
        try:
            return self._call_completion(call_params)
        except Exception as e:
            logger.error(f"Error in LiteLLM multi-turn completion: {e}")
            raise

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from LLM response, handling markdown fences and preamble."""
        text = text.strip()

        # Strip markdown code fences
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: find the first top-level { ... } block
        match = re.search(r"\{", text)
        if match:
            start = match.start()
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        return json.loads(text[start : i + 1])

        raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}...")

    def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        **kwargs: Any,
    ) -> T:
        """Structured output: call complete(), parse JSON, validate with Pydantic.

        Handles markdown-fenced responses and retries on parse failure.
        """
        max_retries = kwargs.pop("max_retries", 3)
        last_error: Exception | None = None

        for attempt in range(max_retries):
            response = self.complete(prompt, **kwargs)
            try:
                data = self._extract_json(response.text)
                return response_model.model_validate(data)
            except (json.JSONDecodeError, ValueError, Exception) as e:
                last_error = e
                logger.warning(
                    "Structured parse attempt %d/%d failed: %s",
                    attempt + 1,
                    max_retries,
                    e,
                )

        raise ValueError(
            f"Failed to parse structured output after {max_retries} attempts: {last_error}"
        )

    async def acomplete(
        self, prompt: str, system: Optional[str] = None, **kwargs: Any
    ) -> LLMResponse:
        """Async version of complete."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        call_params = self._build_call_params(messages, **kwargs)
        try:
            if self.router:
                response = await self.router.acompletion(**call_params)
            else:
                response = await acompletion(**call_params)

            text = response.choices[0].message.content or ""
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
        """Generate completion with streaming support."""
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

    # -- Helpers --------------------------------------------------------------

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
        """List commonly supported models."""
        if not LITELLM_AVAILABLE:
            return []
        return [
            "gpt-4",
            "gpt-4-turbo-preview",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-3.5-turbo",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "gemini-pro",
            "command",
            "llama-2-70b",
            "mistral-7b",
            "mixtral-8x7b",
        ]
