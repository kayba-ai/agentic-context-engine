"""Model registry — discovery, validation, and provider detection.

Delegates entirely to LiteLLM for provider detection, environment
validation, and model discovery. No external API calls for discovery —
only ``validate_connection`` makes a real (tiny) LLM call.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

def _litellm():
    """Return the litellm module, importing it on first call."""
    global _litellm_mod
    try:
        return _litellm_mod  # type: ignore[name-defined]
    except NameError:
        pass
    try:
        import litellm as _mod

        _litellm_mod = _mod
        return _mod
    except ImportError:
        _litellm_mod = None
        return None


# Example model strings per provider (for user guidance in the CLI)
PROVIDER_MODEL_EXAMPLES: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "gemini": "gemini/gemini-2.0-flash",
    "deepseek": "deepseek/deepseek-chat",
    "groq": "groq/llama-3.1-70b",
    "bedrock": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "ollama": "ollama/llama2",
    "azure": "azure/gpt-4",
    "openrouter": "openrouter/anthropic/claude-3.5-sonnet",
    "minimax": "openai/MiniMax-M2.7",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of a model + key validation."""

    success: bool
    model: str = ""
    provider: str = ""
    latency_ms: int = 0
    error: str = ""


@dataclass
class ModelInfo:
    """Metadata about a model from LiteLLM's registry."""

    model: str
    provider: str
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    input_cost_per_m: float | None = None  # per million tokens
    output_cost_per_m: float | None = None
    key_found: bool = False


# ---------------------------------------------------------------------------
# Provider detection (delegated to LiteLLM)
# ---------------------------------------------------------------------------


def get_provider(model: str) -> str:
    """Return the provider name for a model string, or 'unknown'."""
    ll = _litellm()
    if ll is None:
        raise ImportError("LiteLLM is required for model validation.")

    try:
        _, provider, _, _ = ll.get_llm_provider(model)
    except Exception:
        provider = "unknown"

    return provider


def get_missing_keys(model: str) -> list[str]:
    """Return env var names that LiteLLM says are missing for *model*."""
    ll = _litellm()
    if ll is None:
        return []

    try:
        result = ll.validate_environment(model=model)
        return result.get("missing_keys", [])
    except Exception:
        return []


def keys_are_set(model: str) -> bool:
    """Check whether the required keys for *model* are in the environment."""
    return len(get_missing_keys(model)) == 0


# ---------------------------------------------------------------------------
# Connection validation
# ---------------------------------------------------------------------------


def validate_connection(model: str, api_key: str | None = None) -> ValidationResult:
    """Make a minimal LLM call to verify model + key work.

    Sends a 3-token request ("Say 'ok'") to confirm authentication,
    model availability, and network connectivity.

    Args:
        model: LiteLLM model string.
        api_key: Explicit key, or None to use environment.
    """
    ll = _litellm()
    if ll is None:
        return ValidationResult(
            success=False, model=model, error="LiteLLM is not installed."
        )

    call_params: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Say 'ok'"}],
        "max_tokens": 3,
        "temperature": 0.0,
        "timeout": 15,
    }
    if api_key:
        call_params["api_key"] = api_key

    # Suppress LiteLLM's noisy debug output during validation
    prev_verbose = getattr(ll, "suppress_debug_info", False)
    ll.suppress_debug_info = True

    start = time.monotonic()
    try:
        response = ll.completion(**call_params)
        elapsed_ms = int((time.monotonic() - start) * 1000)

        provider = "unknown"
        if hasattr(response, "_hidden_params"):
            provider = response._hidden_params.get("custom_llm_provider", "unknown")

        return ValidationResult(
            success=True,
            model=model,
            provider=provider,
            latency_ms=elapsed_ms,
        )
    except ll.AuthenticationError:
        return ValidationResult(
            success=False, model=model, error="Invalid API key."
        )
    except ll.NotFoundError:
        return ValidationResult(
            success=False,
            model=model,
            error=f"Model '{model}' not found at the provider.",
        )
    except ll.APIConnectionError:
        return ValidationResult(
            success=False,
            model=model,
            error="Could not connect to the provider.",
        )
    except Exception as e:
        return ValidationResult(success=False, model=model, error=str(e))
    finally:
        ll.suppress_debug_info = prev_verbose


# ---------------------------------------------------------------------------
# Model search / discovery
# ---------------------------------------------------------------------------


PROVIDER_KEY_ENV: dict[str, str | list[str]] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "bedrock": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"],
    "bedrock_converse": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION_NAME"],
    "vertex_ai": "GOOGLE_APPLICATION_CREDENTIALS",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "fireworks_ai": "FIREWORKS_AI_API_KEY",
    "replicate": "REPLICATE_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "perplexity": "PERPLEXITYAI_API_KEY",
    "anyscale": "ANYSCALE_API_KEY",
    "minimax": "MINIMAX_API_KEY",
}


_PROVIDER_ALT_KEYS: dict[str, list[str]] = {
    "bedrock_converse": ["AWS_BEARER_TOKEN_BEDROCK"],
}


def _quick_key_check(provider: str) -> bool:
    """Fast check: are the required env vars set for this provider?"""
    env_var = PROVIDER_KEY_ENV.get(provider)
    if env_var is not None:
        if isinstance(env_var, list):
            if all(bool(os.environ.get(v)) for v in env_var):
                return True
        elif bool(os.environ.get(env_var)):
            return True

    # Alternative auth (e.g. bearer token for Bedrock)
    alt_vars = _PROVIDER_ALT_KEYS.get(provider)
    if alt_vars:
        return any(bool(os.environ.get(v)) for v in alt_vars)

    return False


def search_models(
    query: str = "",
    provider: str | None = None,
    chat_only: bool = True,
    limit: int = 20,
) -> tuple[list[ModelInfo], int]:
    """Search LiteLLM's model registry.

    Args:
        query: Substring to match against model names.
        provider: Filter to a specific provider.
        chat_only: Only return chat/completion models.
        limit: Maximum results.

    Returns:
        (results, total_matches) — results capped at *limit*,
        total_matches is the full count of matching models.
    """
    ll = _litellm()
    if ll is None:
        return [], 0

    results: list[ModelInfo] = []
    total = 0
    terms = query.lower().split() if query else []
    for model_id, info in ll.model_cost.items():
        if chat_only and info.get("mode") != "chat":
            continue
        if provider and info.get("litellm_provider") != provider:
            continue
        model_lower = model_id.lower()
        if terms and not all(t in model_lower for t in terms):
            continue

        total += 1

        if len(results) >= limit:
            continue  # keep counting total

        prov = info.get("litellm_provider", "unknown")
        input_cost = info.get("input_cost_per_token")
        output_cost = info.get("output_cost_per_token")

        # Fast key check — just look for the provider's standard env var.
        # We avoid litellm.validate_environment() here because it's slow
        # and some providers (e.g. GitHub Copilot) trigger interactive auth.
        key_found = _quick_key_check(prov)

        results.append(
            ModelInfo(
                model=model_id,
                provider=prov,
                max_input_tokens=info.get("max_input_tokens"),
                max_output_tokens=info.get("max_output_tokens"),
                input_cost_per_m=input_cost * 1_000_000 if input_cost else None,
                output_cost_per_m=output_cost * 1_000_000 if output_cost else None,
                key_found=key_found,
            )
        )

    return results, total


def suggest_models(typo: str, limit: int = 5) -> list[str]:
    """Return model names similar to *typo* (simple substring matching)."""
    ll = _litellm()
    if ll is None:
        return []

    candidates: list[str] = []
    typo_lower = typo.lower()

    for model_id, info in ll.model_cost.items():
        if info.get("mode") != "chat":
            continue
        if model_id.lower().startswith(typo_lower):
            candidates.append(model_id)
        elif typo_lower in model_id.lower():
            candidates.append(model_id)
        if len(candidates) >= limit:
            break

    return candidates
