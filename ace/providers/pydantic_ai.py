"""PydanticAI model resolution helpers.

Converts ACE model identifiers (which follow LiteLLM conventions) into
PydanticAI model strings or provider objects.

Resolution strategy (see ``resolve_model`` for details):

1. Already has a PydanticAI provider prefix (``openai:gpt-4o``) -> pass through.
2. Starts with ``bedrock/`` and ``AWS_BEARER_TOKEN_BEDROCK`` is set ->
   create a ``BedrockProvider(api_key=...)`` with the correct model.
3. Starts with a LiteLLM prefix that has a PydanticAI native equivalent
   (``bedrock/model``) -> rewrite to ``bedrock:model``.
4. Everything else -> prepend ``litellm:`` for the LiteLLM proxy provider.

Why not always use ``litellm:``?  PydanticAI's LiteLLM provider is an
OpenAI-compatible HTTP client.  Providers that aren't OpenAI-compatible
(Bedrock via SigV4, Anthropic's native API, etc.) need PydanticAI's
native provider instead.

Provider SDK requirements
~~~~~~~~~~~~~~~~~~~~~~~~~

By default ACE installs ``pydantic-ai-slim[litellm]`` — LiteLLM is the
only provider backend available out of the box.  To use a PydanticAI
**native** provider (faster, no litellm proxy overhead, uses the
provider's own API key env vars directly), install the corresponding
``pydantic-ai-slim`` extra and its SDK:

============  ====================================  ========================
Provider      Install                               API key env var
============  ====================================  ========================
Anthropic     ``pip install pydantic-ai-slim[anthropic]``  ``ANTHROPIC_API_KEY``
OpenAI        ``pip install pydantic-ai-slim[openai]``     ``OPENAI_API_KEY``
Bedrock       ``pip install pydantic-ai-slim[bedrock]``    AWS credentials / ``AWS_BEARER_TOKEN_BEDROCK``
Google        ``pip install pydantic-ai-slim[google]``     ``GEMINI_API_KEY``
============  ====================================  ========================

Without the native extra, models that match a known prefix (e.g.
``openai/gpt-4o-mini``, ``anthropic/claude-...``) are rewritten to
the native PydanticAI prefix — but will fail at runtime if the SDK
package is missing.  Models with **no** recognized prefix fall through
to ``litellm:<model>`` automatically.

.. warning::

   LiteLLM may override API keys if proxy-related env vars (e.g.
   ``LITELLM_API_KEY``, ``SPH_LITELLM_KEY``) are set.  When using
   native providers, ensure these are unset or scoped to avoid key
   conflicts.
"""

from __future__ import annotations

import os
from typing import Any, Union

from pydantic_ai.settings import ModelSettings

from .config import ModelConfig

# PydanticAI provider names accepted as ``<provider>:<model>`` prefixes.
_PYDANTIC_AI_PROVIDERS: frozenset[str] = frozenset(
    {
        "anthropic",
        "azure",
        "bedrock",
        "cerebras",
        "cohere",
        "deepseek",
        "google",
        "google-gla",
        "google-vertex",
        "grok",
        "groq",
        "litellm",
        "mistral",
        "openai",
        "openai-chat",
        "openai-responses",
        "openrouter",
        "vercel",
        "vertexai",
    }
)

# LiteLLM uses ``provider/model`` while PydanticAI uses ``provider:model``.
# When the first path segment of a LiteLLM string matches a PydanticAI
# native provider, we rewrite ``/`` -> ``:`` so PydanticAI uses its own
# provider (with proper auth, API format, etc.) instead of the generic
# OpenAI-compatible LiteLLM proxy.
_LITELLM_PREFIX_TO_NATIVE: frozenset[str] = frozenset(
    {
        "anthropic",
        "azure",
        "azure_ai",
        "bedrock",
        "cohere",
        "deepseek",
        "groq",
        "mistral",
        "openrouter",
        "vertex_ai",
    }
)


def resolve_model(model: str) -> Any:
    """Resolve an ACE/LiteLLM model string for PydanticAI.

    Returns either a string (for PydanticAI's auto-provider detection)
    or a ``(provider, model_name)`` tuple when explicit provider
    configuration is needed (e.g. Bedrock API key auth).

    Resolution paths:

    1. **PydanticAI-native prefix** -- Already starts with a known
       PydanticAI provider prefix (e.g. ``openai:gpt-4o``).  Returned
       unchanged.

    2. **Bedrock with API key** -- Starts with ``bedrock/`` and
       ``AWS_BEARER_TOKEN_BEDROCK`` is set.  Returns a ``BedrockModel``
       configured with bearer-token auth.

    3. **LiteLLM prefix with native equivalent** -- First path segment
       matches a PydanticAI native provider (e.g. ``bedrock/model-id:0``).
       Rewrites ``/`` to ``:`` for PydanticAI's native provider (SigV4).

    4. **Fallback** -- Prepend ``litellm:`` for the LiteLLM proxy.

    Examples::

        resolve_model("gpt-4o-mini")
        # -> "litellm:gpt-4o-mini"

        resolve_model("bedrock/us.anthropic.claude-haiku-4-5-v1:0")
        # -> BedrockModel (if AWS_BEARER_TOKEN_BEDROCK set)
        # -> "bedrock:us.anthropic.claude-haiku-4-5-v1:0" (otherwise)

        resolve_model("openai:gpt-4o")
        # -> "openai:gpt-4o"  (unchanged)

    Args:
        model: Model identifier -- LiteLLM convention
            (``"gpt-4o-mini"``, ``"bedrock/model-id:0"``) or
            PydanticAI convention (``"openai:gpt-4o"``).

    Returns:
        PydanticAI model string or model object.
    """
    # Path 1: already has a PydanticAI provider prefix
    if ":" in model:
        prefix = model.split(":", 1)[0]
        if prefix in _PYDANTIC_AI_PROVIDERS:
            return model

    # Path 2: Bedrock with API key (bearer token auth)
    if "/" in model:
        litellm_prefix = model.split("/", 1)[0]
        if litellm_prefix == "bedrock":
            bedrock_api_key = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
            if bedrock_api_key:
                return _create_bedrock_model(model, bedrock_api_key)

    # Path 3: LiteLLM prefix with a native PydanticAI equivalent
    if "/" in model:
        litellm_prefix = model.split("/", 1)[0]
        if litellm_prefix in _LITELLM_PREFIX_TO_NATIVE:
            rest = model.split("/", 1)[1]
            pydantic_prefix = litellm_prefix
            # Normalize LiteLLM aliases to PydanticAI names
            if litellm_prefix == "vertex_ai":
                pydantic_prefix = "google-vertex"
            elif litellm_prefix == "azure_ai":
                pydantic_prefix = "azure"
            return f"{pydantic_prefix}:{rest}"

    # Path 4: no recognized prefix -> route through LiteLLM provider
    return f"litellm:{model}"


def _create_bedrock_model(model: str, api_key: str) -> Any:
    """Create a PydanticAI BedrockModel with API key (bearer token) auth.

    This is used when ``AWS_BEARER_TOKEN_BEDROCK`` is set, bypassing
    boto3 SigV4 auth entirely.

    Args:
        model: Full LiteLLM model string (e.g.
            ``"bedrock/us.anthropic.claude-haiku-4-5-v1:0"``).
        api_key: Bedrock API key (bearer token).

    Returns:
        PydanticAI ``BedrockModel`` configured with bearer auth.
    """
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

    # Extract model ID: "bedrock/us.anthropic.claude-..." -> "us.anthropic.claude-..."
    model_id = model.split("/", 1)[1]

    # Infer region from inference profile prefix
    region = "us-east-1"
    if model_id.startswith("eu."):
        region = "eu-west-1"

    provider = BedrockProvider(api_key=api_key, region_name=region)
    return BedrockConverseModel(model_name=model_id, provider=provider)


def settings_from_config(config: ModelConfig) -> ModelSettings:
    """Create ``ModelSettings`` from a ``ModelConfig``.

    Maps ACE configuration (temperature, max_tokens) to PydanticAI's
    model settings.

    Args:
        config: ACE model configuration.

    Returns:
        PydanticAI ``ModelSettings`` with temperature and max_tokens.
    """
    return ModelSettings(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
