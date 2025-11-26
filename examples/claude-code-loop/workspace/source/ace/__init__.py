"""Agentic Context Engineering (ACE) reproduction framework."""

from typing import Optional
from .playbook import Bullet, Playbook
from .delta import DeltaOperation, DeltaBatch
from .llm import LLMClient, DummyLLMClient, TransformersLLMClient
from .roles import (
    Generator,
    ReplayGenerator,
    Reflector,
    Curator,
    GeneratorOutput,
    ReflectorOutput,
    CuratorOutput,
)
from .adaptation import (
    OfflineAdapter,
    OnlineAdapter,
    Sample,
    TaskEnvironment,
    SimpleEnvironment,
    EnvironmentResult,
    AdapterStepResult,
)

# Import optional feature detection
from .features import has_opik, has_litellm

# Import observability components if available
if has_opik():
    try:
        from .observability import OpikIntegration as _OpikIntegration

        OpikIntegration: Optional[type] = _OpikIntegration
        OBSERVABILITY_AVAILABLE = True
    except ImportError:
        OpikIntegration: Optional[type] = None  # type: ignore
        OBSERVABILITY_AVAILABLE = False
else:
    OpikIntegration: Optional[type] = None  # type: ignore
    OBSERVABILITY_AVAILABLE = False

# Import production LLM clients if available
if has_litellm():
    try:
        from .llm_providers import LiteLLMClient as _LiteLLMClient

        LiteLLMClient: Optional[type] = _LiteLLMClient
        LITELLM_AVAILABLE = True
    except ImportError:
        LiteLLMClient: Optional[type] = None  # type: ignore
        LITELLM_AVAILABLE = False
else:
    LiteLLMClient: Optional[type] = None  # type: ignore
    LITELLM_AVAILABLE = False

# Import integrations (LiteLLM, browser-use, LangChain, etc.) if available
try:
    from .integrations import (
        ACELiteLLM as _ACELiteLLM,
        ACEAgent as _ACEAgent,
        ACELangChain as _ACELangChain,
        wrap_playbook_context as _wrap_playbook_context,
        BROWSER_USE_AVAILABLE as _BROWSER_USE_AVAILABLE,
        LANGCHAIN_AVAILABLE as _LANGCHAIN_AVAILABLE,
    )

    ACELiteLLM: Optional[type] = _ACELiteLLM
    ACEAgent: Optional[type] = _ACEAgent
    ACELangChain: Optional[type] = _ACELangChain
    wrap_playbook_context: Optional[type] = _wrap_playbook_context  # type: ignore
    BROWSER_USE_AVAILABLE = _BROWSER_USE_AVAILABLE
    LANGCHAIN_AVAILABLE = _LANGCHAIN_AVAILABLE
except ImportError:
    ACELiteLLM: Optional[type] = None  # type: ignore
    ACEAgent: Optional[type] = None  # type: ignore
    ACELangChain: Optional[type] = None  # type: ignore
    wrap_playbook_context: Optional[type] = None  # type: ignore
    BROWSER_USE_AVAILABLE = False
    LANGCHAIN_AVAILABLE = False

__all__ = [
    # Core components
    "Bullet",
    "Playbook",
    "DeltaOperation",
    "DeltaBatch",
    "LLMClient",
    "DummyLLMClient",
    "TransformersLLMClient",
    "LiteLLMClient",
    "Generator",
    "ReplayGenerator",
    "Reflector",
    "Curator",
    "GeneratorOutput",
    "ReflectorOutput",
    "CuratorOutput",
    "OfflineAdapter",
    "OnlineAdapter",
    "Sample",
    "TaskEnvironment",
    "SimpleEnvironment",
    "EnvironmentResult",
    "AdapterStepResult",
    # Out-of-box integrations
    "ACELiteLLM",  # LiteLLM integration (quick start)
    "ACEAgent",  # Browser-use integration
    "ACELangChain",  # LangChain integration (complex workflows)
    # Utilities
    "wrap_playbook_context",
    # Feature flags
    "OpikIntegration",
    "LITELLM_AVAILABLE",
    "OBSERVABILITY_AVAILABLE",
    "BROWSER_USE_AVAILABLE",
    "LANGCHAIN_AVAILABLE",
]
