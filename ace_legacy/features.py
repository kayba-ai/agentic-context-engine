"""Centralized optional dependency detection for ACE framework."""

from typing import Dict, Optional

_FEATURE_CACHE: Dict[str, bool] = {}


def _check_import(module_name: str, package: Optional[str] = None) -> bool:
    """Check if a module can be imported."""
    if module_name in _FEATURE_CACHE:
        return _FEATURE_CACHE[module_name]

    try:
        __import__(module_name, fromlist=[package] if package else [])
        _FEATURE_CACHE[module_name] = True
        return True
    except ImportError:
        _FEATURE_CACHE[module_name] = False
        return False


def has_opik() -> bool:
    """Check if Opik observability integration is available."""
    return _check_import("opik")


def has_litellm() -> bool:
    """Check if LiteLLM client is available."""
    return _check_import("litellm")


def has_transformers() -> bool:
    """Check if Transformers library for local models is available."""
    return _check_import("transformers")


def has_torch() -> bool:
    """Check if PyTorch is available."""
    return _check_import("torch")


def has_instructor() -> bool:
    """Check if Instructor library for structured outputs is available."""
    return _check_import("instructor")


def get_available_features() -> Dict[str, bool]:
    """Get a dictionary of all available features."""
    return {
        "opik": has_opik(),
        "litellm": has_litellm(),
        "transformers": has_transformers(),
        "torch": has_torch(),
        "instructor": has_instructor(),
    }
