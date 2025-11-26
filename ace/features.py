"""Centralized optional dependency detection for ACE framework.

This module provides a clean interface for checking which optional dependencies
are available, avoiding scattered try/except imports throughout the codebase.

Usage:
    >>> from ace.features import has_opik, has_litellm, has_langchain
    >>> if has_opik():
    ...     from ace.observability import OpikIntegration
    ...     integration = OpikIntegration()
"""

from typing import Dict, Optional


# Cache for dependency checks to avoid repeated imports
_FEATURE_CACHE: Dict[str, bool] = {}


def _check_import(module_name: str, package: Optional[str] = None) -> bool:
    """
    Check if a module can be imported.

    Args:
        module_name: Name of the module to import
        package: Optional package name for relative imports

    Returns:
        True if module can be imported, False otherwise
    """
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


def has_langchain() -> bool:
    """Check if LangChain integration is available."""
    return _check_import("langchain_core")


def has_transformers() -> bool:
    """Check if Transformers library for local models is available."""
    return _check_import("transformers")


def has_torch() -> bool:
    """Check if PyTorch is available."""
    return _check_import("torch")


def has_browser_use() -> bool:
    """Check if browser-use library for browser automation is available."""
    return _check_import("browser_use")


def has_playwright() -> bool:
    """Check if Playwright browser automation is available."""
    return _check_import("playwright")


def has_instructor() -> bool:
    """Check if Instructor library for structured outputs is available."""
    return _check_import("instructor")


def has_numpy() -> bool:
    """Check if NumPy is available for similarity computations."""
    return _check_import("numpy")


def has_sentence_transformers() -> bool:
    """Check if sentence-transformers is available for local embeddings."""
    return _check_import("sentence_transformers")


def get_available_features() -> Dict[str, bool]:
    """
    Get a dictionary of all available features.

    Returns:
        Dictionary mapping feature names to availability status

    Example:
        >>> features = get_available_features()
        >>> print(features)
        {'opik': True, 'litellm': True, 'langchain': False, ...}
    """
    return {
        "opik": has_opik(),
        "litellm": has_litellm(),
        "langchain": has_langchain(),
        "transformers": has_transformers(),
        "torch": has_torch(),
        "browser_use": has_browser_use(),
        "playwright": has_playwright(),
        "instructor": has_instructor(),
        "numpy": has_numpy(),
        "sentence_transformers": has_sentence_transformers(),
    }


def print_feature_status() -> None:
    """Print a formatted table of available features."""
    features = get_available_features()

    print("\n" + "=" * 50)
    print("ACE Framework - Available Features")
    print("=" * 50)

    for feature, available in features.items():
        status = "✓ Available" if available else "✗ Not installed"
        print(f"  {feature:<15} {status}")

    print("=" * 50 + "\n")


if __name__ == "__main__":
    # When run directly, print feature status
    print_feature_status()
