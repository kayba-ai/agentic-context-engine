"""Registry mapping prompt version names to reflector prompt templates."""

from .prompts import REFLECTOR_RECURSIVE_PROMPT
from .prompts_rr_v3 import REFLECTOR_RECURSIVE_V3_PROMPT
from .prompts_rr_v4 import REFLECTOR_RECURSIVE_V4_PROMPT

ALL_PROMPT_VERSION_NAMES = ["base", "v3", "v4"]

_REGISTRY = {
    "base": REFLECTOR_RECURSIVE_PROMPT,
    "v3": REFLECTOR_RECURSIVE_V3_PROMPT,
    "v4": REFLECTOR_RECURSIVE_V4_PROMPT,
}


def get_prompt_template(version: str) -> str:
    """Return the prompt template for a given version name.

    Args:
        version: One of ALL_PROMPT_VERSION_NAMES (e.g. "base", "v3")

    Returns:
        The prompt template string.

    Raises:
        ValueError: If version is not registered.
    """
    if version not in _REGISTRY:
        raise ValueError(
            f"Unknown prompt version: {version}. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[version]
