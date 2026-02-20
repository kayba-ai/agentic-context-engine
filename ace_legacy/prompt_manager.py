"""
Unified Prompt Manager for ACE Framework (v2.1 only).

Usage:
    >>> from ace_legacy.prompt_manager import PromptManager
    >>> manager = PromptManager()
    >>> agent_prompt = manager.get_agent_prompt()
    >>> reflector_prompt = manager.get_reflector_prompt()
    >>> skill_manager_prompt = manager.get_skill_manager_prompt()
"""

from datetime import datetime
from typing import Dict, Any, Optional

__all__ = [
    "PromptManager",
    "validate_prompt_output_v2_1",
    "wrap_skillbook_for_external_agent",
]

# Lazy load to avoid circular imports
_prompts_cache: Dict[str, Any] = {}


def _get_v2_1_prompts():
    """Lazily load v2.1 prompts to avoid circular imports."""
    if "v2_1" not in _prompts_cache:
        from . import prompts_v2_1

        _prompts_cache["v2_1"] = {
            "AGENT_V2_1_PROMPT": prompts_v2_1.AGENT_V2_1_PROMPT,
            "AGENT_MATH_V2_1_PROMPT": prompts_v2_1.AGENT_MATH_V2_1_PROMPT,
            "AGENT_CODE_V2_1_PROMPT": prompts_v2_1.AGENT_CODE_V2_1_PROMPT,
            "REFLECTOR_V2_1_PROMPT": prompts_v2_1.REFLECTOR_V2_1_PROMPT,
            "SKILL_MANAGER_V2_1_PROMPT": prompts_v2_1.SKILL_MANAGER_V2_1_PROMPT,
            "SKILLBOOK_USAGE_INSTRUCTIONS": prompts_v2_1.SKILLBOOK_USAGE_INSTRUCTIONS,
            "wrap_skillbook_for_external_agent": prompts_v2_1.wrap_skillbook_for_external_agent,
        }
    return _prompts_cache["v2_1"]


class PromptManager:
    """
    Prompt Manager supporting ACE v2.1 prompts.

    Features:
    - Domain-specific prompt selection (math, code)
    - Usage tracking

    Example:
        >>> manager = PromptManager()
        >>> prompt = manager.get_agent_prompt(domain="math")
    """

    PROMPTS = {
        "agent": {
            "2.1": "v2_1:AGENT_V2_1_PROMPT",
            "2.1-math": "v2_1:AGENT_MATH_V2_1_PROMPT",
            "2.1-code": "v2_1:AGENT_CODE_V2_1_PROMPT",
        },
        "reflector": {
            "2.1": "v2_1:REFLECTOR_V2_1_PROMPT",
        },
        "skill_manager": {
            "2.1": "v2_1:SKILL_MANAGER_V2_1_PROMPT",
        },
    }

    def __init__(self, default_version: str = "2.1"):
        self.default_version = default_version
        self.usage_stats: Dict[str, int] = {}

    def get_agent_prompt(
        self, domain: Optional[str] = None, version: Optional[str] = None
    ) -> str:
        version = version or self.default_version

        if domain and f"{version}-{domain}" in self.PROMPTS["agent"]:
            prompt_key = f"{version}-{domain}"
        else:
            prompt_key = version

        prompt = self._resolve_prompt("agent", prompt_key)
        self._track_usage(f"agent-{prompt_key}")

        if prompt is not None and "{current_date}" in prompt:
            prompt = prompt.replace(
                "{current_date}", datetime.now().strftime("%Y-%m-%d")
            )

        if prompt is None:
            raise ValueError(f"No agent prompt found for version {version}")

        return prompt

    def get_reflector_prompt(self, version: Optional[str] = None) -> str:
        version = version or self.default_version
        prompt = self._resolve_prompt("reflector", version)
        self._track_usage(f"reflector-{version}")

        if prompt is None:
            raise ValueError(f"No reflector prompt found for version {version}")

        return prompt

    def get_skill_manager_prompt(self, version: Optional[str] = None) -> str:
        version = version or self.default_version
        prompt = self._resolve_prompt("skill_manager", version)
        self._track_usage(f"skill_manager-{version}")

        if prompt is None:
            raise ValueError(f"No skill_manager prompt found for version {version}")

        return prompt

    def _resolve_prompt(self, role: str, version: str) -> Optional[str]:
        ref = self.PROMPTS.get(role, {}).get(version)
        if ref is None:
            return None

        if ref.startswith("v2_1:"):
            attr_name = ref.split(":")[1]
            return _get_v2_1_prompts()[attr_name]

        return ref

    def _track_usage(self, prompt_id: str) -> None:
        self.usage_stats[prompt_id] = self.usage_stats.get(prompt_id, 0) + 1

    def get_stats(self) -> Dict[str, Any]:
        return {
            "usage": self.usage_stats.copy(),
            "total_calls": sum(self.usage_stats.values()),
        }

    @staticmethod
    def list_available_versions() -> Dict[str, list]:
        return {
            role: list(prompts.keys())
            for role, prompts in PromptManager.PROMPTS.items()
        }


def wrap_skillbook_for_external_agent(skillbook, version: str = "2.1") -> str:
    """
    Wrap skillbook skills with explanation for external agents.

    Args:
        skillbook: Skillbook instance with learned strategies
        version: Wrapper version (only "2.1" supported)

    Returns:
        Formatted text with skillbook strategies and usage instructions.
        Returns empty string if skillbook has no skills.
    """
    return _get_v2_1_prompts()["wrap_skillbook_for_external_agent"](skillbook)


def validate_prompt_output_v2_1(
    output: str, role: str
) -> tuple[bool, list[str], Dict[str, float]]:
    """
    Enhanced validation for v2.1 prompt outputs with quality metrics.

    Args:
        output: The LLM output to validate
        role: The role (agent, reflector, skill_manager)

    Returns:
        (is_valid, error_messages, quality_metrics)
    """
    import json

    errors = []
    metrics: Dict[str, float] = {}

    try:
        data = json.loads(output)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return False, errors, {}

    if role == "agent":
        required = ["reasoning", "final_answer"]
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")

    elif role == "reflector":
        required = ["reasoning", "error_identification", "skill_tags"]
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")

    elif role == "skill_manager":
        required = ["reasoning", "operations"]
        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        for op in data.get("operations", []):
            if op.get("type") not in ["ADD", "UPDATE", "TAG", "REMOVE"]:
                errors.append(f"Invalid operation type: {op.get('type')}")

    return len(errors) == 0, errors, metrics
