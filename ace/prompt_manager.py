"""
Unified Prompt Manager for ACE Framework.

This module provides a single PromptManager class supporting all prompt versions
(1.0, 2.0, 2.1, 3.0) with backward compatibility.

Usage:
    >>> from ace.prompt_manager import PromptManager
    >>> manager = PromptManager(default_version="2.1")
    >>> agent_prompt = manager.get_agent_prompt()
    >>> reflector_prompt = manager.get_reflector_prompt()
    >>> skill_manager_prompt = manager.get_skill_manager_prompt(version="3.0")
"""

from datetime import datetime
from typing import Dict, Any, List, Optional

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


def _get_v3_prompts():
    """Lazily load v3 prompts to avoid circular imports."""
    if "v3" not in _prompts_cache:
        from . import prompts_v3

        _prompts_cache["v3"] = {
            "SKILL_MANAGER_V3_PROMPT": prompts_v3.SKILL_MANAGER_V3_PROMPT,
            "wrap_skillbook_for_external_agent": prompts_v3.wrap_skillbook_for_external_agent,
        }
    return _prompts_cache["v3"]


class PromptManager:
    """
    Unified Prompt Manager supporting all ACE prompt versions.

    Features:
    - Version control (1.0, 2.0, 2.1, 3.0)
    - Domain-specific prompt selection (math, code)
    - Quality metrics tracking
    - A/B testing support
    - Backward compatibility with all legacy imports

    Example:
        >>> manager = PromptManager(default_version="2.1")
        >>> prompt = manager.get_agent_prompt(domain="math")
        >>> prompt = manager.get_skill_manager_prompt(version="3.0")
    """

    # Unified version registry for all prompt versions
    # Using string references for lazy loading
    PROMPTS = {
        "agent": {
            "1.0": "ace.prompts.AGENT_PROMPT",
            "2.0": "ace.prompts_v2.AGENT_V2_PROMPT",
            "2.1": "v2_1:AGENT_V2_1_PROMPT",
            "2.1-math": "v2_1:AGENT_MATH_V2_1_PROMPT",
            "2.1-code": "v2_1:AGENT_CODE_V2_1_PROMPT",
        },
        "reflector": {
            "1.0": "ace.prompts.REFLECTOR_PROMPT",
            "2.0": "ace.prompts_v2.REFLECTOR_V2_PROMPT",
            "2.1": "v2_1:REFLECTOR_V2_1_PROMPT",
        },
        "skill_manager": {
            "1.0": "ace.prompts.SKILL_MANAGER_PROMPT",
            "2.0": "ace.prompts_v2.SKILL_MANAGER_V2_PROMPT",
            "2.1": "v2_1:SKILL_MANAGER_V2_1_PROMPT",
            "3.0": "v3:SKILL_MANAGER_V3_PROMPT",
        },
    }

    def __init__(self, default_version: str = "2.1"):
        """
        Initialize prompt manager.

        Args:
            default_version: Default version to use (1.0, 2.0, 2.1, or 3.0)
        """
        self.default_version = default_version
        self.usage_stats: Dict[str, int] = {}
        self.quality_scores: Dict[str, List[float]] = {}

    def get_agent_prompt(
        self, domain: Optional[str] = None, version: Optional[str] = None
    ) -> str:
        """
        Get agent prompt for specific domain and version.

        Args:
            domain: Domain (math, code, etc.) or None for general
            version: Version string (1.0, 2.0, 2.1) or None for default

        Returns:
            Formatted prompt template
        """
        version = version or self.default_version

        # Check for domain-specific variant
        if domain and f"{version}-{domain}" in self.PROMPTS["agent"]:
            prompt_key = f"{version}-{domain}"
        else:
            prompt_key = version

        prompt = self._resolve_prompt("agent", prompt_key)

        # Track usage
        self._track_usage(f"agent-{prompt_key}")

        # Add current date for v2+ prompts
        if (
            prompt is not None
            and version.startswith("2")
            and "{current_date}" in prompt
        ):
            prompt = prompt.replace(
                "{current_date}", datetime.now().strftime("%Y-%m-%d")
            )

        if prompt is None:
            raise ValueError(f"No agent prompt found for version {version}")

        return prompt

    def get_reflector_prompt(self, version: Optional[str] = None) -> str:
        """
        Get reflector prompt for specific version.

        Args:
            version: Version string (1.0, 2.0, 2.1) or None for default

        Returns:
            Formatted prompt template
        """
        version = version or self.default_version
        prompt = self._resolve_prompt("reflector", version)

        self._track_usage(f"reflector-{version}")

        if prompt is None:
            raise ValueError(f"No reflector prompt found for version {version}")

        return prompt

    def get_skill_manager_prompt(self, version: Optional[str] = None) -> str:
        """
        Get skill_manager prompt for specific version.

        Args:
            version: Version string (1.0, 2.0, 2.1, 3.0) or None for default

        Returns:
            Formatted prompt template
        """
        version = version or self.default_version
        prompt = self._resolve_prompt("skill_manager", version)

        self._track_usage(f"skill_manager-{version}")

        if prompt is None:
            raise ValueError(f"No skill_manager prompt found for version {version}")

        return prompt

    def _resolve_prompt(self, role: str, version: str) -> Optional[str]:
        """Resolve a prompt reference to actual content."""
        ref = self.PROMPTS.get(role, {}).get(version)
        if ref is None:
            return None

        # Handle v2.1 lazy references
        if ref.startswith("v2_1:"):
            attr_name = ref.split(":")[1]
            return _get_v2_1_prompts()[attr_name]

        # Handle v3 lazy references
        if ref.startswith("v3:"):
            attr_name = ref.split(":")[1]
            return _get_v3_prompts()[attr_name]

        # Handle legacy ace.* module references
        if ref.startswith("ace."):
            return self._resolve_module_reference(ref)

        # Direct string content
        return ref

    def _resolve_module_reference(self, ref: str) -> str:
        """Resolve a module reference string to actual prompt content."""
        module_parts = ref.split(".")
        module_name = module_parts[1]
        attr_name = module_parts[-1]

        if module_name == "prompts_v2":
            from ace import prompts_v2

            return getattr(prompts_v2, attr_name)
        elif module_name == "prompts_v2_1":
            from ace import prompts_v2_1

            return getattr(prompts_v2_1, attr_name)
        else:
            from ace import prompts

            return getattr(prompts, attr_name)

    def _track_usage(self, prompt_id: str) -> None:
        """Track prompt usage for analysis."""
        self.usage_stats[prompt_id] = self.usage_stats.get(prompt_id, 0) + 1

    def track_quality(self, prompt_id: str, score: float) -> None:
        """
        Track quality scores for prompts.

        Args:
            prompt_id: Identifier for the prompt
            score: Quality score (0.0-1.0)
        """
        if prompt_id not in self.quality_scores:
            self.quality_scores[prompt_id] = []
        self.quality_scores[prompt_id].append(score)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive prompt statistics."""
        avg_quality = {}
        for prompt_id, scores in self.quality_scores.items():
            if scores:
                avg_quality[prompt_id] = sum(scores) / len(scores)

        return {
            "usage": self.usage_stats.copy(),
            "average_quality": avg_quality,
            "total_calls": sum(self.usage_stats.values()),
        }

    @staticmethod
    def list_available_versions() -> Dict[str, list]:
        """List all available prompt versions."""
        return {
            role: list(prompts.keys())
            for role, prompts in PromptManager.PROMPTS.items()
        }

    def compare_versions(self, role: str, test_input: Dict[str, Any]) -> Dict[str, str]:
        """
        Compare different prompt versions for A/B testing.

        Args:
            role: The role (agent, reflector, skill_manager)
            test_input: Input parameters for testing

        Returns:
            Dict mapping version to formatted prompt
        """
        results = {}
        for version in self.PROMPTS.get(role, {}).keys():
            if version.startswith("2") or version.startswith("3"):
                prompt = self._resolve_prompt(role, version)
                if prompt and not prompt.startswith("ace."):
                    # Format with test input
                    try:
                        formatted = prompt.format(**test_input)
                        results[version] = formatted[:500] + "..."  # Preview
                    except KeyError:
                        results[version] = "Missing required parameters"
        return results


def wrap_skillbook_for_external_agent(skillbook, version: str = "2.1") -> str:
    """
    Wrap skillbook skills with explanation for external agents.

    This is the canonical function for injecting skillbook context into
    external agentic systems (browser-use, custom agents, LangChain, etc.).

    Args:
        skillbook: Skillbook instance with learned strategies
        version: Wrapper version ("2.1" for full instructions, "3.0" for minimal)

    Returns:
        Formatted text with skillbook strategies and usage instructions.
        Returns empty string if skillbook has no skills.

    Example:
        >>> from ace import Skillbook
        >>> from ace.prompt_manager import wrap_skillbook_for_external_agent
        >>> skillbook = Skillbook()
        >>> skillbook.add_skill("general", "Always verify inputs")
        >>> context = wrap_skillbook_for_external_agent(skillbook)
        >>> enhanced_task = f"{task}\\n\\n{context}"
    """
    if version.startswith("3"):
        return _get_v3_prompts()["wrap_skillbook_for_external_agent"](skillbook)
    else:
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

    # Check if valid JSON
    try:
        data = json.loads(output)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return False, errors, {}

    # Role-specific validation with v2.1 enhancements
    if role == "agent":
        required = ["reasoning", "final_answer"]

        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Check v2.1 quality fields
        if "quality_check" in data:
            qc = data["quality_check"]
            metrics["completeness"] = (
                int(qc.get("addresses_question", False))
                + int(qc.get("reasoning_complete", False))
                + int(qc.get("citations_provided", False))
            ) / 3.0

        # Validate confidence scores
        if "confidence_scores" in data:
            for skill_id, score in data["confidence_scores"].items():
                if not 0 <= score <= 1:
                    errors.append(f"Invalid confidence score for {skill_id}: {score}")
                else:
                    metrics[f"confidence_{skill_id}"] = score

        if "answer_confidence" in data:
            metrics["overall_confidence"] = data["answer_confidence"]

    elif role == "reflector":
        required = ["reasoning", "error_identification", "skill_tags"]

        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Check v2.1 atomicity scoring
        if "extracted_learnings" in data:
            atomicity_scores = []
            for learning in data["extracted_learnings"]:
                if "atomicity_score" in learning:
                    score = learning["atomicity_score"]
                    if not 0 <= score <= 1:
                        errors.append(f"Invalid atomicity score: {score}")
                    else:
                        atomicity_scores.append(score)

            if atomicity_scores:
                metrics["avg_atomicity"] = sum(atomicity_scores) / len(atomicity_scores)

        # Validate tags
        for tag in data.get("skill_tags", []):
            if tag.get("tag") not in ["helpful", "harmful", "neutral"]:
                errors.append(f"Invalid tag: {tag.get('tag')}")
            if "impact_score" in tag:
                metrics[f"impact_{tag.get('id')}"] = tag["impact_score"]

    elif role == "skill_manager":
        required = ["reasoning", "operations"]

        for field in required:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Check v2.1 quality metrics
        if "quality_metrics" in data:
            qm = data["quality_metrics"]
            metrics["avg_atomicity"] = qm.get("avg_atomicity", 0)
            metrics["estimated_impact"] = qm.get("estimated_impact", 0)

        # Validate operations with atomicity
        for op in data.get("operations", []):
            if op.get("type") not in ["ADD", "UPDATE", "TAG", "REMOVE"]:
                errors.append(f"Invalid operation type: {op.get('type')}")

            if "atomicity_score" in op:
                score = op["atomicity_score"]
                if not 0 <= score <= 1:
                    errors.append(f"Invalid atomicity score: {score}")
                elif score < 0.4:
                    errors.append(f"Atomicity too low ({score}) - should not add")

    # Calculate overall quality
    if metrics:
        metrics["overall_quality"] = sum(metrics.values()) / len(metrics)

    return len(errors) == 0, errors, metrics
