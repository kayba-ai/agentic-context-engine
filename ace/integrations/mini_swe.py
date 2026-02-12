"""
mini-swe-agent integration for ACE framework.

This module provides ACEMiniSWE, a wrapper for mini-swe-agent that automatically
learns from SWE-bench-style coding task execution.

mini-swe-agent is a 100-line agent from Princeton/Stanford that scores 74%+ on
SWE-bench verified. This integration follows the ACE pattern:
1. External framework (mini-swe-agent) executes task
2. ACE injects skillbook context beforehand
3. ACE learns from execution afterward (Reflector + SkillManager)

Example:
    from ace.integrations import ACEMiniSWE

    agent = ACEMiniSWE(model="claude-sonnet-4-20250514")
    result = agent.run(task="Fix the bug in utils.py")
    agent.save_skillbook("swe_expert.json")
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

try:
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models import get_model
    from minisweagent.environments import get_environment

    MINI_SWE_AVAILABLE = True
except ImportError:
    MINI_SWE_AVAILABLE = False
    DefaultAgent = None  # type: ignore[misc,assignment]
    get_model = None  # type: ignore[misc,assignment]
    get_environment = None  # type: ignore[misc,assignment]

from ..llm_providers import LiteLLMClient
from ..skillbook import Skillbook
from ..roles import Reflector, SkillManager, AgentOutput
from ..prompts_v2_1 import PromptManager
from .base import wrap_skillbook_context

if TYPE_CHECKING:
    from ..deduplication import DeduplicationConfig


@dataclass
class MiniSWEResult:
    """Result from mini-swe-agent execution."""

    status: str
    """Exit status: 'Submitted', 'LimitsExceeded', or exception name."""

    message: str
    """Final output message or error."""

    trace: List[Dict[str, Any]]
    """Full execution trace (agent.messages)."""

    steps: int
    """Number of steps taken."""

    success: bool
    """Whether task completed successfully (status == 'Submitted')."""


class ACEMiniSWE:
    """
    mini-swe-agent with ACE learning capabilities.

    Wraps mini-swe-agent to automatically:
    - Inject learned strategies into tasks
    - Reflect on execution results
    - Update skillbook with new learnings

    Key difference from standard mini-swe-agent:
    - No ACE Agent (mini-swe-agent executes directly)
    - Skillbook provides context only
    - Reflector + SkillManager run AFTER execution

    Insight Level: Meso
        ACE sees the full agent execution trace (thoughts, commands, observations)
        without external ground truth. Learns from execution patterns rather than
        correctness feedback. See docs/COMPLETE_GUIDE_TO_ACE.md for details.

    Usage:
        # Simple usage
        agent = ACEMiniSWE(model="claude-sonnet-4-20250514")
        result = agent.run(task="Fix the TypeError in main.py")

        # Reuse across tasks (learns from each)
        agent = ACEMiniSWE(model="gpt-4o")
        agent.run(task="Task 1")
        agent.run(task="Task 2")  # Uses Task 1 learnings
        agent.save_skillbook("expert.json")

        # Start with existing knowledge
        agent = ACEMiniSWE(
            model="claude-sonnet-4-20250514",
            skillbook_path="expert.json"
        )
        result = agent.run(task="New task")

        # Disable learning for debugging
        agent = ACEMiniSWE(
            model="claude-sonnet-4-20250514",
            skillbook_path="expert.json",
            is_learning=False
        )
        result = agent.run(task="Test task")
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        ace_model: str = "gpt-4o-mini",
        ace_llm: Optional[LiteLLMClient] = None,
        ace_max_tokens: int = 2048,
        skillbook: Optional[Skillbook] = None,
        skillbook_path: Optional[str] = None,
        is_learning: bool = True,
        dedup_config: Optional["DeduplicationConfig"] = None,
        # Environment config
        env_type: str = "local",
        env_config: Optional[Dict[str, Any]] = None,
        # Agent config
        step_limit: int = 30,
        cost_limit: float = 3.0,
        **agent_kwargs,
    ):
        """
        Initialize ACEMiniSWE.

        Args:
            model: Model name for mini-swe-agent execution (default: claude-sonnet-4-20250514)
            ace_model: Model name for ACE learning (Reflector/SkillManager)
            ace_llm: Custom LLM client for ACE (overrides ace_model)
            ace_max_tokens: Max tokens for ACE learning LLM (default: 2048)
            skillbook: Existing Skillbook instance
            skillbook_path: Path to load skillbook from
            is_learning: Enable/disable ACE learning
            dedup_config: Optional DeduplicationConfig for skill deduplication
            env_type: Environment type ('local', 'docker', 'singularity', etc.)
            env_config: Environment configuration dict
            step_limit: Maximum agent steps (default: 30)
            cost_limit: Maximum cost in USD (default: 3.0)
            **agent_kwargs: Additional DefaultAgent parameters
        """
        if not MINI_SWE_AVAILABLE:
            raise ImportError(
                "mini-swe-agent is not installed. Install with: "
                "pip install mini-swe-agent"
            )

        self.model = model
        self.is_learning = is_learning
        self.env_type = env_type
        self.env_config = env_config or {}
        self.step_limit = step_limit
        self.cost_limit = cost_limit
        self.agent_kwargs = agent_kwargs

        # Load or create skillbook
        if skillbook_path:
            self.skillbook = Skillbook.load_from_file(skillbook_path)
        elif skillbook:
            self.skillbook = skillbook
        else:
            self.skillbook = Skillbook()

        # Create ACE LLM (for Reflector/SkillManager, NOT execution)
        self.ace_llm = ace_llm or LiteLLMClient(
            model=ace_model, max_tokens=ace_max_tokens
        )

        # Create ACE learning components with v2.1 prompts (NO ACE AGENT!)
        prompt_mgr = PromptManager()
        self.reflector = Reflector(
            self.ace_llm, prompt_template=prompt_mgr.get_reflector_prompt()
        )

        # Create DeduplicationManager if config provided
        dedup_manager = None
        if dedup_config is not None:
            from ..deduplication import DeduplicationManager

            dedup_manager = DeduplicationManager(dedup_config)

        self.skill_manager = SkillManager(
            self.ace_llm,
            prompt_template=prompt_mgr.get_skill_manager_prompt(),
            dedup_manager=dedup_manager,
        )

    def run(
        self,
        task: str,
        working_dir: Optional[str] = None,
        **run_kwargs,
    ) -> MiniSWEResult:
        """
        Run SWE task with mini-swe-agent and ACE learning.

        Args:
            task: Task description (e.g., "Fix the bug in utils.py")
            working_dir: Working directory for the environment
            **run_kwargs: Additional run() parameters

        Returns:
            MiniSWEResult with status, message, trace, and steps
        """
        # 1. INJECT: Add skillbook context to task
        if self.is_learning and self.skillbook and self.skillbook.skills():
            skillbook_context = wrap_skillbook_context(self.skillbook)
            enhanced_task = f"""{task}

{skillbook_context}"""
        else:
            enhanced_task = task

        # 2. EXECUTE: Create and run mini-swe-agent
        # Create model
        swe_model = get_model(self.model)

        # Create environment
        env_config = {**self.env_config}
        if working_dir:
            env_config["cwd"] = working_dir
        swe_env = get_environment(env_config, default_type=self.env_type)

        # Create agent
        agent = DefaultAgent(
            model=swe_model,
            env=swe_env,
            step_limit=self.step_limit,
            cost_limit=self.cost_limit,
            **self.agent_kwargs,
        )

        # Run task
        status, message = agent.run(enhanced_task, **run_kwargs)
        trace = agent.messages
        success = status == "Submitted"

        # Count actual steps (assistant messages = steps)
        steps = sum(1 for m in trace if m.get("role") == "assistant")

        # 3. LEARN: ACE learns from execution
        if self.is_learning:
            self._learn_from_execution(task, status, message, trace, success)

        return MiniSWEResult(
            status=status,
            message=message,
            trace=trace,
            steps=steps,
            success=success,
        )

    def _build_rich_feedback(
        self, trace: List[Dict[str, Any]], status: str, message: str, success: bool
    ) -> Dict[str, Any]:
        """
        Build comprehensive trace information from mini-swe-agent execution.

        Returns dict with:
        - feedback: formatted feedback string for Reflector
        - raw_trace: structured trace data for AgentOutput.raw
        - steps: number of steps executed
        """
        # Count steps (assistant messages)
        steps = sum(1 for m in trace if m.get("role") == "assistant")

        feedback_parts = []

        # Overall status
        status_desc = "succeeded" if success else "failed"
        feedback_parts.append(f"SWE task {status_desc} in {steps} steps")
        feedback_parts.append(f"Exit status: {status}")

        # Build CHRONOLOGICAL execution trace
        feedback_parts.append("\n\n=== SWE AGENT EXECUTION TRACE (Chronological) ===")

        step_num = 0
        for msg in trace:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                feedback_parts.append(f"\n[SYSTEM] {content[:200]}...")
            elif role == "user":
                if step_num == 0:
                    feedback_parts.append(f"\n[TASK] {content[:500]}...")
                else:
                    # Observation from environment
                    feedback_parts.append(f"\n[OBSERVATION] {content[:1000]}")
            elif role == "assistant":
                step_num += 1
                feedback_parts.append(f"\n--- Step {step_num} ---")
                feedback_parts.append(f"[AGENT RESPONSE]\n{content}")

        feedback_parts.append("\n=== END EXECUTION TRACE ===")

        # Add final output
        if message:
            feedback_parts.append(f"\n\nFinal output: {message[:500]}")

        return {
            "feedback": "\n".join(feedback_parts),
            "raw_trace": trace,
            "steps": steps,
            "output": message,
        }

    def _extract_cited_ids_from_trace(self, trace: List[Dict[str, Any]]) -> List[str]:
        """
        Extract cited skill IDs from agent's reasoning in trace.

        Parses assistant messages for [skill_xxx] patterns.

        Args:
            trace: List of message dicts from agent.messages

        Returns:
            List of cited skill IDs found in agent reasoning
        """
        cited_ids = []
        pattern = r"\[skill_[a-f0-9]{8}\]"

        for msg in trace:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                matches = re.findall(pattern, content)
                for match in matches:
                    skill_id = match.strip("[]")
                    if skill_id not in cited_ids:
                        cited_ids.append(skill_id)

        return cited_ids

    def _learn_from_execution(
        self,
        task: str,
        status: str,
        message: str,
        trace: List[Dict[str, Any]],
        success: bool,
    ):
        """
        Run ACE learning pipeline AFTER mini-swe-agent execution.

        Flow: Reflector -> SkillManager -> Update Skillbook
        (No ACE Agent - mini-swe-agent already executed)
        """
        # Extract rich trace information
        trace_info = self._build_rich_feedback(trace, status, message, success)

        # Extract cited skill IDs from agent reasoning
        cited_ids = self._extract_cited_ids_from_trace(trace)

        # Filter to valid IDs that exist in skillbook
        valid_cited_ids = [
            skill_id
            for skill_id in cited_ids
            if self.skillbook.get_skill(skill_id) is not None
        ]

        # Create AgentOutput (mini-swe-agent executed, not ACE Agent)
        agent_output = AgentOutput(
            reasoning=trace_info["feedback"],
            final_answer=trace_info["output"],
            skill_ids=valid_cited_ids,
            raw={
                "steps": trace_info["steps"],
                "success": success,
                "status": status,
                "execution_mode": "mini-swe-agent",
                "cited_strategies": cited_ids,
            },
        )

        # Build concise feedback summary
        status_desc = "succeeded" if success else "failed"
        feedback_summary = f"SWE task {status_desc} in {trace_info['steps']} steps"
        feedback_summary += f" (status: {status})"
        if not success:
            feedback_summary += f"\nError: {message[:200]}"

        # Run Reflector (sync LLM call)
        reflection = self.reflector.reflect(
            question=task,
            agent_output=agent_output,
            skillbook=self.skillbook,
            ground_truth=None,
            feedback=feedback_summary,
        )

        # Run SkillManager with enriched context
        skill_manager_output = self.skill_manager.update_skills(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context=(
                f"task: {task}\n"
                f"feedback: {feedback_summary}\n"
                f"success: {success}\n"
                f"steps: {trace_info['steps']}"
            ),
            progress=f"SWE task: {task}",
        )

        # Update skillbook with learned strategies
        self.skillbook.apply_update(skill_manager_output.update)

    def enable_learning(self):
        """Enable ACE learning."""
        self.is_learning = True

    def disable_learning(self):
        """Disable ACE learning (execution only, no updates to skillbook)."""
        self.is_learning = False

    def save_skillbook(self, path: str):
        """Save learned skillbook to file."""
        self.skillbook.save_to_file(path)

    def load_skillbook(self, path: str):
        """Load skillbook from file."""
        self.skillbook = Skillbook.load_from_file(path)

    def get_strategies(self) -> str:
        """Get current skillbook strategies as formatted text."""
        if not self.skillbook:
            return ""
        return wrap_skillbook_context(self.skillbook)

    def __repr__(self) -> str:
        """String representation."""
        skills_count = len(self.skillbook.skills()) if self.skillbook else 0
        return (
            f"ACEMiniSWE(model='{self.model}', "
            f"strategies={skills_count}, "
            f"learning={'enabled' if self.is_learning else 'disabled'})"
        )


# Export for integration module
__all__ = ["ACEMiniSWE", "MiniSWEResult", "MINI_SWE_AVAILABLE"]
