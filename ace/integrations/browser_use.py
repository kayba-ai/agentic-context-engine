"""
Browser-use integration for ACE framework.

This module provides ACEAgent, a drop-in replacement for browser-use Agent
that automatically learns from execution feedback.

This is the reference implementation for ACE integrations with external agentic
frameworks. It demonstrates the pattern:
1. External framework (browser-use) executes task
2. ACE injects playbook context beforehand
3. ACE learns from execution afterward (Reflector + Curator)

Example:
    from ace.integrations import ACEAgent
    from browser_use import ChatBrowserUse

    agent = ACEAgent(llm=ChatBrowserUse())
    await agent.run(task="Find top HN post")
    agent.save_playbook("hn_expert.json")
"""

import asyncio
from typing import Optional, Any, Callable
from pathlib import Path

try:
    from browser_use import Agent, Browser

    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False
    Agent = None
    Browser = None

from ..llm_providers import LiteLLMClient
from ..playbook import Playbook
from ..roles import Reflector, Curator, GeneratorOutput
from .base import wrap_playbook_context


class ACEAgent:
    """
    Browser-use Agent with ACE learning capabilities.

    Drop-in replacement for browser-use Agent that automatically:
    - Injects learned strategies into tasks
    - Reflects on execution results
    - Updates playbook with new learnings

    Key difference from standard Agent:
    - No ACE Generator (browser-use executes directly)
    - Playbook provides context only
    - Reflector + Curator run AFTER execution

    Usage:
        # Simple usage
        agent = ACEAgent(
            llm=ChatBrowserUse(),      # Browser execution LLM
            ace_model="gpt-4o-mini"    # ACE learning LLM (default)
        )
        history = await agent.run(task="Find AI news")

        # Reuse across tasks (learns from each)
        agent = ACEAgent(llm=ChatBrowserUse())
        await agent.run(task="Task 1")
        await agent.run(task="Task 2")  # Uses Task 1 learnings
        agent.save_playbook("expert.json")

        # Start with existing knowledge
        agent = ACEAgent(
            llm=ChatBrowserUse(),
            playbook_path="expert.json"
        )
        await agent.run(task="New task")

        # Disable learning for debugging
        agent = ACEAgent(
            llm=ChatBrowserUse(),
            playbook_path="expert.json",
            is_learning=False
        )
        await agent.run(task="Test task")
    """

    def __init__(
        self,
        task: Optional[str] = None,
        llm: Any = None,
        browser: Optional[Any] = None,
        ace_model: str = "gpt-4o-mini",
        ace_llm: Optional[LiteLLMClient] = None,
        playbook: Optional[Playbook] = None,
        playbook_path: Optional[str] = None,
        is_learning: bool = True,
        **agent_kwargs,
    ):
        """
        Initialize ACEAgent.

        Args:
            task: Browser automation task (can also be set in run())
            llm: LLM for browser-use execution (ChatOpenAI, ChatBrowserUse, etc.)
            browser: Browser instance (optional, created automatically if None)
            ace_model: Model name for ACE learning (Reflector/Curator)
            ace_llm: Custom LLM client for ACE (overrides ace_model)
            playbook: Existing Playbook instance
            playbook_path: Path to load playbook from
            is_learning: Enable/disable ACE learning
            **agent_kwargs: Additional browser-use Agent parameters
                (max_steps, use_vision, step_timeout, max_failures, etc.)
        """
        if not BROWSER_USE_AVAILABLE:
            raise ImportError(
                "browser-use is not installed. Install with: "
                "pip install ace-framework[browser-use]"
            )

        self.task = task
        self.browser_llm = llm
        self.browser = browser
        self.is_learning = is_learning
        self.agent_kwargs = agent_kwargs

        # Always create playbook and ACE components
        # (but only use them if is_learning=True)

        # Load or create playbook
        if playbook_path:
            self.playbook = Playbook.load_from_file(playbook_path)
        elif playbook:
            self.playbook = playbook
        else:
            self.playbook = Playbook()

        # Create ACE LLM (for Reflector/Curator, NOT execution)
        self.ace_llm = ace_llm or LiteLLMClient(model=ace_model)

        # Create ACE learning components (NO GENERATOR!)
        self.reflector = Reflector(self.ace_llm)
        self.curator = Curator(self.ace_llm)

    async def run(
        self,
        task: Optional[str] = None,
        max_steps: Optional[int] = None,
        on_step_start: Optional[Callable] = None,
        on_step_end: Optional[Callable] = None,
        **run_kwargs,
    ):
        """
        Run browser automation task with ACE learning.

        Args:
            task: Task to execute (overrides constructor task)
            max_steps: Maximum steps (overrides agent_kwargs)
            on_step_start: Lifecycle hook
            on_step_end: Lifecycle hook
            **run_kwargs: Additional run() parameters

        Returns:
            Browser-use history object
        """
        # Determine task
        current_task = task or self.task
        if not current_task:
            raise ValueError("Task must be provided either in constructor or run()")

        # Get learned strategies if learning enabled and playbook has bullets
        if self.is_learning and self.playbook and self.playbook.bullets():
            playbook_context = wrap_playbook_context(self.playbook)
            # Inject strategies into task
            enhanced_task = f"""{current_task}

{playbook_context}"""
        else:
            enhanced_task = current_task

        # Build Agent parameters
        agent_params = {
            **self.agent_kwargs,
            "task": enhanced_task,
            "llm": self.browser_llm,
        }

        if self.browser:
            agent_params["browser"] = self.browser

        if max_steps:
            agent_params["max_steps"] = max_steps

        # Create browser-use Agent
        agent = Agent(**agent_params)

        # Execute browser task
        success = False
        error = None
        try:
            history = await agent.run(
                on_step_start=on_step_start, on_step_end=on_step_end, **run_kwargs
            )
            success = True

            # Learn from successful execution (only if is_learning=True)
            if self.is_learning:
                await self._learn_from_execution(current_task, history, success=True)

            return history

        except Exception as e:
            error = str(e)
            # Learn from failure too (only if is_learning=True)
            if self.is_learning:
                await self._learn_from_execution(
                    current_task,
                    history if "history" in locals() else None,
                    success=False,
                    error=error,
                )
            raise

    async def _learn_from_execution(
        self, task: str, history: Any, success: bool, error: Optional[str] = None
    ):
        """
        Run ACE learning pipeline AFTER browser execution.

        Flow: Reflector → Curator → Update Playbook
        (No Generator - browser-use already executed)
        """
        # Extract execution details
        if history:
            try:
                output = (
                    history.final_result()
                    if hasattr(history, "final_result")
                    else str(history)
                )
            except:
                output = ""

            try:
                steps = (
                    history.number_of_steps()
                    if hasattr(history, "number_of_steps")
                    else (
                        len(history.action_names())
                        if hasattr(history, "action_names") and history.action_names()
                        else 0
                    )
                )
            except:
                steps = 0
        else:
            output = ""
            steps = 0

        # Create GeneratorOutput (browser executed, not ACE Generator)
        # This is a "fake" output to satisfy Reflector's interface
        generator_output = GeneratorOutput(
            reasoning=f"Browser automation task: {task}",
            final_answer=output,
            bullet_ids=[],  # Browser-use didn't use Generator
            raw={"steps": steps, "success": success, "execution_mode": "browser-use"},
        )

        # Build feedback
        if success:
            feedback = (
                f"Browser task completed successfully in {steps} steps.\n"
                f"Output: {output[:200]}{'...' if len(output) > 200 else ''}"
            )
        else:
            feedback = f"Browser task failed after {steps} steps.\n" f"Error: {error}"

        # Run Reflector
        reflection = self.reflector.reflect(
            question=task,
            generator_output=generator_output,
            playbook=self.playbook,
            ground_truth=None,
            feedback=feedback,
        )

        # Run Curator
        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=(
                f"task: {task}\n"
                f"feedback: {feedback}\n"
                f"success: {success}\n"
                f"steps: {steps}"
            ),
            progress=f"Browser task: {task}",
        )

        # Update playbook with learned strategies
        self.playbook.apply_delta(curator_output.delta)

    def enable_learning(self):
        """Enable ACE learning."""
        self.is_learning = True

    def disable_learning(self):
        """Disable ACE learning (execution only, no updates to playbook)."""
        self.is_learning = False

    def save_playbook(self, path: str):
        """Save learned playbook to file."""
        self.playbook.save_to_file(path)

    def load_playbook(self, path: str):
        """Load playbook from file."""
        self.playbook = Playbook.load_from_file(path)

    def get_strategies(self) -> str:
        """Get current playbook strategies as formatted text."""
        if not self.playbook:
            return ""
        return wrap_playbook_context(self.playbook)


# Export for integration module
__all__ = ["ACEAgent", "BROWSER_USE_AVAILABLE"]
