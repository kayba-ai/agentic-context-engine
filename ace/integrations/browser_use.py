"""
Browser-use integration for ACE framework.

This module provides ACEAgent, a drop-in replacement for browser-use Agent
that automatically learns from execution feedback.

This is the reference implementation for ACE integrations with external agentic
frameworks. It demonstrates the pattern:
1. External framework (browser-use) executes task
2. ACE injects skillbook context beforehand
3. ACE learns from execution afterward (Reflector + SkillManager)

Example:
    from ace.integrations import ACEAgent
    from browser_use import ChatBrowserUse

    agent = ACEAgent(llm=ChatBrowserUse())
    await agent.run(task="Find top HN post")
    agent.save_skillbook("hn_expert.json")
"""

import asyncio
from typing import TYPE_CHECKING, Optional, Any, Callable, Dict, List
from pathlib import Path

try:
    from browser_use import Agent, Browser

    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False
    Agent = None  # type: ignore[misc,assignment]
    Browser = None  # type: ignore[misc,assignment]

from ..llm_providers import LiteLLMClient
from ..skillbook import Skillbook
from ..roles import Reflector, SkillManager, AgentOutput
from ..prompts_v2_1 import PromptManager
from .base import wrap_skillbook_context

if TYPE_CHECKING:
    from ..deduplication import DeduplicationConfig


class ACEAgent:
    """
    Browser-use Agent with ACE learning capabilities.

    Drop-in replacement for browser-use Agent that automatically:
    - Injects learned strategies into tasks
    - Reflects on execution results
    - Updates skillbook with new learnings

    Key difference from standard Agent:
    - No ACE Agent (browser-use executes directly)
    - Skillbook provides context only
    - Reflector + SkillManager run AFTER execution

    Insight Level: Meso
        ACE sees the full browser execution trace (thoughts, actions, observations)
        without external ground truth. Learns from execution patterns rather than
        correctness feedback. See docs/COMPLETE_GUIDE_TO_ACE.md for details.

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
        agent.save_skillbook("expert.json")

        # Start with existing knowledge
        agent = ACEAgent(
            llm=ChatBrowserUse(),
            skillbook_path="expert.json"
        )
        await agent.run(task="New task")

        # Disable learning for debugging
        agent = ACEAgent(
            llm=ChatBrowserUse(),
            skillbook_path="expert.json",
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
        ace_max_tokens: int = 2048,
        skillbook: Optional[Skillbook] = None,
        skillbook_path: Optional[str] = None,
        is_learning: bool = True,
        async_learning: bool = False,
        dedup_config: Optional["DeduplicationConfig"] = None,
        **agent_kwargs,
    ):
        """
        Initialize ACEAgent.

        Args:
            task: Browser automation task (can also be set in run())
            llm: LLM for browser-use execution (ChatOpenAI, ChatBrowserUse, etc.)
            browser: Browser instance (optional, created automatically if None)
            ace_model: Model name for ACE learning (Reflector/SkillManager)
            ace_llm: Custom LLM client for ACE (overrides ace_model)
            ace_max_tokens: Max tokens for ACE learning LLM (default: 2048).
                Reflector typically needs 400-800 tokens for analysis.
                SkillManager typically needs 300-1000 tokens for update operations.
                Increase for complex tasks with long execution histories.
            skillbook: Existing Skillbook instance
            skillbook_path: Path to load skillbook from
            is_learning: Enable/disable ACE learning
            async_learning: If True, learning happens in background (non-blocking).
                Use wait_for_learning() before saving skillbook.
            dedup_config: Optional DeduplicationConfig for skill deduplication
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
        self._async_learning = async_learning
        self.agent_kwargs = agent_kwargs

        # Async learning task tracking
        self._learning_tasks: List[asyncio.Task] = []

        # Always create skillbook and ACE components
        # (but only use them if is_learning=True)

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

        # Get learned strategies if learning enabled and skillbook has skills
        if self.is_learning and self.skillbook and self.skillbook.skills():
            skillbook_context = wrap_skillbook_context(self.skillbook)
            # Inject strategies into task
            enhanced_task = f"""{current_task}

{skillbook_context}"""
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
                if self._async_learning:
                    # Fire and forget - learning in background
                    learning_task = asyncio.create_task(
                        self._learn_from_execution(current_task, history, success=True)
                    )
                    self._learning_tasks.append(learning_task)
                else:
                    # Sync mode - wait for learning
                    await self._learn_from_execution(
                        current_task, history, success=True
                    )

            return history

        except Exception as e:
            error = str(e)
            # Learn from failure too (only if is_learning=True)
            if self.is_learning:
                if self._async_learning:
                    learning_task = asyncio.create_task(
                        self._learn_from_execution(
                            current_task,
                            history if "history" in locals() else None,
                            success=False,
                            error=error,
                        )
                    )
                    self._learning_tasks.append(learning_task)
                else:
                    await self._learn_from_execution(
                        current_task,
                        history if "history" in locals() else None,
                        success=False,
                        error=error,
                    )
            raise

    def _build_rich_feedback(
        self, history: Any, success: bool, error: Optional[str] = None
    ) -> dict:
        """
        Extract comprehensive trace information from browser-use history.

        Returns dict with:
        - feedback: formatted feedback string for Reflector
        - raw_trace: structured trace data for AgentOutput.raw
        - steps: number of steps executed
        - output: final output from execution
        """
        if not history:
            return {
                "feedback": (
                    f"Task failed: {error}" if error else "No execution history"
                ),
                "raw_trace": {},
                "steps": 0,
                "output": "",
            }

        # Extract basic info
        try:
            output = (
                history.final_result() if hasattr(history, "final_result") else ""
            ) or ""
        except:
            output = ""

        try:
            steps = (
                history.number_of_steps() if hasattr(history, "number_of_steps") else 0
            )
        except:
            steps = 0

        # Extract rich trace data in CHRONOLOGICAL order
        # Use history.history to get step-by-step execution
        trace_data: dict = {}
        chronological_steps: list = []

        try:
            if hasattr(history, "history"):
                # Iterate through each step in chronological order
                for step_idx, step in enumerate(history.history, 1):
                    step_data: dict = {"step_number": step_idx}

                    # Extract thought from model_output
                    if step.model_output:
                        step_data["thought"] = {
                            "thinking": step.model_output.thinking,
                            "evaluation": step.model_output.evaluation_previous_goal,
                            "memory": step.model_output.memory,
                            "next_goal": step.model_output.next_goal,
                        }

                        # Extract action(s) from model_output
                        if step.model_output.action:
                            step_data["actions"] = [
                                {k: v for k, v in action.model_dump().items()}
                                for action in step.model_output.action
                            ]

                    # Extract result(s)
                    if step.result:
                        step_data["results"] = [
                            {
                                "is_done": r.is_done,
                                "success": r.success,
                                "error": r.error,
                                "extracted_content": r.extracted_content,
                            }
                            for r in step.result
                        ]

                    # Extract state (URL, etc.)
                    if step.state:
                        step_data["url"] = step.state.url
                        if hasattr(step.state, "screenshot") and step.state.screenshot:
                            step_data["has_screenshot"] = True

                    chronological_steps.append(step_data)

                trace_data["chronological_steps"] = chronological_steps

            # Also extract duration
            if hasattr(history, "total_duration_seconds"):
                trace_data["duration_seconds"] = round(
                    history.total_duration_seconds(), 2
                )
        except Exception as e:
            trace_data["extraction_error"] = str(e)  # type: ignore[assignment]

        # Build comprehensive feedback string
        feedback_parts = []

        # Overall status
        status = "succeeded" if success else "failed"
        feedback_parts.append(f"Browser task {status} in {steps} steps")

        # Add duration if available
        if "duration_seconds" in trace_data:
            feedback_parts.append(f"Duration: {trace_data['duration_seconds']}s")

        # Add final output
        if output:
            output_preview = output[:150] + ("..." if len(output) > 150 else "")
            feedback_parts.append(f"\nFinal output: {output_preview}")

        # Add error if failed
        if error:
            feedback_parts.append(f"\nFailure reason: {error}")

        # Build CHRONOLOGICAL execution trace (for Reflector analysis)
        if "chronological_steps" in trace_data and trace_data["chronological_steps"]:
            feedback_parts.append("\n\n=== BROWSER EXECUTION TRACE (Chronological) ===")

            for step in trace_data["chronological_steps"]:
                step_num = step["step_number"]
                feedback_parts.append(f"\n--- Step {step_num} ---")

                # 1. Thought (what the agent was thinking)
                if "thought" in step:
                    thought = step["thought"]
                    if thought.get("thinking"):
                        feedback_parts.append(f"🧠 Thinking: {thought['thinking']}")
                    if thought.get("evaluation"):
                        feedback_parts.append(f"   Evaluation: {thought['evaluation']}")
                    if thought.get("memory"):
                        feedback_parts.append(f"   Memory: {thought['memory']}")
                    if thought.get("next_goal"):
                        feedback_parts.append(f"   Next Goal: {thought['next_goal']}")

                # 2. Action (what the agent did)
                if "actions" in step:
                    for action in step["actions"]:
                        action_name = list(action.keys())[0] if action else "unknown"
                        action_params = action.get(action_name, {})
                        feedback_parts.append(
                            f"▶️  Action: {action_name}({action_params})"
                        )

                # 3. Result (what happened)
                if "results" in step:
                    for result in step["results"]:
                        result_parts = []
                        if result.get("success") is not None:
                            result_parts.append(f"success={result['success']}")
                        if result.get("is_done") is not None:
                            result_parts.append(f"done={result['is_done']}")
                        if result.get("error"):
                            result_parts.append(f"error={result['error']}")
                        if result.get("extracted_content"):
                            content = str(result["extracted_content"])[:200]
                            result_parts.append(f"content={content}...")
                        feedback_parts.append(f"📊 Result: {', '.join(result_parts)}")

                # 4. URL (where the agent was)
                if "url" in step:
                    feedback_parts.append(f"🌐 URL: {step['url']}")

            feedback_parts.append("\n=== END EXECUTION TRACE ===")

        return {
            "feedback": "\n".join(feedback_parts),
            "raw_trace": trace_data,
            "steps": steps,
            "output": output,
        }

    def _extract_cited_ids_from_history(self, history: Any) -> List[str]:
        """
        Extract cited skill IDs from browser-use agent thoughts.

        Parses only the agent's reasoning (model_thoughts), filtering out
        tool calls, action results, and other noise.

        Args:
            history: Browser-use AgentHistoryList

        Returns:
            List of cited skill IDs found in agent thoughts
        """
        if not history or not hasattr(history, "model_thoughts"):
            return []

        try:
            thoughts = history.model_thoughts()
            # Extract only the thinking/reasoning text (filter noise)
            thoughts_text = "\n".join(
                t.thinking for t in thoughts if hasattr(t, "thinking") and t.thinking
            )

            # Use public utility to extract IDs
            from ..roles import extract_cited_skill_ids

            return extract_cited_skill_ids(thoughts_text)
        except Exception:
            # Graceful degradation if extraction fails
            return []

    async def _learn_from_execution(
        self, task: str, history: Any, success: bool, error: Optional[str] = None
    ):
        """
        Run ACE learning pipeline AFTER browser execution.

        Flow: Reflector → SkillManager → Update Skillbook
        (No ACE Agent - browser-use already executed)

        Uses asyncio.to_thread() to run sync LLM calls in a thread pool,
        preventing event loop blocking when async_learning=True.
        """
        # Extract rich trace information (fast, no LLM calls)
        trace_info = self._build_rich_feedback(history, success, error)

        # Extract cited skill IDs from agent thoughts (clean, no tool noise)
        cited_ids = self._extract_cited_ids_from_history(history)

        # Filter out invalid skill IDs (ones that don't exist in skillbook)
        # This prevents errors from hallucinated or malformed citations
        valid_cited_ids = [
            skill_id
            for skill_id in cited_ids
            if self.skillbook.get_skill(skill_id) is not None
        ]

        # Run sync learning in thread pool (doesn't block event loop)
        await asyncio.to_thread(
            self._sync_learn,
            task,
            trace_info,
            valid_cited_ids,
            cited_ids,
            success,
            error,
        )

    def _sync_learn(
        self,
        task: str,
        trace_info: Dict[str, Any],
        valid_cited_ids: List[str],
        cited_ids: List[str],
        success: bool,
        error: Optional[str],
    ):
        """
        Synchronous learning logic (runs in thread pool).

        This method contains the actual LLM calls (Reflector + SkillManager)
        which are synchronous and would block the event loop if called directly.
        """
        # Create AgentOutput (browser executed, not ACE Agent)
        # This is a "fake" output to satisfy Reflector's interface
        # IMPORTANT: Pass full trace as reasoning so Reflector can analyze agent's thoughts
        agent_output = AgentOutput(
            reasoning=trace_info[
                "feedback"
            ],  # Full chronological trace with thoughts/actions/results
            final_answer=trace_info["output"],
            skill_ids=valid_cited_ids,  # Filtered to only valid IDs that exist in skillbook
            raw={
                "steps": trace_info["steps"],
                "success": success,
                "execution_mode": "browser-use",
                "trace": trace_info["raw_trace"],  # Include full trace
                "cited_strategies": cited_ids,  # Include for debugging
            },
        )

        # Build concise feedback summary (success/error context)
        # Full trace is already in agent_output.reasoning
        status = "succeeded" if success else "failed"
        feedback_summary = f"Browser task {status} in {trace_info['steps']} steps"
        if "duration_seconds" in trace_info["raw_trace"]:
            feedback_summary += f" ({trace_info['raw_trace']['duration_seconds']}s)"
        if error:
            feedback_summary += f"\nError: {error}"

        # Run Reflector (sync LLM call)
        reflection = self.reflector.reflect(
            question=task,
            agent_output=agent_output,
            skillbook=self.skillbook,
            ground_truth=None,
            feedback=feedback_summary,
        )

        # Run SkillManager with enriched context (sync LLM call)
        skill_manager_output = self.skill_manager.update_skills(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context=(
                f"task: {task}\n"
                f"feedback: {feedback_summary}\n"
                f"success: {success}\n"
                f"steps: {trace_info['steps']}\n"
                f"duration: {trace_info['raw_trace'].get('duration_seconds', 'N/A')}s"
            ),
            progress=f"Browser task: {task}",
        )

        # Update skillbook with learned strategies
        self.skillbook.apply_update(skill_manager_output.update)

    def enable_learning(self):
        """Enable ACE learning."""
        self.is_learning = True

    def disable_learning(self):
        """Disable ACE learning (execution only, no updates to skillbook)."""
        self.is_learning = False

    # -----------------------------------------------------------------------
    # Async Learning Control
    # -----------------------------------------------------------------------

    async def wait_for_learning(self, timeout: Optional[float] = None) -> bool:
        """Wait for all background learning tasks to complete.

        Args:
            timeout: Max seconds to wait (None = wait forever)

        Returns:
            True if all tasks completed, False if timeout
        """
        if not self._learning_tasks:
            return True

        # Clean up completed tasks first
        self._learning_tasks = [t for t in self._learning_tasks if not t.done()]

        if not self._learning_tasks:
            return True

        try:
            await asyncio.wait_for(
                asyncio.gather(*self._learning_tasks, return_exceptions=True),
                timeout=timeout,
            )
            self._learning_tasks.clear()
            return True
        except asyncio.TimeoutError:
            return False

    @property
    def learning_stats(self) -> Dict[str, Any]:
        """Get learning progress statistics.

        Returns:
            Dict with tasks_submitted, pending, completed counts
        """
        # Clean up completed tasks
        pending = [t for t in self._learning_tasks if not t.done()]
        completed = len(self._learning_tasks) - len(pending)

        return {
            "tasks_submitted": len(self._learning_tasks),
            "pending": len(pending),
            "completed": completed,
            "async_learning": self._async_learning,
        }

    def stop_async_learning(self):
        """Cancel all pending learning tasks."""
        for task in self._learning_tasks:
            if not task.done():
                task.cancel()
        self._learning_tasks.clear()

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


# Export for integration module
__all__ = ["ACEAgent", "BROWSER_USE_AVAILABLE"]
