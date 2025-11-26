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
from typing import Optional, Any, Callable, List
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
from ..prompts_v2_1 import PromptManager
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
        ace_max_tokens: int = 2048,
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
            ace_max_tokens: Max tokens for ACE learning LLM (default: 2048).
                Reflector typically needs 400-800 tokens for analysis.
                Curator typically needs 300-1000 tokens for delta operations.
                Increase for complex tasks with long execution histories.
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
        self.ace_llm = ace_llm or LiteLLMClient(
            model=ace_model, max_tokens=ace_max_tokens
        )

        # Create ACE learning components with v2.1 prompts (NO GENERATOR!)
        prompt_mgr = PromptManager()
        self.reflector = Reflector(
            self.ace_llm, prompt_template=prompt_mgr.get_reflector_prompt()
        )
        self.curator = Curator(
            self.ace_llm, prompt_template=prompt_mgr.get_curator_prompt()
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

    def _build_rich_feedback(
        self, history: Any, success: bool, error: Optional[str] = None
    ) -> dict:
        """
        Extract comprehensive trace information from browser-use history.

        Returns dict with:
        - feedback: formatted feedback string for Reflector
        - raw_trace: structured trace data for GeneratorOutput.raw
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
        Extract cited bullet IDs from browser-use agent thoughts.

        Parses only the agent's reasoning (model_thoughts), filtering out
        tool calls, action results, and other noise.

        Args:
            history: Browser-use AgentHistoryList

        Returns:
            List of cited bullet IDs found in agent thoughts
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
            from ..roles import extract_cited_bullet_ids

            return extract_cited_bullet_ids(thoughts_text)
        except Exception:
            # Graceful degradation if extraction fails
            return []

    async def _learn_from_execution(
        self, task: str, history: Any, success: bool, error: Optional[str] = None
    ):
        """
        Run ACE learning pipeline AFTER browser execution.

        Flow: Reflector → Curator → Update Playbook
        (No Generator - browser-use already executed)
        """
        # Extract rich trace information
        trace_info = self._build_rich_feedback(history, success, error)

        # Extract cited bullet IDs from agent thoughts (clean, no tool noise)
        cited_ids = self._extract_cited_ids_from_history(history)

        # Filter out invalid bullet IDs (ones that don't exist in playbook)
        # This prevents errors from hallucinated or malformed citations
        valid_cited_ids = [
            bullet_id
            for bullet_id in cited_ids
            if self.playbook.get_bullet(bullet_id) is not None
        ]

        # Create GeneratorOutput (browser executed, not ACE Generator)
        # This is a "fake" output to satisfy Reflector's interface
        # IMPORTANT: Pass full trace as reasoning so Reflector can analyze agent's thoughts
        generator_output = GeneratorOutput(
            reasoning=trace_info[
                "feedback"
            ],  # Full chronological trace with thoughts/actions/results
            final_answer=trace_info["output"],
            bullet_ids=valid_cited_ids,  # Filtered to only valid IDs that exist in playbook
            raw={
                "steps": trace_info["steps"],
                "success": success,
                "execution_mode": "browser-use",
                "trace": trace_info["raw_trace"],  # Include full trace
                "cited_strategies": cited_ids,  # Include for debugging
            },
        )

        # Build concise feedback summary (success/error context)
        # Full trace is already in generator_output.reasoning
        status = "succeeded" if success else "failed"
        feedback_summary = f"Browser task {status} in {trace_info['steps']} steps"
        if "duration_seconds" in trace_info["raw_trace"]:
            feedback_summary += f" ({trace_info['raw_trace']['duration_seconds']}s)"
        if error:
            feedback_summary += f"\nError: {error}"

        # Run Reflector
        reflection = self.reflector.reflect(
            question=task,
            generator_output=generator_output,
            playbook=self.playbook,
            ground_truth=None,
            feedback=feedback_summary,
        )

        # Run Curator with enriched context
        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=(
                f"task: {task}\n"
                f"feedback: {feedback_summary}\n"
                f"success: {success}\n"
                f"steps: {trace_info['steps']}\n"
                f"duration: {trace_info['raw_trace'].get('duration_seconds', 'N/A')}s"
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
