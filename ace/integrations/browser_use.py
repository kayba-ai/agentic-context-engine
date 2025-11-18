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

        # Extract rich trace data
        trace_data = {}

        # 1. Agent thoughts (reasoning at each step)
        try:
            if hasattr(history, "model_thoughts"):
                thoughts = history.model_thoughts()
                if thoughts:
                    # Keep ALL thoughts - no filtering
                    trace_data["thoughts"] = [
                        {
                            "thinking": t.thinking,
                            "evaluation": t.evaluation_previous_goal,
                            "memory": t.memory,
                            "next_goal": t.next_goal,
                        }
                        for t in thoughts
                    ]
        except Exception as e:
            trace_data["thoughts_error"] = str(e)  # type: ignore[assignment]

        # 2. Actions taken
        try:
            if hasattr(history, "model_actions"):
                actions = history.model_actions()
                if actions:
                    # Keep ALL actions - no filtering
                    trace_data["actions"] = actions
        except Exception as e:
            trace_data["actions_error"] = str(e)  # type: ignore[assignment]

        # 3. URLs visited
        try:
            if hasattr(history, "urls"):
                urls = history.urls()
                trace_data["urls"] = list(filter(None, urls))  # ALL URLs
        except Exception as e:
            trace_data["urls_error"] = str(e)  # type: ignore[assignment]

        # 4. Errors encountered
        try:
            if hasattr(history, "errors"):
                errors = history.errors()
                step_errors = [e for e in errors if e]
                if step_errors:
                    trace_data["step_errors"] = step_errors  # ALL errors
        except Exception as e:
            trace_data["errors_error"] = str(e)  # type: ignore[assignment]

        # 5. Action results
        try:
            if hasattr(history, "action_results"):
                results = history.action_results()
                if results:
                    # Extract key info from ALL results - no filtering
                    trace_data["action_results"] = [
                        {
                            "is_done": r.is_done,
                            "success": r.success,
                            "error": r.error,
                            "extracted_content": r.extracted_content,  # Full content
                        }
                        for r in results
                    ]
        except Exception as e:
            trace_data["action_results_error"] = str(e)  # type: ignore[assignment]

        # 6. Duration
        try:
            if hasattr(history, "total_duration_seconds"):
                trace_data["duration_seconds"] = round(
                    history.total_duration_seconds(), 2
                )
        except:
            pass

        # Build comprehensive feedback string
        feedback_parts = []

        # Overall status
        status = "succeeded" if success else "failed"
        feedback_parts.append(f"Browser task {status} in {steps} steps")

        # Add duration if available
        if "duration_seconds" in trace_data:
            feedback_parts.append(f"Duration: {trace_data['duration_seconds']}s")

        # Add thought summary
        if "thoughts" in trace_data and trace_data["thoughts"]:
            feedback_parts.append("\nAgent Reasoning:")
            for i, thought in enumerate(trace_data["thoughts"][:3], 1):  # First 3
                goal = thought.get("next_goal", "")[:100]
                if goal:
                    feedback_parts.append(f"  Step {i}: {goal}")

        # Add action summary
        if "actions" in trace_data and trace_data["actions"]:
            action_summary = ", ".join(
                list(action.keys())[0] for action in trace_data["actions"][:5] if action
            )
            feedback_parts.append(f"\nActions taken: {action_summary}")

        # Add URLs
        if "urls" in trace_data and trace_data["urls"]:
            feedback_parts.append(
                f"URLs visited: {', '.join(trace_data['urls'][:3])}"  # type: ignore[arg-type]
            )

        # Add errors
        if "step_errors" in trace_data:
            feedback_parts.append(
                f"\nErrors encountered: {'; '.join(trace_data['step_errors'][:2])}"  # type: ignore[arg-type]
            )

        # Add final output
        if output:
            output_preview = output[:150] + ("..." if len(output) > 150 else "")
            feedback_parts.append(f"\nFinal output: {output_preview}")

        # Add error if failed
        if error:
            feedback_parts.append(f"\nFailure reason: {error}")

        # Build detailed execution logs (for Reflector analysis)
        execution_log_parts = []

        # Add detailed thoughts
        if "thoughts" in trace_data and trace_data["thoughts"]:
            execution_log_parts.append("\n=== Agent Reasoning (Detailed) ===")
            for i, thought in enumerate(trace_data["thoughts"], 1):
                execution_log_parts.append(f"\nThought {i}:")
                if thought.get("thinking"):
                    execution_log_parts.append(f"  Thinking: {thought['thinking']}")
                if thought.get("evaluation"):
                    execution_log_parts.append(f"  Evaluation: {thought['evaluation']}")
                if thought.get("next_goal"):
                    execution_log_parts.append(f"  Next Goal: {thought['next_goal']}")
                if thought.get("memory"):
                    execution_log_parts.append(f"  Memory: {thought['memory']}")

        # Add detailed actions
        if "actions" in trace_data and trace_data["actions"]:
            execution_log_parts.append("\n=== Actions Taken ===")
            for i, action in enumerate(trace_data["actions"], 1):
                execution_log_parts.append(f"Action {i}: {action}")

        # Add detailed action results
        if "action_results" in trace_data and trace_data["action_results"]:
            execution_log_parts.append("\n=== Action Results ===")
            for i, result in enumerate(trace_data["action_results"], 1):
                result_parts = []
                if "is_done" in result:
                    result_parts.append(f"is_done={result['is_done']}")
                if "success" in result:
                    result_parts.append(f"success={result['success']}")
                if result.get("error"):
                    result_parts.append(f"error={result['error']}")
                if result.get("extracted_content"):
                    result_parts.append(f"content={result['extracted_content']}")
                execution_log_parts.append(f"Result {i}: {', '.join(result_parts)}")

        # Add URLs (full list)
        if "urls" in trace_data and trace_data["urls"]:
            execution_log_parts.append("\n=== URLs Visited ===")
            for i, url in enumerate(trace_data["urls"], 1):
                execution_log_parts.append(f"{i}. {url}")

        # Add errors (full list)
        if "step_errors" in trace_data and trace_data["step_errors"]:
            execution_log_parts.append("\n=== Errors Encountered ===")
            for i, err in enumerate(trace_data["step_errors"], 1):
                execution_log_parts.append(f"{i}. {err}")

        # Append detailed logs to feedback if available
        if execution_log_parts:
            feedback_parts.append("\n\n=== BROWSER EXECUTION DETAILS ===")
            feedback_parts.extend(execution_log_parts)
            feedback_parts.append("\n=== END EXECUTION DETAILS ===")

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

        # Create GeneratorOutput (browser executed, not ACE Generator)
        # This is a "fake" output to satisfy Reflector's interface
        generator_output = GeneratorOutput(
            reasoning=f"Browser automation task: {task}",
            final_answer=trace_info["output"],
            bullet_ids=cited_ids,  # Extracted from agent thoughts
            raw={
                "steps": trace_info["steps"],
                "success": success,
                "execution_mode": "browser-use",
                "trace": trace_info["raw_trace"],  # Include full trace
                "cited_strategies": cited_ids,  # Include for debugging
            },
        )

        # Use rich feedback
        feedback = trace_info["feedback"]

        # Run Reflector
        reflection = self.reflector.reflect(
            question=task,
            generator_output=generator_output,
            playbook=self.playbook,
            ground_truth=None,
            feedback=feedback,
        )

        # Run Curator with enriched context
        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=(
                f"task: {task}\n"
                f"feedback: {feedback}\n"
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
