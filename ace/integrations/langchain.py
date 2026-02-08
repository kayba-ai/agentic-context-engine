"""
ACE + LangChain integration for learning from chain/agent execution.

This module provides ACELangChain, a wrapper that adds ACE learning capabilities
to any LangChain Runnable (chains, agents, LangGraph graphs, custom runnables).

Supported Runnable Types:
- Simple chains: prompt | llm patterns
- AgentExecutor: Tool-calling agents with intermediate step tracing
- LangGraph CompiledStateGraph: Modern LangGraph agents (create_react_agent, etc.)
- Custom runnables: Any LangChain Runnable

When to Use ACELangChain:
- Complex workflows: Multi-step LangChain chains
- Tool-using agents: LangChain agents with tools
- LangGraph agents: create_react_agent, create_tool_calling_agent, StateGraph
- Custom runnables: Your own LangChain components
- Production workflows: LangChain orchestration with learning

When NOT to Use ACELangChain:
- Simple Q&A → Use ACELiteLLM
- Browser automation → Use ACEAgent (browser-use)
- Custom agent (non-LangChain) → Use integration pattern (see docs)

Example (LangChain chain):
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    from ace.integrations import ACELangChain

    # Create your LangChain chain
    prompt = PromptTemplate.from_template("Answer: {question}")
    llm = ChatOpenAI(temperature=0)
    chain = prompt | llm

    # Wrap with ACE learning
    ace_chain = ACELangChain(runnable=chain)

    # Use it (learns automatically)
    result = ace_chain.invoke({"question": "What is ACE?"})

    # Save learned strategies
    ace_chain.save_skillbook("chain_expert.json")

Example (LangGraph agent):
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage
    from langgraph.prebuilt import create_react_agent
    from ace.integrations import ACELangChain

    # Create LangGraph agent
    llm = ChatOpenAI(model="gpt-4")
    tools = [...]
    graph = create_react_agent(llm, tools)

    # Wrap with ACE learning
    ace_agent = ACELangChain(runnable=graph)

    # Use with message format (LangGraph I/O)
    result = ace_agent.invoke({"messages": [HumanMessage(content="What is 2+2?")]})
    # Returns: "4" (extracted from AIMessage)

    # ACE automatically extracts traces from message history for learning
"""

from typing import TYPE_CHECKING, Any, Optional, Dict, Callable, List
import asyncio
import logging

from ..skillbook import Skillbook
from ..roles import Reflector, SkillManager, AgentOutput
from ..prompt_manager import PromptManager
from .base import wrap_skillbook_context

if TYPE_CHECKING:
    from ..deduplication import DeduplicationConfig

try:
    from langchain_core.runnables import Runnable

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Runnable = None  # type: ignore

# Try to import AgentExecutor for rich tracing
try:
    from langchain.agents import AgentExecutor

    AGENT_EXECUTOR_AVAILABLE = True
except ImportError:
    AGENT_EXECUTOR_AVAILABLE = False
    AgentExecutor = None  # type: ignore

# Try to import LangGraph CompiledStateGraph for LangGraph agent support
try:
    from langgraph.graph.state import CompiledStateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    CompiledStateGraph = None  # type: ignore

logger = logging.getLogger(__name__)


class ACELangChain:
    """
    LangChain Runnable wrapper with ACE learning.

    Wraps any LangChain Runnable (chain, agent, custom) and adds ACE learning
    capabilities. The runnable executes normally, but ACE learns from results
    to improve future executions.

    Pattern:
        1. INJECT: Add skillbook context to input
        2. EXECUTE: Run the LangChain runnable
        3. LEARN: Update skillbook via Reflector + SkillManager

    Attributes:
        runnable: The LangChain Runnable being wrapped
        skillbook: Learned strategies (Skillbook instance)
        is_learning: Whether learning is enabled

    Example:
        # Basic usage
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template("Answer: {input}")
        chain = prompt | ChatOpenAI(temperature=0)

        ace_chain = ACELangChain(runnable=chain)
        result = ace_chain.invoke({"input": "What is 2+2?"})

        # With existing skillbook
        ace_chain = ACELangChain(
            runnable=chain,
            skillbook_path="expert.json"
        )

        # Async execution
        result = await ace_chain.ainvoke({"input": "Question"})
    """

    def __init__(
        self,
        runnable: Any,
        ace_model: str = "gpt-4o-mini",
        skillbook_path: Optional[str] = None,
        is_learning: bool = True,
        async_learning: bool = False,
        output_parser: Optional[Callable[[Any], str]] = None,
        dedup_config: Optional["DeduplicationConfig"] = None,
    ):
        """
        Initialize ACELangChain wrapper.

        Args:
            runnable: LangChain Runnable (chain, agent, custom)
            ace_model: Model for ACE learning (Reflector/SkillManager)
            skillbook_path: Path to existing skillbook (optional)
            is_learning: Enable/disable learning (default: True)
            async_learning: Run learning in background for ainvoke() (default: False)
                           When True, ainvoke() returns immediately while
                           Reflector/SkillManager process in background.
            output_parser: Custom function to parse runnable output to string
                          (default: converts to string)
            dedup_config: Optional DeduplicationConfig for skill deduplication

        Raises:
            ImportError: If LangChain is not installed

        Example:
            # Basic
            ace_chain = ACELangChain(my_chain)

            # With async learning
            ace_chain = ACELangChain(my_chain, async_learning=True)
            result = await ace_chain.ainvoke(input)
            # Result returns immediately, learning continues in background
            await ace_chain.wait_for_learning()

            # With custom output parser
            def parse_output(result):
                return result["output"]["final_answer"]

            ace_chain = ACELangChain(
                my_chain,
                output_parser=parse_output
            )

            # With deduplication
            from ace import DeduplicationConfig
            ace_chain = ACELangChain(
                my_chain,
                dedup_config=DeduplicationConfig(similarity_threshold=0.85)
            )
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install with:\n"
                "pip install ace-framework[langchain]\n"
                "or: pip install langchain-core"
            )

        self.runnable = runnable
        self.is_learning = is_learning
        self._async_learning = async_learning
        self._learning_tasks: List[asyncio.Task] = []
        self._tasks_submitted_count: int = 0
        self._tasks_completed_count: int = 0
        self.output_parser = output_parser or self._default_output_parser

        # Load or create skillbook
        if skillbook_path:
            self.skillbook = Skillbook.load_from_file(skillbook_path)
        else:
            self.skillbook = Skillbook()

        # Setup ACE learning components
        try:
            from ..llm_providers import LiteLLMClient
        except ImportError:
            raise ImportError(
                "ACELangChain requires LiteLLM. Install with:\n"
                "pip install ace-framework  # (includes LiteLLM by default)"
            )

        self.llm = LiteLLMClient(model=ace_model, max_tokens=2048)

        # Create ACE components with v2.1 prompts
        prompt_mgr = PromptManager()
        self.reflector = Reflector(
            self.llm, prompt_template=prompt_mgr.get_reflector_prompt()
        )

        # Create DeduplicationManager if config provided
        dedup_manager = None
        if dedup_config is not None:
            from ..deduplication import DeduplicationManager

            dedup_manager = DeduplicationManager(dedup_config)

        self.skill_manager = SkillManager(
            self.llm,
            prompt_template=prompt_mgr.get_skill_manager_prompt(),
            dedup_manager=dedup_manager,
        )

    def invoke(self, input: Any, **kwargs) -> Any:
        """
        Execute runnable with ACE learning (sync).

        For AgentExecutor runnables, automatically enables intermediate step
        extraction to provide richer reasoning traces to the Reflector.

        For LangGraph CompiledStateGraph runnables, automatically handles
        message-based I/O format.

        Args:
            input: Input for the runnable (string, dict, etc.)
            **kwargs: Additional arguments passed to runnable.invoke()

        Returns:
            Output from the runnable

        Example:
            # String input
            result = ace_chain.invoke("What is ACE?")

            # Dict input
            result = ace_chain.invoke({"question": "What is ACE?"})

            # LangGraph message format
            result = ace_chain.invoke({"messages": [HumanMessage(content="What is ACE?")]})
        """
        # Step 1: Inject skillbook context
        enhanced_input = self._inject_context(input)

        # Step 2: Detect runnable type and configure
        is_agent = self._is_agent_executor()
        is_langgraph = self._is_langgraph()
        original_setting = False
        if is_agent:
            original_setting = getattr(
                self.runnable, "return_intermediate_steps", False
            )
            self.runnable.return_intermediate_steps = True

        # Step 3: Execute runnable
        try:
            result = self.runnable.invoke(enhanced_input, **kwargs)
        except Exception as e:
            logger.error(f"Error executing runnable: {e}")
            # Restore original setting
            if is_agent:
                self.runnable.return_intermediate_steps = original_setting
            # Learn from failure
            if self.is_learning:
                self._learn_from_failure(input, str(e))
            raise
        finally:
            # Restore original setting
            if is_agent:
                self.runnable.return_intermediate_steps = original_setting

        # Step 4: Learn from result
        if self.is_learning:
            if is_agent and isinstance(result, dict) and "intermediate_steps" in result:
                # Rich trace from AgentExecutor
                self._learn_with_trace(input, result)
            elif is_langgraph and isinstance(result, dict) and "messages" in result:
                # Rich trace from LangGraph message history
                self._learn_with_langgraph_trace(input, result)
            else:
                # Basic learning
                self._learn(input, result)

        # Step 5: Return clean output
        if is_agent and isinstance(result, dict) and "output" in result:
            return result["output"]
        if is_langgraph and isinstance(result, dict) and "messages" in result:
            return self._extract_langgraph_output(result)
        return result

    async def ainvoke(self, input: Any, **kwargs) -> Any:
        """
        Execute runnable with ACE learning (async).

        For AgentExecutor runnables, automatically enables intermediate step
        extraction to provide richer reasoning traces to the Reflector.

        For LangGraph CompiledStateGraph runnables, automatically handles
        message-based I/O format.

        Args:
            input: Input for the runnable (string, dict, etc.)
            **kwargs: Additional arguments passed to runnable.ainvoke()

        Returns:
            Output from the runnable

        Example:
            result = await ace_chain.ainvoke({"input": "Question"})

            # With async_learning=True, learning happens in background
            ace_chain = ACELangChain(chain, async_learning=True)
            result = await ace_chain.ainvoke({"input": "Question"})
            # Result returns immediately
            await ace_chain.wait_for_learning()  # Wait for learning to complete

            # LangGraph message format
            result = await ace_chain.ainvoke({"messages": [HumanMessage(content="Question")]})
        """
        # Step 1: Inject skillbook context
        enhanced_input = self._inject_context(input)

        # Step 2: Detect runnable type and configure
        is_agent = self._is_agent_executor()
        is_langgraph = self._is_langgraph()
        original_setting = False
        if is_agent:
            original_setting = getattr(
                self.runnable, "return_intermediate_steps", False
            )
            self.runnable.return_intermediate_steps = True

        # Step 3: Execute runnable (async)
        try:
            result = await self.runnable.ainvoke(enhanced_input, **kwargs)
        except Exception as e:
            logger.error(f"Error executing runnable: {e}")
            # Restore original setting
            if is_agent:
                self.runnable.return_intermediate_steps = original_setting
            # Learn from failure
            if self.is_learning:
                if self._async_learning:
                    task = asyncio.create_task(self._alearn_from_failure(input, str(e)))
                    task.add_done_callback(self._on_task_done)
                    self._learning_tasks.append(task)
                    self._tasks_submitted_count += 1
                else:
                    await self._alearn_from_failure(input, str(e))
            raise
        finally:
            # Restore original setting
            if is_agent:
                self.runnable.return_intermediate_steps = original_setting

        # Step 4: Learn from result
        if self.is_learning:
            # Determine which learning method to use
            use_agent_trace = (
                is_agent and isinstance(result, dict) and "intermediate_steps" in result
            )
            use_langgraph_trace = (
                is_langgraph and isinstance(result, dict) and "messages" in result
            )

            if self._async_learning:
                if use_agent_trace:
                    task = asyncio.create_task(self._alearn_with_trace(input, result))
                elif use_langgraph_trace:
                    task = asyncio.create_task(
                        self._alearn_with_langgraph_trace(input, result)
                    )
                else:
                    task = asyncio.create_task(self._alearn(input, result))
                task.add_done_callback(self._on_task_done)
                self._learning_tasks.append(task)
                self._tasks_submitted_count += 1
            else:
                if use_agent_trace:
                    await self._alearn_with_trace(input, result)
                elif use_langgraph_trace:
                    await self._alearn_with_langgraph_trace(input, result)
                else:
                    await self._alearn(input, result)

        # Step 5: Return clean output
        if is_agent and isinstance(result, dict) and "output" in result:
            return result["output"]
        if is_langgraph and isinstance(result, dict) and "messages" in result:
            return self._extract_langgraph_output(result)
        return result

    def _inject_context(self, input: Any) -> Any:
        """
        Add skillbook context to input.

        Handles common input formats:
        - String: Append skillbook context
        - Dict with "messages" key: Enhance first HumanMessage (LangGraph format)
        - Dict with "input" key: Enhance input field (AgentExecutor format)
        - Dict without: Add skillbook_context key
        - Other: Return unchanged (no skillbook strategies yet)

        Args:
            input: Original input

        Returns:
            Enhanced input with skillbook context
        """
        # No context if no strategies yet
        if not self.skillbook or not self.skillbook.skills():
            return input

        skillbook_context = wrap_skillbook_context(self.skillbook)

        # String input - append context
        if isinstance(input, str):
            return f"{input}\n\n{skillbook_context}"

        # LangGraph message format - enhance first HumanMessage
        if isinstance(input, dict) and "messages" in input:
            messages = input["messages"]
            if messages and hasattr(messages[0], "content"):
                enhanced_messages = list(messages)
                first_msg = enhanced_messages[0]
                enhanced_content = f"{skillbook_context}\n\n{first_msg.content}"
                # Create new message of the same type with enhanced content
                enhanced_messages[0] = type(first_msg)(content=enhanced_content)
                # Preserve other keys in the input dict
                return {
                    "messages": enhanced_messages,
                    **{k: v for k, v in input.items() if k != "messages"},
                }
            return input

        # Dict input with "input" key - enhance that field
        if isinstance(input, dict) and "input" in input:
            enhanced = input.copy()
            enhanced["input"] = f"{input['input']}\n\n{skillbook_context}"
            return enhanced

        # Dict input without "input" key - add skillbook_context key
        if isinstance(input, dict):
            enhanced = input.copy()
            enhanced["skillbook_context"] = skillbook_context
            return enhanced

        # Other types - return unchanged
        return input

    def _learn(self, original_input: Any, result: Any):
        """
        Learn from successful execution.

        Args:
            original_input: Original input to runnable
            result: Output from runnable
        """
        try:
            # Parse output to string
            output_str = self.output_parser(result)

            # Build task description
            if isinstance(original_input, str):
                task = original_input
            elif isinstance(original_input, dict):
                # Try common keys
                task = (
                    original_input.get("input")
                    or original_input.get("question")
                    or original_input.get("query")
                    or str(original_input)
                )
            else:
                task = str(original_input)

            # Create adapter for Reflector interface
            # Note: LangChain chains don't expose intermediate reasoning, so we
            # focus the learning on task patterns and output quality rather than
            # reasoning steps. This prevents the Reflector from generating
            # meta-strategies about the learning system itself.
            agent_output = AgentOutput(
                reasoning=f"""Question/Task: {task}

Chain Output: {output_str}

Note: This is an external LangChain chain execution. Learning should focus on task patterns and output quality, not internal system behavior.""",
                final_answer=output_str,
                skill_ids=[],  # LangChain runnables don't cite skills
                raw={"input": original_input, "output": result},
            )

            # Build feedback - focus on task outcome, not system behavior
            feedback = f"External chain completed for task: {task[:200]}"

            # Reflect: Analyze execution
            reflection = self.reflector.reflect(
                question=task,
                agent_output=agent_output,
                skillbook=self.skillbook,
                ground_truth=None,
                feedback=feedback,
            )

            # SkillManager: Generate skillbook updates
            skill_manager_output = self.skill_manager.update_skills(
                reflection=reflection,
                skillbook=self.skillbook,
                question_context=f"task: {task}",
                progress=task,
            )

            # Apply updates
            self.skillbook.apply_update(skill_manager_output.update)

        except Exception as e:
            logger.error(f"ACE learning failed: {e}")
            # Don't crash - continue without learning

    def _learn_from_failure(self, original_input: Any, error_msg: str):
        """
        Learn from execution failure.

        Args:
            original_input: Original input to runnable
            error_msg: Error message
        """
        try:
            # Build task description
            if isinstance(original_input, str):
                task = original_input
            elif isinstance(original_input, dict):
                task = (
                    original_input.get("input")
                    or original_input.get("question")
                    or str(original_input)
                )
            else:
                task = str(original_input)

            # Create adapter for Reflector interface
            # Note: Focus learning on task failure patterns, not system behavior
            agent_output = AgentOutput(
                reasoning=f"""Question/Task: {task}

Execution Result: FAILED
Error: {error_msg}

Note: This is an external LangChain chain execution that failed. Learning should focus on task patterns that may have caused the failure, not internal system behavior.""",
                final_answer=f"Failed: {error_msg}",
                skill_ids=[],
                raw={"input": original_input, "error": error_msg},
            )

            # Build failure feedback - focus on task, not system
            feedback = f"Chain execution failed for task: {task[:200]}. Error: {error_msg[:200]}"

            # Reflect on failure
            reflection = self.reflector.reflect(
                question=task,
                agent_output=agent_output,
                skillbook=self.skillbook,
                ground_truth=None,
                feedback=feedback,
            )

            # SkillManager: Learn from failure patterns
            skill_manager_output = self.skill_manager.update_skills(
                reflection=reflection,
                skillbook=self.skillbook,
                question_context=f"task: {task}",
                progress=f"Failed: {task}",
            )

            # Apply updates
            self.skillbook.apply_update(skill_manager_output.update)

        except Exception as e:
            logger.error(f"ACE failure learning failed: {e}")
            # Don't crash

    def _learn_with_trace(self, original_input: Any, result: Dict[str, Any]):
        """
        Learn from AgentExecutor result with intermediate_steps.

        Extracts rich reasoning trace from AgentExecutor's intermediate_steps
        including agent thoughts, tool calls, and observations.

        Args:
            original_input: Original input to runnable
            result: AgentExecutor result dict with 'output' and 'intermediate_steps'
        """
        try:
            task = self._get_task_str(original_input)
            output = result.get("output", "")
            steps = result.get("intermediate_steps", [])

            # Build rich reasoning from steps
            parts = [f"Question/Task: {task}", ""]
            parts.append(f"=== AGENT EXECUTION TRACE ({len(steps)} steps) ===")

            for i, step_tuple in enumerate(steps, 1):
                if len(step_tuple) != 2:
                    continue
                action, observation = step_tuple

                parts.append(f"\n--- Step {i} ---")
                # Extract thought from action.log (agent's reasoning)
                if hasattr(action, "log") and action.log:
                    parts.append(f"Thought: {action.log}")
                # Extract tool and input
                if hasattr(action, "tool"):
                    parts.append(f"Action: {action.tool}")
                    tool_input = str(action.tool_input)[:300]
                    parts.append(f"Action Input: {tool_input}")
                # Extract observation (tool output)
                observation_str = str(observation)[:300]
                parts.append(f"Observation: {observation_str}")

            parts.append("\n=== END TRACE ===")
            parts.append(f"\nFinal Answer: {output}")

            reasoning = "\n".join(parts)

            agent_output = AgentOutput(
                reasoning=reasoning,
                final_answer=output,
                skill_ids=[],
                raw={"input": original_input, "output": result},
            )

            feedback = f"Agent completed task in {len(steps)} steps"

            # Reflect and SkillManager
            reflection = self.reflector.reflect(
                question=task,
                agent_output=agent_output,
                skillbook=self.skillbook,
                ground_truth=None,
                feedback=feedback,
            )

            skill_manager_output = self.skill_manager.update_skills(
                reflection=reflection,
                skillbook=self.skillbook,
                question_context=f"task: {task}",
                progress=task,
            )

            self.skillbook.apply_update(skill_manager_output.update)

        except Exception as e:
            logger.error(f"ACE learning with trace failed: {e}")
            # Don't crash - continue without learning

    def _learn_with_langgraph_trace(self, original_input: Any, result: Dict[str, Any]):
        """
        Learn from LangGraph execution with message history.

        Extracts reasoning trace from LangGraph's message-based output format
        and passes it to the Reflector for analysis.

        Args:
            original_input: Original input to runnable
            result: LangGraph result dict with 'messages' key
        """
        try:
            task = self._get_task_str(original_input)
            output = self._extract_langgraph_output(result)
            trace, steps = self._extract_langgraph_trace(result)

            num_messages = len(result.get("messages", []))

            reasoning = f"""Question/Task: {task}

=== LANGGRAPH EXECUTION TRACE ({num_messages} messages) ===
{trace}
=== END TRACE ===

Final Answer: {output}"""

            agent_output = AgentOutput(
                reasoning=reasoning,
                final_answer=output,
                skill_ids=[],
                raw={"input": original_input, "output": result},
            )

            feedback = f"LangGraph agent completed in {num_messages} messages"

            # Reflect and SkillManager
            reflection = self.reflector.reflect(
                question=task,
                agent_output=agent_output,
                skillbook=self.skillbook,
                ground_truth=None,
                feedback=feedback,
            )

            skill_manager_output = self.skill_manager.update_skills(
                reflection=reflection,
                skillbook=self.skillbook,
                question_context=f"task: {task}",
                progress=task,
            )

            self.skillbook.apply_update(skill_manager_output.update)

        except Exception as e:
            logger.error(f"ACE LangGraph learning failed: {e}")
            # Don't crash - continue without learning

    async def _alearn(self, original_input: Any, result: Any):
        """
        Async version of _learn for background execution.

        Uses asyncio.to_thread() to run sync learning in a thread pool,
        preventing event loop blocking when async_learning=True.
        """
        await asyncio.to_thread(self._learn, original_input, result)

    async def _alearn_from_failure(self, original_input: Any, error_msg: str):
        """
        Async version of _learn_from_failure for background execution.

        Uses asyncio.to_thread() to run sync learning in a thread pool,
        preventing event loop blocking when async_learning=True.
        """
        await asyncio.to_thread(self._learn_from_failure, original_input, error_msg)

    async def _alearn_with_trace(self, original_input: Any, result: Dict[str, Any]):
        """
        Async version of _learn_with_trace for background execution.

        Uses asyncio.to_thread() to run sync learning in a thread pool,
        preventing event loop blocking when async_learning=True.
        """
        await asyncio.to_thread(self._learn_with_trace, original_input, result)

    async def _alearn_with_langgraph_trace(
        self, original_input: Any, result: Dict[str, Any]
    ):
        """
        Async version of _learn_with_langgraph_trace for background execution.

        Uses asyncio.to_thread() to run sync learning in a thread pool,
        preventing event loop blocking when async_learning=True.
        """
        await asyncio.to_thread(
            self._learn_with_langgraph_trace, original_input, result
        )

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Callback when a learning task completes (success or failure)."""
        self._tasks_completed_count += 1

    async def wait_for_learning(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all background learning tasks to complete.

        Only relevant when using async_learning=True.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            True if all learning completed, False if timeout reached

        Example:
            ace_chain = ACELangChain(chain, async_learning=True)
            result = await ace_chain.ainvoke(input)
            # Do other work while learning happens...
            success = await ace_chain.wait_for_learning(timeout=60.0)
            if success:
                print("Learning complete!")
        """
        if not self._learning_tasks:
            return True

        # Clean up completed tasks
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
        """
        Get async learning statistics.

        Returns:
            Dictionary with learning progress info:
            - tasks_submitted: Total tasks ever created
            - pending: Number of tasks still running
            - completed: Number of tasks finished
            - async_learning: Whether async mode is enabled

        Example:
            stats = ace_chain.learning_stats
            print(f"Pending: {stats['pending']}")
        """
        pending = len([t for t in self._learning_tasks if not t.done()])
        return {
            "tasks_submitted": self._tasks_submitted_count,
            "pending": pending,
            "completed": self._tasks_completed_count,
            "async_learning": self._async_learning,
        }

    def stop_async_learning(self):
        """
        Cancel all pending learning tasks.

        Call this to stop background learning early.

        Example:
            ace_chain = ACELangChain(chain, async_learning=True)
            await ace_chain.ainvoke(input)
            # Decide to stop early...
            ace_chain.stop_async_learning()
        """
        for task in self._learning_tasks:
            if not task.done():
                task.cancel()
        self._learning_tasks.clear()

    @staticmethod
    def _default_output_parser(result: Any) -> str:
        """
        Default output parser - converts result to string.

        Args:
            result: Runnable output

        Returns:
            String representation
        """
        # String - return as is
        if isinstance(result, str):
            return result

        # LangChain messages (AIMessage, etc.) have .content attribute
        if hasattr(result, "content"):
            return str(result.content)

        # Dict - try common output keys
        if isinstance(result, dict):
            for key in ["output", "answer", "result", "text"]:
                if key in result:
                    return str(result[key])
            return str(result)

        # Fallback to string representation
        return str(result)

    def _is_agent_executor(self) -> bool:
        """
        Check if the wrapped runnable is an AgentExecutor.

        AgentExecutor supports rich tracing via return_intermediate_steps=True.
        """
        if not AGENT_EXECUTOR_AVAILABLE or AgentExecutor is None:
            return False
        return isinstance(self.runnable, AgentExecutor)

    def _is_langgraph(self) -> bool:
        """
        Check if the wrapped runnable is a LangGraph CompiledStateGraph.

        LangGraph agents (from create_react_agent, create_tool_calling_agent,
        or custom StateGraph().compile()) use message-based I/O format:
        - Input: {"messages": [HumanMessage(...)]}
        - Output: {"messages": [HumanMessage, AIMessage, ToolMessage, ...]}
        """
        if not LANGGRAPH_AVAILABLE or CompiledStateGraph is None:
            return False
        return isinstance(self.runnable, CompiledStateGraph)

    def _extract_langgraph_output(self, result: Dict[str, Any]) -> str:
        """
        Extract final answer from LangGraph message-based output.

        Finds the last AIMessage with content (skipping tool messages).

        Args:
            result: LangGraph result dict with "messages" key

        Returns:
            Final answer string, or empty string if not found
        """
        messages = result.get("messages", [])
        # Find last AIMessage with content (not a tool message)
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                # Skip tool messages
                msg_type = getattr(msg, "type", msg.__class__.__name__.lower())
                if msg_type != "tool":
                    return str(msg.content)
        return ""

    def _extract_langgraph_trace(
        self, result: Dict[str, Any]
    ) -> tuple[str, List[tuple[Any, Any]]]:
        """
        Extract reasoning trace and intermediate steps from LangGraph output.

        Parses the message history to build a reasoning trace for the Reflector
        and extracts tool calls/observations as intermediate_steps.

        Args:
            result: LangGraph result dict with "messages" key

        Returns:
            Tuple of (trace_string, intermediate_steps_list)
        """
        messages = result.get("messages", [])
        parts = []
        intermediate_steps: List[tuple[Any, Any]] = []

        for msg in messages:
            msg_type = getattr(msg, "type", msg.__class__.__name__.lower())
            content = getattr(msg, "content", str(msg))

            if msg_type == "human":
                parts.append(f"Human: {str(content)[:300]}")
            elif msg_type == "ai":
                if content:
                    parts.append(f"Assistant: {str(content)[:300]}")
                # Extract tool calls if present
                tool_calls = getattr(msg, "tool_calls", [])
                for tc in tool_calls:
                    tool_name = (
                        tc.get("name", "unknown")
                        if isinstance(tc, dict)
                        else getattr(tc, "name", "unknown")
                    )
                    parts.append(f"  Tool Call: {tool_name}")
                    # Add to intermediate_steps with None observation (filled later)
                    intermediate_steps.append((tc, None))
            elif msg_type == "tool":
                parts.append(f"Tool Result: {str(content)[:300]}")
                # Match with last tool call that has no observation
                for i in range(len(intermediate_steps) - 1, -1, -1):
                    if intermediate_steps[i][1] is None:
                        tc = intermediate_steps[i][0]
                        intermediate_steps[i] = (tc, content)
                        break

        return "\n".join(parts), intermediate_steps

    def _get_task_str(self, original_input: Any) -> str:
        """Extract task string from input."""
        if isinstance(original_input, str):
            return original_input
        elif isinstance(original_input, dict):
            # Handle LangGraph message format
            if "messages" in original_input:
                messages = original_input["messages"]
                if messages and hasattr(messages[0], "content"):
                    return str(messages[0].content)
            # Handle standard dict formats
            return (
                original_input.get("input")
                or original_input.get("question")
                or original_input.get("query")
                or str(original_input)
            )
        return str(original_input)

    def save_skillbook(self, path: str):
        """
        Save learned skillbook to file.

        Args:
            path: File path to save to

        Example:
            ace_chain.save_skillbook("chain_expert.json")
        """
        self.skillbook.save_to_file(path)

    def load_skillbook(self, path: str):
        """
        Load skillbook from file (replaces current).

        Args:
            path: File path to load from

        Example:
            ace_chain.load_skillbook("expert.json")
        """
        self.skillbook = Skillbook.load_from_file(path)

    def enable_learning(self):
        """Enable learning (allows learn() to update skillbook)."""
        self.is_learning = True

    def disable_learning(self):
        """Disable learning (prevents learn() from updating skillbook)."""
        self.is_learning = False

    def get_strategies(self) -> str:
        """
        Get current skillbook strategies as formatted text.

        Returns:
            Formatted string with learned strategies (empty if none)

        Example:
            strategies = ace_chain.get_strategies()
            print(strategies)
        """
        if not self.skillbook or not self.skillbook.skills():
            return ""
        return wrap_skillbook_context(self.skillbook)

    def __repr__(self) -> str:
        """String representation."""
        skills_count = len(self.skillbook.skills()) if self.skillbook else 0
        return (
            f"ACELangChain("
            f"runnable={self.runnable.__class__.__name__}, "
            f"strategies={skills_count}, "
            f"learning={'enabled' if self.is_learning else 'disabled'})"
        )


__all__ = ["ACELangChain", "LANGCHAIN_AVAILABLE", "LANGGRAPH_AVAILABLE"]
