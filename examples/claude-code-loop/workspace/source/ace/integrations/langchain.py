"""
ACE + LangChain integration for learning from chain/agent execution.

This module provides ACELangChain, a wrapper that adds ACE learning capabilities
to any LangChain Runnable (chains, agents, custom runnables).

When to Use ACELangChain:
- Complex workflows: Multi-step LangChain chains
- Tool-using agents: LangChain agents with tools
- Custom runnables: Your own LangChain components
- Production workflows: LangChain orchestration with learning

When NOT to Use ACELangChain:
- Simple Q&A → Use ACELiteLLM
- Browser automation → Use ACEAgent (browser-use)
- Custom agent (non-LangChain) → Use integration pattern (see docs)

Example:
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
    ace_chain.save_playbook("chain_expert.json")
"""

from typing import Any, Optional, Dict, Callable
import logging

from ..playbook import Playbook
from ..roles import Reflector, Curator, GeneratorOutput
from ..prompts_v2_1 import PromptManager
from .base import wrap_playbook_context

try:
    from langchain_core.runnables import Runnable

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Runnable = None  # type: ignore

logger = logging.getLogger(__name__)


class ACELangChain:
    """
    LangChain Runnable wrapper with ACE learning.

    Wraps any LangChain Runnable (chain, agent, custom) and adds ACE learning
    capabilities. The runnable executes normally, but ACE learns from results
    to improve future executions.

    Pattern:
        1. INJECT: Add playbook context to input
        2. EXECUTE: Run the LangChain runnable
        3. LEARN: Update playbook via Reflector + Curator

    Attributes:
        runnable: The LangChain Runnable being wrapped
        playbook: Learned strategies (Playbook instance)
        is_learning: Whether learning is enabled

    Example:
        # Basic usage
        from langchain_openai import ChatOpenAI
        from langchain.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template("Answer: {input}")
        chain = prompt | ChatOpenAI(temperature=0)

        ace_chain = ACELangChain(runnable=chain)
        result = ace_chain.invoke({"input": "What is 2+2?"})

        # With existing playbook
        ace_chain = ACELangChain(
            runnable=chain,
            playbook_path="expert.json"
        )

        # Async execution
        result = await ace_chain.ainvoke({"input": "Question"})
    """

    def __init__(
        self,
        runnable: Any,
        ace_model: str = "gpt-4o-mini",
        playbook_path: Optional[str] = None,
        is_learning: bool = True,
        output_parser: Optional[Callable[[Any], str]] = None,
    ):
        """
        Initialize ACELangChain wrapper.

        Args:
            runnable: LangChain Runnable (chain, agent, custom)
            ace_model: Model for ACE learning (Reflector/Curator)
            playbook_path: Path to existing playbook (optional)
            is_learning: Enable/disable learning (default: True)
            output_parser: Custom function to parse runnable output to string
                          (default: converts to string)

        Raises:
            ImportError: If LangChain is not installed

        Example:
            # Basic
            ace_chain = ACELangChain(my_chain)

            # With custom output parser
            def parse_output(result):
                return result["output"]["final_answer"]

            ace_chain = ACELangChain(
                my_chain,
                output_parser=parse_output
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
        self.output_parser = output_parser or self._default_output_parser

        # Load or create playbook
        if playbook_path:
            self.playbook = Playbook.load_from_file(playbook_path)
        else:
            self.playbook = Playbook()

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
        self.curator = Curator(
            self.llm, prompt_template=prompt_mgr.get_curator_prompt()
        )

    def invoke(self, input: Any, **kwargs) -> Any:
        """
        Execute runnable with ACE learning (sync).

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
        """
        # Step 1: Inject playbook context
        enhanced_input = self._inject_context(input)

        # Step 2: Execute runnable
        try:
            result = self.runnable.invoke(enhanced_input, **kwargs)
        except Exception as e:
            logger.error(f"Error executing runnable: {e}")
            # Learn from failure
            if self.is_learning:
                self._learn_from_failure(input, str(e))
            raise

        # Step 3: Learn from result
        if self.is_learning:
            self._learn(input, result)

        return result

    async def ainvoke(self, input: Any, **kwargs) -> Any:
        """
        Execute runnable with ACE learning (async).

        Args:
            input: Input for the runnable (string, dict, etc.)
            **kwargs: Additional arguments passed to runnable.ainvoke()

        Returns:
            Output from the runnable

        Example:
            result = await ace_chain.ainvoke({"input": "Question"})
        """
        # Step 1: Inject playbook context
        enhanced_input = self._inject_context(input)

        # Step 2: Execute runnable (async)
        try:
            result = await self.runnable.ainvoke(enhanced_input, **kwargs)
        except Exception as e:
            logger.error(f"Error executing runnable: {e}")
            # Learn from failure
            if self.is_learning:
                self._learn_from_failure(input, str(e))
            raise

        # Step 3: Learn from result
        if self.is_learning:
            self._learn(input, result)

        return result

    def _inject_context(self, input: Any) -> Any:
        """
        Add playbook context to input.

        Handles common input formats:
        - String: Append playbook context
        - Dict with "input" key: Enhance input field
        - Dict without: Add playbook_context key
        - Other: Return unchanged (no playbook strategies yet)

        Args:
            input: Original input

        Returns:
            Enhanced input with playbook context
        """
        # No context if no strategies yet
        if not self.playbook or not self.playbook.bullets():
            return input

        playbook_context = wrap_playbook_context(self.playbook)

        # String input - append context
        if isinstance(input, str):
            return f"{input}\n\n{playbook_context}"

        # Dict input with "input" key - enhance that field
        if isinstance(input, dict) and "input" in input:
            enhanced = input.copy()
            enhanced["input"] = f"{input['input']}\n\n{playbook_context}"
            return enhanced

        # Dict input without "input" key - add playbook_context key
        if isinstance(input, dict):
            enhanced = input.copy()
            enhanced["playbook_context"] = playbook_context
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
            generator_output = GeneratorOutput(
                reasoning=f"Task: {task}\nOutput: {output_str}",
                final_answer=output_str,
                bullet_ids=[],  # LangChain runnables don't cite bullets
                raw={"input": original_input, "output": result},
            )

            # Build feedback
            feedback = f"Task completed successfully. Output: {output_str[:500]}"

            # Reflect: Analyze execution
            reflection = self.reflector.reflect(
                question=task,
                generator_output=generator_output,
                playbook=self.playbook,
                ground_truth=None,
                feedback=feedback,
            )

            # Curate: Generate playbook updates
            curator_output = self.curator.curate(
                reflection=reflection,
                playbook=self.playbook,
                question_context=f"task: {task}",
                progress=task,
            )

            # Apply updates
            self.playbook.apply_delta(curator_output.delta)

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
            generator_output = GeneratorOutput(
                reasoning=f"Task: {task}\nError: {error_msg}",
                final_answer=f"Failed: {error_msg}",
                bullet_ids=[],
                raw={"input": original_input, "error": error_msg},
            )

            # Build failure feedback
            feedback = f"Task failed with error: {error_msg}"

            # Reflect on failure
            reflection = self.reflector.reflect(
                question=task,
                generator_output=generator_output,
                playbook=self.playbook,
                ground_truth=None,
                feedback=feedback,
            )

            # Curate: Learn from failure patterns
            curator_output = self.curator.curate(
                reflection=reflection,
                playbook=self.playbook,
                question_context=f"task: {task}",
                progress=f"Failed: {task}",
            )

            # Apply updates
            self.playbook.apply_delta(curator_output.delta)

        except Exception as e:
            logger.error(f"ACE failure learning failed: {e}")
            # Don't crash

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

    def save_playbook(self, path: str):
        """
        Save learned playbook to file.

        Args:
            path: File path to save to

        Example:
            ace_chain.save_playbook("chain_expert.json")
        """
        self.playbook.save_to_file(path)

    def load_playbook(self, path: str):
        """
        Load playbook from file (replaces current).

        Args:
            path: File path to load from

        Example:
            ace_chain.load_playbook("expert.json")
        """
        self.playbook = Playbook.load_from_file(path)

    def enable_learning(self):
        """Enable learning (allows learn() to update playbook)."""
        self.is_learning = True

    def disable_learning(self):
        """Disable learning (prevents learn() from updating playbook)."""
        self.is_learning = False

    def __repr__(self) -> str:
        """String representation."""
        bullets_count = len(self.playbook.bullets()) if self.playbook else 0
        return (
            f"ACELangChain("
            f"runnable={self.runnable.__class__.__name__}, "
            f"strategies={bullets_count}, "
            f"learning={'enabled' if self.is_learning else 'disabled'})"
        )


__all__ = ["ACELangChain", "LANGCHAIN_AVAILABLE"]
