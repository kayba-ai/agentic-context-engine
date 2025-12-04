"""
ACE + LiteLLM integration for quick-start learning agents.

This module provides ACELiteLLM, a high-level wrapper bundling ACE learning
with LiteLLM for easy prototyping and simple tasks.

When to Use ACELiteLLM:
- Quick start: Want to try ACE with minimal setup
- Simple tasks: Q&A, classification, reasoning
- Prototyping: Experimenting with ACE learning
- No framework needed: Direct LLM usage with learning

When NOT to Use ACELiteLLM:
- Browser automation → Use ACEAgent (browser-use)
- LangChain chains/agents → Use ACELangChain
- Custom agentic system → Use integration pattern (see docs/INTEGRATION_GUIDE.md)

Example:
    from ace.integrations import ACELiteLLM

    # Create LiteLLM-enhanced agent
    agent = ACELiteLLM(model="gpt-4o-mini")

    # Ask questions (uses current knowledge)
    answer = agent.ask("What is 2+2?")

    # Learn from examples
    from ace import Sample, SimpleEnvironment
    samples = [
        Sample(question="What is 2+2?", ground_truth="4"),
        Sample(question="Capital of France?", ground_truth="Paris"),
    ]
    agent.learn(samples, SimpleEnvironment())

    # Save learned knowledge
    agent.save_skillbook("my_agent.json")

    # Load in next session
    agent = ACELiteLLM(model="gpt-4o-mini", skillbook_path="my_agent.json")
"""

from typing import TYPE_CHECKING, List, Optional, Dict, Any, Tuple, Union

from ..skillbook import Skillbook
from ..roles import Agent, Reflector, SkillManager, AgentOutput
from ..adaptation import OfflineACE, Sample, TaskEnvironment
from ..prompts_v2_1 import PromptManager

if TYPE_CHECKING:
    from ..deduplication import DeduplicationConfig


class ACELiteLLM:
    """
    LiteLLM integration with ACE learning.

    Bundles Agent, Reflector, SkillManager, and Skillbook into a simple interface
    powered by LiteLLM (supports 100+ LLM providers).

    Perfect for:
    - Quick start with ACE
    - Q&A, classification, and reasoning tasks
    - Prototyping and experimentation
    - Learning without external frameworks

    Insight Level: Micro
        Uses the full ACE loop with TaskEnvironment for ground truth evaluation.
        The learn() method runs OfflineACE which evaluates correctness and
        learns from whether answers are right or wrong.
        See docs/COMPLETE_GUIDE_TO_ACE.md for details.

    For other use cases:
    - ACEAgent (browser-use): Browser automation with learning (meso-level)
    - ACELangChain: LangChain chains/agents with learning (meso for AgentExecutor)
    - Integration pattern: Custom agent systems (see docs)

    Attributes:
        skillbook: Learned strategies (Skillbook instance)
        is_learning: Whether learning is enabled
        model: LiteLLM model name

    Example:
        # Basic usage
        agent = ACELiteLLM(model="gpt-4o-mini")
        answer = agent.ask("What is the capital of France?")
        print(answer)  # "Paris"

        # Learning from feedback
        from ace import Sample, SimpleEnvironment
        samples = [
            Sample(question="What is 2+2?", ground_truth="4"),
            Sample(question="What is 3+3?", ground_truth="6"),
        ]
        agent.learn(samples, SimpleEnvironment(), epochs=1)

        # Save and load
        agent.save_skillbook("learned.json")
        agent2 = ACELiteLLM(model="gpt-4o-mini", skillbook_path="learned.json")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        # Authentication & endpoint
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        # HTTP/SSL settings
        extra_headers: Optional[Dict[str, str]] = None,
        ssl_verify: Optional[Union[bool, str]] = None,
        # ACE-specific settings
        skillbook_path: Optional[str] = None,
        is_learning: bool = True,
        dedup_config: Optional["DeduplicationConfig"] = None,
        # Pass-through for advanced LiteLLM options
        **llm_kwargs: Any,
    ):
        """
        Initialize ACELiteLLM agent.

        Args:
            model: LiteLLM model name (default: gpt-4o-mini)
                   Supports 100+ providers: OpenAI, Anthropic, Google, etc.
            max_tokens: Max tokens for responses (default: 2048)
            temperature: Sampling temperature (default: 0.0)
            api_key: API key for the LLM provider. Falls back to env vars if not set.
            base_url: Custom API endpoint URL (e.g., http://localhost:1234/v1)
            extra_headers: Custom HTTP headers dict (e.g., {"X-Tenant-ID": "abc"})
            ssl_verify: SSL verification. False to disable, or path to CA bundle.
            skillbook_path: Path to existing skillbook (optional)
            is_learning: Enable/disable learning (default: True)
            dedup_config: Optional DeduplicationConfig for skill deduplication
            **llm_kwargs: Additional LiteLLM parameters (timeout, max_retries, etc.)

        Raises:
            ImportError: If LiteLLM is not installed

        Example:
            # OpenAI
            agent = ACELiteLLM(model="gpt-4o-mini")

            # Anthropic
            agent = ACELiteLLM(model="claude-3-haiku-20240307")

            # Google
            agent = ACELiteLLM(model="gemini/gemini-pro")

            # With explicit API key
            agent = ACELiteLLM(model="gpt-4", api_key="sk-...")

            # Custom endpoint (LM Studio, Ollama)
            agent = ACELiteLLM(
                model="openai/local-model",
                base_url="http://localhost:1234/v1"
            )

            # Enterprise with custom headers and SSL
            agent = ACELiteLLM(
                model="gpt-4",
                base_url="https://proxy.company.com/v1",
                extra_headers={"X-Tenant-ID": "team-alpha"},
                ssl_verify="/path/to/internal-ca.pem"
            )

            # With existing skillbook
            agent = ACELiteLLM(
                model="gpt-4o-mini",
                skillbook_path="expert.json"
            )

            # With deduplication
            from ace import DeduplicationConfig
            agent = ACELiteLLM(
                model="gpt-4o-mini",
                dedup_config=DeduplicationConfig(similarity_threshold=0.85)
            )
        """
        # Import LiteLLM (required for this integration)
        try:
            from ..llm_providers import LiteLLMClient
        except ImportError:
            raise ImportError(
                "ACELiteLLM requires LiteLLM. Install with:\n"
                "pip install ace-framework  # (LiteLLM included by default)\n"
                "or: pip install litellm"
            )

        self.model = model
        self.is_learning = is_learning
        self.dedup_config = dedup_config

        # Load or create skillbook
        if skillbook_path:
            self.skillbook = Skillbook.load_from_file(skillbook_path)
        else:
            self.skillbook = Skillbook()

        # Create LLM client with configuration
        self.llm = LiteLLMClient(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            api_key=api_key,
            api_base=base_url,  # Map user-friendly name to LiteLLM's api_base
            extra_headers=extra_headers,
            ssl_verify=ssl_verify,
            **llm_kwargs,
        )

        # Create ACE components with v2.1 prompts
        prompt_mgr = PromptManager()
        self.agent = Agent(self.llm, prompt_template=prompt_mgr.get_agent_prompt())
        self.reflector = Reflector(
            self.llm, prompt_template=prompt_mgr.get_reflector_prompt()
        )
        self.skill_manager = SkillManager(
            self.llm, prompt_template=prompt_mgr.get_skill_manager_prompt()
        )

        # Store ACE reference for async learning control
        self._ace: Optional[OfflineACE] = None

        # Store last interaction for learn_from_feedback()
        self._last_interaction: Optional[Tuple[str, AgentOutput]] = None

    def ask(self, question: str, context: str = "") -> str:
        """
        Ask a question and get an answer (uses current skillbook).

        This uses the ACE Agent with the current skillbook's learned strategies.
        The full AgentOutput trace is stored internally for potential learning
        via learn_from_feedback().

        Args:
            question: Question to answer
            context: Additional context (optional)

        Returns:
            Answer string

        Example:
            agent = ACELiteLLM()
            answer = agent.ask("What is the capital of Japan?")
            print(answer)  # "Tokyo"

            # With context
            answer = agent.ask(
                "What is GDP?",
                context="Economics question"
            )

            # Learn from feedback
            agent.learn_from_feedback(feedback="correct")
        """
        result = self.agent.generate(
            question=question, context=context, skillbook=self.skillbook
        )
        # Store full trace for potential learning via learn_from_feedback()
        self._last_interaction = (question, result)
        return result.final_answer

    def learn(
        self,
        samples: List[Sample],
        environment: TaskEnvironment,
        epochs: int = 1,
        async_learning: bool = False,
        max_reflector_workers: int = 3,
        checkpoint_interval: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Learn from examples (offline learning).

        Uses OfflineACE to learn from a batch of samples.

        Insight Level: Micro
            This is micro-level learning with ground truth evaluation.
            The TaskEnvironment evaluates each answer for correctness,
            and the Reflector learns from whether answers are right or wrong.

        Args:
            samples: List of Sample objects to learn from
            environment: TaskEnvironment for evaluating results
            epochs: Number of training epochs (default: 1)
            async_learning: Run learning in background (default: False)
                           When True, Agent returns immediately while
                           Reflector/SkillManager process in background.
            max_reflector_workers: Number of parallel Reflector threads
                                  (default: 3, only used when async_learning=True)
            checkpoint_interval: Save skillbook every N samples (optional)
            checkpoint_dir: Directory for checkpoints (optional)

        Returns:
            List of ACEStepResult from training

        Example:
            from ace import Sample, SimpleEnvironment

            samples = [
                Sample(question="What is 2+2?", ground_truth="4"),
                Sample(question="Capital of France?", ground_truth="Paris"),
            ]

            agent = ACELiteLLM()
            results = agent.learn(samples, SimpleEnvironment(), epochs=1)

            print(f"Learned {len(agent.skillbook.skills())} strategies")

            # Async learning example
            results = agent.learn(
                samples, SimpleEnvironment(),
                async_learning=True,
                max_reflector_workers=3
            )
            # Results return immediately, learning continues in background
            agent.wait_for_learning()  # Block until complete
            print(agent.learning_stats)
        """
        if not self.is_learning:
            raise ValueError("Learning is disabled. Set is_learning=True first.")

        # Create offline ACE
        self._ace = OfflineACE(
            skillbook=self.skillbook,
            agent=self.agent,
            reflector=self.reflector,
            skill_manager=self.skill_manager,
            async_learning=async_learning,
            max_reflector_workers=max_reflector_workers,
            dedup_config=self.dedup_config,
        )

        # Run learning
        results = self._ace.run(
            samples=samples,
            environment=environment,
            epochs=epochs,
            checkpoint_interval=checkpoint_interval,
            checkpoint_dir=checkpoint_dir,
            wait_for_learning=not async_learning,  # Don't block if async
        )

        return results

    def learn_from_feedback(
        self,
        feedback: str,
        ground_truth: Optional[str] = None,
    ) -> bool:
        """
        Learn from the last ask() interaction.

        Uses the stored AgentOutput trace from the previous ask() call.
        This allows the Reflector to analyze the full reasoning and skill
        citations, not just the final answer.

        Follows the `learn_from_X` naming pattern from other ACE integrations
        (e.g., ACEAgent._learn_from_execution, ACELangChain._learn_from_failure).

        Args:
            feedback: User feedback describing the outcome. Can be:
                     - Simple: "correct", "wrong", "partially correct"
                     - Detailed: "Good answer but too verbose"
            ground_truth: Optional correct answer if the response was wrong

        Returns:
            True if learning was applied
            False if no prior interaction exists or learning is disabled

        Example:
            agent = ACELiteLLM()

            # Ask and provide feedback
            answer = agent.ask("What is 2+2?")
            agent.learn_from_feedback(feedback="correct")

            # With ground truth for incorrect answers
            answer = agent.ask("Capital of Australia?")
            agent.learn_from_feedback(
                feedback="wrong",
                ground_truth="Canberra"
            )

            # Detailed feedback
            answer = agent.ask("Explain quantum physics")
            agent.learn_from_feedback(
                feedback="Too technical for a beginner audience"
            )
        """
        if not self.is_learning:
            return False

        if self._last_interaction is None:
            return False

        question, agent_output = self._last_interaction

        # Run Reflector with full trace context
        reflection = self.reflector.reflect(
            question=question,
            agent_output=agent_output,  # Full trace: reasoning, skill_ids
            skillbook=self.skillbook,
            ground_truth=ground_truth,
            feedback=feedback,
        )

        # Run SkillManager to generate skillbook updates
        skill_manager_output = self.skill_manager.update_skills(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context=f"User interaction: {question}",
            progress="Learning from user feedback",
        )

        # Apply updates to skillbook
        self.skillbook.apply_update(skill_manager_output.update)
        return True

    def save_skillbook(self, path: str):
        """
        Save learned skillbook to file.

        Args:
            path: File path to save to (creates parent dirs if needed)

        Example:
            agent.save_skillbook("my_agent.json")
        """
        self.skillbook.save_to_file(path)

    def load_skillbook(self, path: str):
        """
        Load skillbook from file (replaces current skillbook).

        Args:
            path: File path to load from

        Example:
            agent.load_skillbook("expert.json")
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
            strategies = agent.get_strategies()
            print(strategies)
        """
        if not self.skillbook or not self.skillbook.skills():
            return ""
        from .base import wrap_skillbook_context

        return wrap_skillbook_context(self.skillbook)

    def wait_for_learning(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for async learning to complete.

        Only relevant when using async_learning=True in learn().

        Args:
            timeout: Maximum seconds to wait (None = wait forever)

        Returns:
            True if all learning completed, False if timeout reached

        Example:
            agent.learn(samples, env, async_learning=True)
            # Do other work while learning happens...
            success = agent.wait_for_learning(timeout=60.0)
            if success:
                print("Learning complete!")
        """
        if self._ace is None:
            return True
        return self._ace.wait_for_learning(timeout)

    @property
    def learning_stats(self) -> Dict[str, Any]:
        """
        Get async learning statistics.

        Returns:
            Dictionary with learning progress info:
            - async_learning: Whether async mode is enabled
            - pending: Number of samples still being processed
            - completed: Number of samples processed
            - queue_size: Reflections waiting for SkillManager

        Example:
            stats = agent.learning_stats
            print(f"Pending: {stats['pending']}")
        """
        if self._ace is None:
            return {"async_learning": False, "pending": 0, "completed": 0}
        return self._ace.learning_stats

    def stop_async_learning(self):
        """
        Stop async learning pipeline.

        Shuts down background threads and clears pending work.
        Call this before exiting to ensure clean shutdown.

        Example:
            agent.learn(samples, env, async_learning=True)
            # Decide to stop early...
            agent.stop_async_learning()
        """
        if self._ace:
            self._ace.stop_async_learning()

    def __repr__(self) -> str:
        """String representation."""
        skills_count = len(self.skillbook.skills()) if self.skillbook else 0
        return (
            f"ACELiteLLM(model='{self.model}', "
            f"strategies={skills_count}, "
            f"learning={'enabled' if self.is_learning else 'disabled'})"
        )


__all__ = ["ACELiteLLM"]
