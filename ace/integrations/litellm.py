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
    agent.save_playbook("my_agent.json")

    # Load in next session
    agent = ACELiteLLM(model="gpt-4o-mini", playbook_path="my_agent.json")
"""

from typing import TYPE_CHECKING, List, Optional, Dict, Any

from ..playbook import Playbook
from ..roles import Generator, Reflector, Curator
from ..adaptation import OfflineAdapter, Sample, TaskEnvironment
from ..prompts_v2_1 import PromptManager

if TYPE_CHECKING:
    from ..deduplication import DeduplicationConfig


class ACELiteLLM:
    """
    LiteLLM integration with ACE learning.

    Bundles Generator, Reflector, Curator, and Playbook into a simple interface
    powered by LiteLLM (supports 100+ LLM providers).

    Perfect for:
    - Quick start with ACE
    - Q&A, classification, and reasoning tasks
    - Prototyping and experimentation
    - Learning without external frameworks

    For other use cases:
    - ACEAgent (browser-use): Browser automation with learning
    - ACELangChain: LangChain chains/agents with learning
    - Integration pattern: Custom agent systems (see docs)

    Attributes:
        playbook: Learned strategies (Playbook instance)
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
        agent.save_playbook("learned.json")
        agent2 = ACELiteLLM(model="gpt-4o-mini", playbook_path="learned.json")
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 512,
        temperature: float = 0.0,
        playbook_path: Optional[str] = None,
        is_learning: bool = True,
        dedup_config: Optional["DeduplicationConfig"] = None,
    ):
        """
        Initialize ACELiteLLM agent.

        Args:
            model: LiteLLM model name (default: gpt-4o-mini)
                   Supports 100+ providers: OpenAI, Anthropic, Google, etc.
            max_tokens: Max tokens for responses (default: 512)
            temperature: Sampling temperature (default: 0.0)
            playbook_path: Path to existing playbook (optional)
            is_learning: Enable/disable learning (default: True)
            dedup_config: Optional DeduplicationConfig for bullet deduplication

        Raises:
            ImportError: If LiteLLM is not installed

        Example:
            # OpenAI
            agent = ACELiteLLM(model="gpt-4o-mini")

            # Anthropic
            agent = ACELiteLLM(model="claude-3-haiku-20240307")

            # Google
            agent = ACELiteLLM(model="gemini/gemini-pro")

            # With existing playbook
            agent = ACELiteLLM(
                model="gpt-4o-mini",
                playbook_path="expert.json"
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

        # Load or create playbook
        if playbook_path:
            self.playbook = Playbook.load_from_file(playbook_path)
        else:
            self.playbook = Playbook()

        # Create LLM client
        self.llm = LiteLLMClient(
            model=model, max_tokens=max_tokens, temperature=temperature
        )

        # Create ACE components with v2.1 prompts
        prompt_mgr = PromptManager()
        self.generator = Generator(
            self.llm, prompt_template=prompt_mgr.get_generator_prompt()
        )
        self.reflector = Reflector(
            self.llm, prompt_template=prompt_mgr.get_reflector_prompt()
        )
        self.curator = Curator(
            self.llm, prompt_template=prompt_mgr.get_curator_prompt()
        )

        # Store adapter reference for async learning control
        self._adapter: Optional[OfflineAdapter] = None

    def ask(self, question: str, context: str = "") -> str:
        """
        Ask a question and get an answer (uses current playbook).

        This uses the ACE Generator with the current playbook's learned strategies.

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
        """
        result = self.generator.generate(
            question=question, context=context, playbook=self.playbook
        )
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

        Uses OfflineAdapter to learn from a batch of samples.

        Args:
            samples: List of Sample objects to learn from
            environment: TaskEnvironment for evaluating results
            epochs: Number of training epochs (default: 1)
            async_learning: Run learning in background (default: False)
                           When True, Generator returns immediately while
                           Reflector/Curator process in background.
            max_reflector_workers: Number of parallel Reflector threads
                                  (default: 3, only used when async_learning=True)
            checkpoint_interval: Save playbook every N samples (optional)
            checkpoint_dir: Directory for checkpoints (optional)

        Returns:
            List of AdapterStepResult from training

        Example:
            from ace import Sample, SimpleEnvironment

            samples = [
                Sample(question="What is 2+2?", ground_truth="4"),
                Sample(question="Capital of France?", ground_truth="Paris"),
            ]

            agent = ACELiteLLM()
            results = agent.learn(samples, SimpleEnvironment(), epochs=1)

            print(f"Learned {len(agent.playbook.bullets())} strategies")

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

        # Create offline adapter
        self._adapter = OfflineAdapter(
            playbook=self.playbook,
            generator=self.generator,
            reflector=self.reflector,
            curator=self.curator,
            async_learning=async_learning,
            max_reflector_workers=max_reflector_workers,
            dedup_config=self.dedup_config,
        )

        # Run learning
        results = self._adapter.run(
            samples=samples,
            environment=environment,
            epochs=epochs,
            checkpoint_interval=checkpoint_interval,
            checkpoint_dir=checkpoint_dir,
            wait_for_learning=not async_learning,  # Don't block if async
        )

        return results

    def save_playbook(self, path: str):
        """
        Save learned playbook to file.

        Args:
            path: File path to save to (creates parent dirs if needed)

        Example:
            agent.save_playbook("my_agent.json")
        """
        self.playbook.save_to_file(path)

    def load_playbook(self, path: str):
        """
        Load playbook from file (replaces current playbook).

        Args:
            path: File path to load from

        Example:
            agent.load_playbook("expert.json")
        """
        self.playbook = Playbook.load_from_file(path)

    def enable_learning(self):
        """Enable learning (allows learn() to update playbook)."""
        self.is_learning = True

    def disable_learning(self):
        """Disable learning (prevents learn() from updating playbook)."""
        self.is_learning = False

    def get_strategies(self) -> str:
        """
        Get current learned strategies as formatted text.

        Returns:
            Formatted string with learned strategies (empty if none)

        Example:
            strategies = agent.get_strategies()
            print(strategies)
        """
        if not self.playbook or not self.playbook.bullets():
            return ""

        lines = ["Learned Strategies:"]
        for i, bullet in enumerate(self.playbook.bullets(), 1):
            score = f"+{bullet.helpful}/-{bullet.harmful}"
            lines.append(f"{i}. [{bullet.id}] {bullet.content}")
            lines.append(f"   Score: {score}")

        return "\n".join(lines)

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
        if self._adapter is None:
            return True
        return self._adapter.wait_for_learning(timeout)

    @property
    def learning_stats(self) -> Dict[str, Any]:
        """
        Get async learning statistics.

        Returns:
            Dictionary with learning progress info:
            - async_learning: Whether async mode is enabled
            - pending: Number of samples still being processed
            - completed: Number of samples processed
            - queue_size: Reflections waiting for Curator

        Example:
            stats = agent.learning_stats
            print(f"Pending: {stats['pending']}")
        """
        if self._adapter is None:
            return {"async_learning": False, "pending": 0, "completed": 0}
        return self._adapter.learning_stats

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
        if self._adapter:
            self._adapter.stop_async_learning()

    def __repr__(self) -> str:
        """String representation."""
        bullets_count = len(self.playbook.bullets()) if self.playbook else 0
        return (
            f"ACELiteLLM(model='{self.model}', "
            f"strategies={bullets_count}, "
            f"learning={'enabled' if self.is_learning else 'disabled'})"
        )


__all__ = ["ACELiteLLM"]
