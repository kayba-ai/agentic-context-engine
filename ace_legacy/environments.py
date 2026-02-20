"""Data contracts for the ACE pipeline — samples, environments, and step results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .roles import AgentOutput, ReflectorOutput, SkillManagerOutput


@dataclass
class Sample:
    """Single task instance presented to ACE."""

    question: str
    context: str = ""
    ground_truth: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)
    id: Optional[str] = None


@dataclass
class EnvironmentResult:
    """Feedback returned by the task environment after executing the generator output."""

    feedback: str
    ground_truth: Optional[str]
    metrics: Dict[str, float] = field(default_factory=dict)


class TaskEnvironment(ABC):
    """
    Abstract interface for evaluating agent outputs.

    Implement this class to define how your specific task evaluates
    the Agent's answers. The environment provides feedback that
    helps ACE learn what works and what doesn't.

    Example Implementation:
        >>> class MathEnvironment(TaskEnvironment):
        ...     def evaluate(self, sample, agent_output):
        ...         # Parse the answer
        ...         predicted = extract_number(agent_output.final_answer)
        ...         correct = str(predicted) == sample.ground_truth
        ...
        ...         # Provide feedback
        ...         if correct:
        ...             feedback = "Correct!"
        ...         else:
        ...             feedback = f"Incorrect. Expected {sample.ground_truth}"
        ...
        ...         return EnvironmentResult(
        ...             feedback=feedback,
        ...             ground_truth=sample.ground_truth,
        ...             metrics={'accuracy': 1.0 if correct else 0.0}
        ...         )
    """

    @abstractmethod
    def evaluate(self, sample: Sample, agent_output: AgentOutput) -> EnvironmentResult:
        """
        Evaluate the agent's output for a given sample.

        Args:
            sample: The input sample with question and context
            agent_output: The Agent's produced answer

        Returns:
            EnvironmentResult with feedback and optional ground truth

        The feedback should be informative enough for the Reflector
        to understand what went right or wrong.
        """


class SimpleEnvironment(TaskEnvironment):
    """
    Simple built-in environment for quick testing and demos.

    Checks if the ground truth appears in the answer (case-insensitive).
    Perfect for getting started without creating a custom environment.

    Example:
        >>> from ace import SimpleEnvironment, Sample
        >>> env = SimpleEnvironment()
        >>> sample = Sample(question="What is 2+2?", ground_truth="4")
        >>> result = env.evaluate(sample, agent_output)
    """

    def evaluate(self, sample: Sample, agent_output: AgentOutput) -> EnvironmentResult:
        """Check if ground truth appears in the answer."""
        if not sample.ground_truth:
            return EnvironmentResult(
                feedback="No ground truth provided",
                ground_truth=None,
                metrics={"correct": 0.0},
            )

        answer = agent_output.final_answer.lower()
        truth = sample.ground_truth.lower()
        is_correct = truth in answer

        return EnvironmentResult(
            feedback=(
                "Correct!"
                if is_correct
                else f"Incorrect. Expected: {sample.ground_truth}"
            ),
            ground_truth=sample.ground_truth,
            metrics={"correct": 1.0 if is_correct else 0.0},
        )


@dataclass
class ACEStepResult:
    """Result from processing a single sample through the ACE pipeline.

    In sync mode, all fields are populated immediately.
    In async mode, reflection and skill_manager_output may be None initially
    (they are processed in background).
    """

    sample: Sample
    agent_output: AgentOutput
    environment_result: EnvironmentResult
    reflection: Optional[ReflectorOutput]  # None in async mode until processed
    skill_manager_output: Optional[
        SkillManagerOutput
    ]  # None in async mode until processed
    skillbook_snapshot: str

    # Observability metadata
    epoch: int = 0
    step: int = 0
    performance_score: float = 0.0
