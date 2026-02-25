"""Data contracts for the ACE pipeline — samples, environments, and step results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

from .outputs import AgentOutput


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
    """Feedback returned by the task environment after evaluating agent output."""

    feedback: str
    ground_truth: Optional[str]
    metrics: Dict[str, float] = field(default_factory=dict)


class TaskEnvironment(ABC):
    """Abstract interface for evaluating agent outputs."""

    @abstractmethod
    def evaluate(self, sample: Sample, agent_output: AgentOutput) -> EnvironmentResult:
        """Evaluate the agent's output for a given sample."""


class SimpleEnvironment(TaskEnvironment):
    """Built-in environment that checks if ground truth appears in the answer."""

    def evaluate(self, sample: Sample, agent_output: AgentOutput) -> EnvironmentResult:
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
