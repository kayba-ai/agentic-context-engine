"""Immutable per-iteration context for the Recursive Reflector inner pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipeline.context import StepContext


@dataclass(frozen=True)
class RRIterationContext(StepContext):
    """Frozen context flowing through the four inner RR steps.

    Each REPL iteration creates a fresh ``RRIterationContext`` with the
    current message history and iteration index.  Inner steps populate the
    remaining fields via ``.replace()``.
    """

    # Input for this iteration
    messages: tuple[dict[str, str], ...] = ()
    iteration: int = 0

    # LLMCallStep output
    llm_response: str | None = None

    # ExtractCodeStep output
    code: str | None = None
    direct_response: str | None = None

    # SandboxExecStep output
    exec_result: Any | None = None  # ExecutionResult from sandbox

    # CheckResultStep output — terminal state
    terminated: bool = False
    reflection: Any | None = None  # ReflectorOutput when FINAL() accepted
    feedback_messages: tuple[dict[str, str], ...] = ()
