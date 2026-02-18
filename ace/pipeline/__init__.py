"""ACE pipeline package — orchestration interfaces and concrete implementations.

Public surface::

    # Interfaces
    ACEPipeline   — abstract base class for all pipelines
    Step          — Protocol for a single composable step
    StepContext   — mutable state flowing through steps

    # Built-in steps
    AgentStep, EvaluateStep, ReflectStep, UpdateStep

    # Concrete pipelines
    OfflineACE    — multi-epoch training over a fixed sample set
    OnlineACE     — continuous learning from a stream of samples

Adding a new pipeline mode
--------------------------
1. Create ``ace/pipeline/<mode>.py`` with a class that extends ``ACEPipeline``.
2. Override ``_default_steps()`` to return the desired step chain.
3. Implement ``run()`` with the appropriate iteration strategy.
4. Re-export it here.

"""

from .base import ACEBase, ACEPipeline, Step, StepContext
from .steps import AgentStep, EvaluateStep, ReflectStep, UpdateStep
from .offline import OfflineACE
from .online import OnlineACE

__all__ = [
    # Interfaces
    "ACEPipeline",
    "Step",
    "StepContext",
    # Built-in steps
    "AgentStep",
    "EvaluateStep",
    "ReflectStep",
    "UpdateStep",
    # Concrete pipelines
    "OfflineACE",
    "OnlineACE",
    # Backward-compatible alias
    "ACEBase",
]
