"""AgentStep — runs the Agent role to produce an answer."""

from __future__ import annotations

from ..core.context import ACEStepContext
from ..core.outputs import AgentOutput
from ..protocols import AgentLike


class AgentStep:
    """Execute the Agent role against the current sample and skillbook.

    Reads the skillbook via ``ctx.skillbook`` (a ``SkillbookView``).
    """

    requires = frozenset({"sample", "skillbook"})
    provides = frozenset({"agent_output"})

    def __init__(self, agent: AgentLike) -> None:
        self.agent = agent

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        agent_output: AgentOutput = self.agent.generate(
            question=ctx.sample.question,
            context=ctx.sample.context,
            skillbook=ctx.skillbook,
            sample=ctx.sample,
        )
        return ctx.replace(agent_output=agent_output)
