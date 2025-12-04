#!/usr/bin/env python3
"""
Minimal example of integrating ACE with a custom agentic system.

This demonstrates the core pattern without any external framework dependencies.
Shows how to wrap any agent with ACE learning capabilities.

Requirements:
    pip install ace-framework
    export OPENAI_API_KEY="your-key"
"""

from dataclasses import dataclass
from ace import Skillbook, Reflector, SkillManager, LiteLLMClient
from ace.integrations.base import wrap_skillbook_context
from ace.roles import AgentOutput


# Simulated custom agent (replace with your actual agent)
@dataclass
class AgentResult:
    output: str
    success: bool
    steps: int = 0


class MyCustomAgent:
    """Example custom agent - replace with your actual agent."""

    def execute(self, task: str) -> AgentResult:
        # Your agent logic here
        # This is just a placeholder
        return AgentResult(output=f"Processed: {task}", success=True, steps=1)


# ACE Wrapper for your custom agent
class ACEWrappedAgent:
    """
    Wraps any agent with ACE learning capabilities.

    Pattern:
    1. Inject skillbook context before execution
    2. Execute agent normally
    3. Learn from results (Reflector + SkillManager)
    """

    def __init__(self, agent, ace_model: str = "gpt-4o-mini", is_learning: bool = True):
        self.agent = agent
        self.skillbook = Skillbook()
        self.is_learning = is_learning

        # Create ACE learning components
        self.llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(self.llm)
        self.skill_manager = SkillManager(self.llm)

    def run(self, task: str) -> AgentResult:
        """Execute task with ACE learning."""
        # 1. Inject skillbook context (if available)
        enhanced_task = task
        if self.is_learning and self.skillbook.skills():
            skillbook_context = wrap_skillbook_context(self.skillbook)
            enhanced_task = f"{task}\n\n{skillbook_context}"

        # 2. Execute agent
        result = self.agent.execute(enhanced_task)

        # 3. Learn from execution
        if self.is_learning:
            self._learn(task, result)

        return result

    def _learn(self, task: str, result: AgentResult):
        """Run ACE learning pipeline."""
        # Create adapter for Reflector (required interface)
        agent_output = AgentOutput(
            reasoning=f"Task: {task}",
            final_answer=result.output,
            skill_ids=[],  # External agent, not using ACE Agent
            raw={"steps": result.steps, "success": result.success},
        )

        # Build feedback
        feedback = (
            f"Task {'succeeded' if result.success else 'failed'} "
            f"in {result.steps} steps.\n"
            f"Output: {result.output}"
        )

        # Reflect: Analyze what went right/wrong
        reflection = self.reflector.reflect(
            question=task,
            agent_output=agent_output,
            skillbook=self.skillbook,
            ground_truth=None,
            feedback=feedback,
        )

        # Curate: Generate skillbook updates
        skill_manager_output = self.skill_manager.curate(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context=f"task: {task}\nfeedback: {feedback}",
            progress=f"Task: {task}",
        )

        # Apply updates
        self.skillbook.apply_update(skill_manager_output.update)

    def save_skillbook(self, path: str):
        """Save learned knowledge."""
        self.skillbook.save_to_file(path)

    def load_skillbook(self, path: str):
        """Load previously learned knowledge."""
        self.skillbook = Skillbook.load_from_file(path)


def main():
    print("🤖 Custom ACE Integration Example")
    print("=" * 50)

    # Create your custom agent
    my_agent = MyCustomAgent()

    # Wrap with ACE learning
    ace_agent = ACEWrappedAgent(my_agent, is_learning=True)

    # Run tasks - ACE learns from each
    tasks = ["Process user data", "Validate inputs", "Generate report"]

    for i, task in enumerate(tasks, 1):
        print(f"\n📋 Task {i}: {task}")
        result = ace_agent.run(task)
        print(f"✅ Result: {result.output}")
        print(f"📚 Learned {len(ace_agent.skillbook.skills())} strategies so far")

    # Show learned strategies
    print("\n" + "=" * 50)
    print("🎯 Learned Strategies:")
    print("=" * 50)
    for i, skill in enumerate(ace_agent.skillbook.skills()[:5], 1):
        print(f"{i}. {skill.content}")
        print(f"   Score: +{skill.helpful}/-{skill.harmful}\n")

    # Save for reuse
    ace_agent.save_skillbook("custom_agent_learned.json")
    print("💾 Skillbook saved to custom_agent_learned.json")
    print("\n✨ Next time, load this skillbook to start with learned knowledge!")


if __name__ == "__main__":
    main()
