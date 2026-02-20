import os
from dotenv import load_dotenv

from ace import (
    OfflineACE,
    AgentStep, EvaluateStep, ReflectStep, UpdateStep,
    Step, StepContext,
    Agent, Reflector, SkillManager,
    Skillbook, Sample, SimpleEnvironment,
)
from ace.llm_providers import LiteLLMClient
from ace.prompt_manager import PromptManager

load_dotenv()

# ---------------------------------------------------------------------------
# Optional: define a custom step — just needs __call__(ctx) -> ctx
# ---------------------------------------------------------------------------

class LogStep:
    """Prints a one-liner after evaluation, before reflection."""
    requires = frozenset({"environment_result"})
    provides = frozenset()

    def __call__(self, ctx: StepContext) -> StepContext:
        feedback = ctx.environment_result.feedback[:70] if ctx.environment_result else "—"
        print(f"  [{ctx.step_index}/{ctx.total_steps}] {feedback}")
        return ctx

# ---------------------------------------------------------------------------
# Wire up the LLM and the three ACE roles
# ---------------------------------------------------------------------------

llm = LiteLLMClient(model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0")
prompt_mgr = PromptManager()

skillbook    = Skillbook()
agent        = Agent(llm,         prompt_template=prompt_mgr.get_agent_prompt())
reflector    = Reflector(llm,     prompt_template=prompt_mgr.get_reflector_prompt())
skill_mgr    = SkillManager(llm,  prompt_template=prompt_mgr.get_skill_manager_prompt())

# ---------------------------------------------------------------------------
# Build the pipeline — inject a custom step between Evaluate and Reflect
# ---------------------------------------------------------------------------

pipeline = OfflineACE(
    steps=[
        AgentStep(agent),
        EvaluateStep(),
        LogStep(),               # <-- custom: any callable(ctx) -> ctx works here
        ReflectStep(reflector),
        UpdateStep(skill_mgr),
    ],
    skillbook=skillbook,
)

# ---------------------------------------------------------------------------
# Learn from examples (equivalent to agent.learn() in the old code)
# ---------------------------------------------------------------------------

samples = [
    Sample(
        question="If all birds fly, can penguins fly?",
        ground_truth="No, penguins cannot fly despite being birds.",
    ),
    Sample(
        question="Give me a strategy to avoid hidden assumptions in syllogisms.",
        ground_truth="Question each premise explicitly before accepting it.",
    ),
]

print("Learning...\n")
results = pipeline.run(samples, SimpleEnvironment(), epochs=1)

# ---------------------------------------------------------------------------
# Inspect learned content
# ---------------------------------------------------------------------------

print(f"\nLearned skills: {len(skillbook.skills())}")

# Use the agent directly with the evolved skillbook — equivalent to agent.ask()
answer = agent.generate(
    question="If all mammals breathe air, do whales breathe air?",
    context="",
    skillbook=skillbook,
)
print(f"Answer: {answer.final_answer}")

# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

skillbook.save_to_file("skillbook_v1.json")
print("\nSkillbook saved.")
