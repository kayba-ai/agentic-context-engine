#!/usr/bin/env python3
"""Compose a custom ACE pipeline from individual steps.

Demonstrates:
  1. Pipeline composition with a single import line from ace_next
  2. Adding a custom step to the pipeline
  3. Inspecting runner presets via build_steps()
  4. Running the pipeline directly via ACERunner

Requires:
  pip install ace-framework
  export OPENAI_API_KEY=...  # or any LiteLLM-supported provider
"""

from __future__ import annotations

from ace_next import (
    # Pipeline engine
    Pipeline,
    StepProtocol,
    # ACE context
    ACEStepContext,
    ACERunner,
    # Roles
    Agent,
    Reflector,
    SkillManager,
    # Steps
    AgentStep,
    EvaluateStep,
    learning_tail,
    # Types
    ACE,
    LiteLLMClient,
    Sample,
    Skillbook,
    SimpleEnvironment,
)

# ------------------------------------------------------------------
# 1. Custom step — print the agent's answer between execute and learn
# ------------------------------------------------------------------


class LogAnswerStep:
    """A custom step that logs the agent's output before learning."""

    requires = frozenset({"agent_output"})
    provides = frozenset()

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        print(f"  -> Agent answered: {ctx.agent_output.final_answer}")
        return ctx


# ------------------------------------------------------------------
# 2. Compose a custom pipeline
# ------------------------------------------------------------------

MODEL = "gpt-4o-mini"

llm = LiteLLMClient(model=MODEL)
skillbook = Skillbook()

pipe = Pipeline(
    [
        AgentStep(Agent(llm)),
        EvaluateStep(SimpleEnvironment()),
        LogAnswerStep(),  # <-- custom step injected here
        *learning_tail(Reflector(llm), SkillManager(llm), skillbook),
    ]
)

print(f"Custom pipeline: {len(pipe._steps)} steps")
print(f"  requires: {pipe.requires}")
print(f"  provides: {pipe.provides}")

# ------------------------------------------------------------------
# 3. Run using ACERunner
# ------------------------------------------------------------------

runner = ACERunner(pipeline=pipe, skillbook=skillbook)

samples = [
    Sample(question="What is 2+2?", context="", ground_truth="4"),
    Sample(question="Capital of France?", context="", ground_truth="Paris"),
]

# Note: ACERunner._build_context is abstract, so we use ACE which
# provides _build_context for Sample objects.
runner_ace = ACE(pipeline=pipe, skillbook=skillbook)
results = runner_ace.run(samples, epochs=1)

print(f"\nResults: {len(results)} samples processed")
print(f"Skills learned: {len(skillbook.skills())}")

# ------------------------------------------------------------------
# 4. Inspect and modify a runner's default steps with build_steps()
# ------------------------------------------------------------------

print("\n--- Inspecting ACE.build_steps() ---")
default_steps = ACE.build_steps(
    agent=Agent(llm),
    reflector=Reflector(llm),
    skill_manager=SkillManager(llm),
    environment=SimpleEnvironment(),
    skillbook=Skillbook(),
)

for i, step in enumerate(default_steps):
    print(f"  [{i}] {type(step).__name__}")

# Modify: insert LogAnswerStep after EvaluateStep
default_steps.insert(2, LogAnswerStep())
print(f"\nAfter inserting LogAnswerStep: {len(default_steps)} steps")

# Build a new pipeline from the modified steps
modified_pipe = Pipeline(default_steps)
print(f"Modified pipeline ready with {len(modified_pipe._steps)} steps")
