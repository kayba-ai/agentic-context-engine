#!/usr/bin/env python3
# %% [markdown]
# # ACE2 Offline Learning — Interactive Demo
#
# This notebook walks through the ACE2 pipeline engine for **offline**
# (multi-epoch, batch) learning.  It covers:
#
# 1. Defining a task environment
# 2. Building an `OfflineACE` runner (two ways)
# 3. Running single-epoch and multi-epoch training
# 4. Inspecting results and the learned skillbook
# 5. Checkpointing and persistence
# 6. Stepping through the pipeline manually
#
# **Requirements:** `uv sync` (or `pip install -e .` from the repo root).
# Set your LLM API key before running:
# ```bash
# export OPENAI_API_KEY="sk-..."
# ```

# %% [markdown]
# ## 1. Setup & Imports

# %%
import sys
from pathlib import Path

import nest_asyncio

nest_asyncio.apply()

# Ensure the project root is on sys.path so `ace`, `ace2`, and `pipeline`
# are importable regardless of where the notebook kernel starts.
_here = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
_root = _here
for _p in [_here] + list(_here.parents):
    if (_p / "pipeline" / "__init__.py").exists():
        _root = _p
        break
sys.path.insert(0, str(_root))

# Load .env from project root (BEDROCK_API_KEY, OPENAI_API_KEY, etc.)
from dotenv import load_dotenv

load_dotenv(_root / ".env")

from ace.adaptation import EnvironmentResult, Sample, TaskEnvironment
from ace.skillbook import Skillbook
from ace2.pipelines import OfflineACE

print("Imports OK")

# %% [markdown]
# ## 2. Define a Task Environment
#
# A `TaskEnvironment` evaluates the agent's answer against each sample's
# ground truth.  The ACE loop uses this feedback to drive reflection and
# skill updates.


# %%
class CapitalCityEnvironment(TaskEnvironment):
    """Score agent answers against capital-city ground truth."""

    def evaluate(self, sample: Sample, agent_output) -> EnvironmentResult:
        expected = (sample.ground_truth or "").strip().lower()
        predicted = agent_output.final_answer.strip().lower()
        correct = predicted == expected
        return EnvironmentResult(
            feedback=(
                "Correct!" if correct else f"Wrong. Expected: {sample.ground_truth}"
            ),
            ground_truth=sample.ground_truth,
            metrics={"accuracy": 1.0 if correct else 0.0},
        )


env = CapitalCityEnvironment()
print("Environment defined")

# %% [markdown]
# ## 3. Prepare Training Samples
#
# Each `Sample` has a question the agent must answer and a ground-truth
# label the environment uses for scoring.

# %%
samples = [
    Sample(question="What is the capital of France?", ground_truth="Paris"),
    Sample(question="What is the capital of Japan?", ground_truth="Tokyo"),
    Sample(question="What is the capital of Brazil?", ground_truth="Brasilia"),
    Sample(question="What is the capital of Australia?", ground_truth="Canberra"),
    Sample(question="What is the capital of Nigeria?", ground_truth="Abuja"),
]

print(f"Prepared {len(samples)} training samples")

# %% [markdown]
# ## 4. Build OfflineACE — The Easy Way
#
# `from_client` creates the Agent, Reflector, and SkillManager internally
# from a single LLM client.  This is the fastest way to get started.

# %%
import os
from ace.llm_providers.litellm_client import LiteLLMClient

MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
API_KEY = os.getenv("BEDROCK_API_KEY")

client = LiteLLMClient(model=MODEL, api_key=API_KEY)
skillbook = Skillbook()

ace = OfflineACE.from_client(client, skillbook=skillbook)

print(f"OfflineACE ready  |  pipeline steps: {len(ace._steps)}")
print(f"Pipeline provides: {ace.provides}")

# %% [markdown]
# ## 5. Run — Single Epoch
#
# One pass over every sample: Agent answers, environment evaluates,
# Reflector analyses, SkillManager updates the skillbook.
#
# ```
# AgentStep → EvaluateStep → ReflectStep → UpdateStep
# ```

# %%
results = ace.run(samples, env, epochs=1)

print(f"Processed {len(results)} sample(s)\n")
for i, r in enumerate(results, 1):
    if r.error:
        print(f"  [{i}] ERROR: {r.error}")
    else:
        answer = r.output.agent_output.final_answer
        feedback = r.output.environment_result.feedback
        print(f"  [{i}] Q: {r.sample.question}")
        print(f"       A: {answer}  |  {feedback}")

# %%
print(f"\nSkillbook after 1 epoch:")
print(f"  {skillbook.stats()}\n")
print(skillbook)

# %% [markdown]
# ## 6. Run — Multi-Epoch Training
#
# Multiple epochs let the agent revisit samples with an updated skillbook.
# Skills accumulate and refine across passes.

# %%
# Start fresh for a clean multi-epoch demo
client2 = LiteLLMClient(model=MODEL, api_key=API_KEY)
skillbook2 = Skillbook()
ace2 = OfflineACE.from_client(client2, skillbook=skillbook2)

results = ace2.run(samples, env, epochs=3)

print(f"Total results across 3 epochs: {len(results)}")

correct = sum(
    1
    for r in results
    if r.error is None and r.output.environment_result.metrics.get("accuracy", 0) == 1.0
)
print(f"Correct answers: {correct}/{len(results)}")
print(f"Skills learned:  {skillbook2.stats()['skills']}")

# %% [markdown]
# ## 7. Build OfflineACE — The Flexible Way
#
# `from_roles` lets you customise each role individually: different prompt
# templates, deduplication config, reflection window size, etc.

# %%
from ace.roles import Agent, Reflector, SkillManager

client3 = LiteLLMClient(model=MODEL, api_key=API_KEY)
skillbook3 = Skillbook()

ace3 = OfflineACE.from_roles(
    agent=Agent(client3),
    reflector=Reflector(client3),
    skill_manager=SkillManager(client3),
    skillbook=skillbook3,
    reflection_window=5,  # keep last 5 reflections in the rolling window
    max_refinement_rounds=1,  # reflector passes per sample
)

results = ace3.run(samples[:2], env, epochs=1)

for r in results:
    if r.error is None:
        print(f"  Q: {r.sample.question}")
        print(f"  A: {r.output.agent_output.final_answer}")
        print(f"  Insight: {r.output.reflection.key_insight}\n")

# %% [markdown]
# ## 8. Checkpointing
#
# Save the skillbook every N successful samples so you can resume after
# interruption or compare skillbook evolution over time.

# %%
import tempfile

client4 = LiteLLMClient(model=MODEL, api_key=API_KEY)
skillbook4 = Skillbook()
ace4 = OfflineACE.from_client(client4, skillbook=skillbook4)

with tempfile.TemporaryDirectory() as tmpdir:
    results = ace4.run(
        samples,
        env,
        epochs=1,
        checkpoint_interval=2,  # save every 2 successful samples
        checkpoint_dir=tmpdir,
    )

    saved = sorted(Path(tmpdir).glob("*.json"))
    print("Checkpoint files:")
    for f in saved:
        print(f"  {f.name}  ({f.stat().st_size} bytes)")

# %% [markdown]
# ## 9. Persistence — Save & Reload
#
# Save the learned skillbook to disk and reload it in a future session.

# %%
with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "learned_skillbook.json"

    # Save
    skillbook2.save_to_file(str(path))
    print(f"Saved to {path.name}  ({path.stat().st_size} bytes)")

    # Reload
    reloaded = Skillbook.from_file(str(path))
    print(f"Reloaded: {reloaded.stats()}")
    print(f"Skills match: {reloaded.stats() == skillbook2.stats()}")

# %% [markdown]
# ## 10. Manual Pipeline Walkthrough
#
# Under the hood, `OfflineACE.run()` builds a `StepContext` for each
# sample and calls the four-step pipeline.  Here we do it by hand to
# show what each step produces.

# %%
from pipeline import StepContext
from ace2.steps import AgentStep, EvaluateStep, ReflectStep, UpdateStep
from ace2.pipelines import ace_pipeline

client5 = LiteLLMClient(model=MODEL, api_key=API_KEY)
skillbook5 = Skillbook()

pipe = ace_pipeline(
    Agent(client5),
    Reflector(client5),
    SkillManager(client5),
)

sample = samples[0]
ctx = StepContext(
    sample=sample,
    skillbook=skillbook5,
    environment=env,
    epoch=1,
    total_epochs=1,
    step_index=1,
    total_steps=1,
    recent_reflections=(),
)

print(f"Before pipeline:")
print(f"  Skills: {skillbook5.stats()['skills']}")
print(f"  agent_output: {ctx.agent_output}")
print()

# Run the full pipeline on a single context
out = pipe(ctx)

print(f"After pipeline:")
print(f"  Agent answer:     {out.agent_output.final_answer}")
print(f"  Env feedback:     {out.environment_result.feedback}")
print(f"  Reflector insight: {out.reflection.key_insight}")
print(f"  Skills now:       {skillbook5.stats()['skills']}")

# %% [markdown]
# ## 11. Error Handling
#
# Failed samples are captured in `SampleResult.error` — the pipeline
# never drops a sample silently.  Other samples continue processing.

# %%
bad_samples = [
    samples[0],
    Sample(question="", ground_truth=""),  # might trigger edge cases
    samples[1],
]

client6 = LiteLLMClient(model=MODEL, api_key=API_KEY)
ace6 = OfflineACE.from_client(client6)

results = ace6.run(bad_samples, env, epochs=1)

for i, r in enumerate(results, 1):
    status = "OK" if r.error is None else f"FAIL ({r.failed_at})"
    answer = r.output.agent_output.final_answer if r.output else "—"
    print(f"  [{i}] {status:20s}  answer={answer}")

# %% [markdown]
# ---
# ## Summary
#
# | What | How |
# |------|-----|
# | Quick start | `OfflineACE.from_client(llm_client)` |
# | Custom roles | `OfflineACE.from_roles(agent=..., reflector=..., skill_manager=...)` |
# | Single epoch | `ace.run(samples, env, epochs=1)` |
# | Multi-epoch | `ace.run(samples, env, epochs=3)` |
# | Checkpointing | `ace.run(..., checkpoint_interval=10, checkpoint_dir="./ckpts")` |
# | Save skillbook | `skillbook.save_to_file("path.json")` |
# | Load skillbook | `Skillbook.from_file("path.json")` |
# | Manual stepping | Build pipeline with `ace_pipeline()`, call `pipe(ctx)` |
# | Inspect results | `result.output.agent_output`, `.environment_result`, `.reflection` |
#
# The pipeline runs: **AgentStep → EvaluateStep → ReflectStep → UpdateStep**
#
# ReflectStep and UpdateStep run in a background thread pool by default
# (async boundary), so the agent returns fast while learning continues.
