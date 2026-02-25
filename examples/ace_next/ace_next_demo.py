#!/usr/bin/env python3
# %% [markdown]
# # ACE Next — Interactive Demo
#
# This notebook walks through the refactored `ace_next` pipeline.
# It covers:
#
# 1. **Runners** — `ACE` (full pipeline) and `TraceAnalyser` (learning-only)
# 2. **Steps** — individual pipeline steps and `learning_tail()`
# 3. **Manual pipeline construction** — composing steps by hand
# 4. **Custom environments** — writing your own evaluator
# 5. **Checkpointing & deduplication** — production features
# 6. **Observability with Opik** — pipeline traces and LLM cost tracking
# 7. **Skillbook persistence** — save / reload
# 8. **TraceAnalyser** — learning from pre-recorded traces
#
# **Requirements:** `uv sync` from the repo root.
# Set your LLM API key before running:
# ```bash
# export OPENAI_API_KEY="sk-..."
# ```

# %% [markdown]
# ## 1. Setup & Imports

# %%
import os
import sys
import tempfile
from pathlib import Path

import nest_asyncio

nest_asyncio.apply()

# Ensure the project root is on sys.path so `ace`, `ace_next`, and `pipeline`
# are importable regardless of where the notebook kernel starts.
_here = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
_root = _here
for _p in [_here] + list(_here.parents):
    if (_p / "pipeline" / "__init__.py").exists():
        _root = _p
        break
sys.path.insert(0, str(_root))

from dotenv import load_dotenv

load_dotenv(_root / ".env")

print(f"Project root: {_root}")
print("Setup OK")

# %% [markdown]
# ## 2. Core Imports
#
# Everything lives in `ace_next` — fully self-contained, zero cross-imports.

# %%
from ace_next import (
    # Runners
    ACE,
    TraceAnalyser,
    # Role implementations
    Agent,
    Reflector,
    SkillManager,
    # LLM providers
    LiteLLMClient,
    # Core types
    Sample,
    Skillbook,
    SimpleEnvironment,
    TaskEnvironment,
    EnvironmentResult,
    AgentOutput,
    # Context types
    ACEStepContext,
    SkillbookView,
)

print("All imports OK")

# %% [markdown]
# ## 3. Configure the LLM Client
#
# We use LiteLLM which supports 100+ providers. Swap the model string
# for any provider: `gpt-4o-mini`, `claude-sonnet-4-5-20250929`,
# `bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0`, etc.

# %%
MODEL = os.getenv("ACE_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0")
client = LiteLLMClient(model=MODEL)

print(f"LLM client ready: {MODEL}")

# %% [markdown]
# ## 4. Build Roles
#
# The three ACE roles share the same LLM client. Each is independently
# customisable (prompt templates, retries, etc.).

# %%
agent = Agent(client)
reflector = Reflector(client)
skill_manager = SkillManager(client)

print("Roles created: Agent, Reflector, SkillManager")

# %% [markdown]
# ## 5. Define Training Samples

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
# ---
# ## 6. ACE Runner — Full Adaptive Pipeline
#
# The `ACE` runner is the full closed-loop pipeline:
# ```
# Agent → Evaluate → Reflect → Tag → Update → Apply
# ```
#
# It takes `Sample` objects and an optional `TaskEnvironment`.

# %% [markdown]
# ### 6a. With SimpleEnvironment
#
# `SimpleEnvironment` checks if the ground truth appears in the agent's
# answer (case-insensitive substring match).

# %%
skillbook = Skillbook()

ace = ACE.from_roles(
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    environment=SimpleEnvironment(),
    skillbook=skillbook,
)

results = ace.run(samples[:3], epochs=1)

print(f"Processed {len(results)} samples\n")
for r in results:
    if r.error:
        print(f"  ERROR at {r.failed_at}: {r.error}")
    elif r.output:
        ctx: ACEStepContext = r.output
        answer = ctx.agent_output.final_answer if ctx.agent_output else "N/A"
        print(f"  Q: {r.sample.question}")
        print(f"  A: {answer}")

# %%
print(f"\nSkillbook after 1 epoch:")
print(f"  Stats: {skillbook.stats()}")
for skill in skillbook.skills()[:5]:
    print(f"  - [{skill.id}] {skill.content}")

# %% [markdown]
# ### 6b. Custom Environment
#
# Create your own evaluator by subclassing `TaskEnvironment`.


# %%
class ExactMatchEnvironment(TaskEnvironment):
    """Strict evaluation: answer must exactly match ground truth."""

    def evaluate(self, sample: Sample, agent_output: AgentOutput) -> EnvironmentResult:
        expected = (sample.ground_truth or "").strip().lower()
        predicted = agent_output.final_answer.strip().lower()
        correct = expected in predicted

        return EnvironmentResult(
            feedback=(
                "Correct!" if correct else f"Wrong. Expected: {sample.ground_truth}"
            ),
            ground_truth=sample.ground_truth,
            metrics={"accuracy": 1.0 if correct else 0.0},
        )


print("ExactMatchEnvironment defined")

# %%
skillbook2 = Skillbook()

ace2 = ACE.from_roles(
    agent=Agent(client),
    reflector=Reflector(client),
    skill_manager=SkillManager(client),
    environment=ExactMatchEnvironment(),
    skillbook=skillbook2,
)

results2 = ace2.run(samples[:2], epochs=1)

for r in results2:
    if r.output:
        ctx = r.output
        print(f"  Q: {r.sample.question}")
        print(f"  A: {ctx.agent_output.final_answer if ctx.agent_output else 'N/A'}")
        if ctx.reflection:
            print(f"  Insight: {ctx.reflection.key_insight}")
        print()

# %% [markdown]
# ### 6c. Without Environment
#
# When no environment is provided, `EvaluateStep` is a no-op. The Reflector
# still learns from ground-truth comparison in the trace.

# %%
skillbook3 = Skillbook()

ace3 = ACE.from_roles(
    agent=Agent(client),
    reflector=Reflector(client),
    skill_manager=SkillManager(client),
    skillbook=skillbook3,
    # No environment — EvaluateStep passes through
)

results3 = ace3.run(samples[:2], epochs=1)
print(f"Processed {len(results3)} samples (no environment)")
print(f"Skills learned: {skillbook3.stats()}")

# %% [markdown]
# ### 6d. Multi-Epoch Training
#
# Multiple epochs let the agent revisit samples with an evolving skillbook.
# Skills accumulate and refine across passes.

# %%
skillbook4 = Skillbook()

ace4 = ACE.from_roles(
    agent=Agent(client),
    reflector=Reflector(client),
    skill_manager=SkillManager(client),
    environment=SimpleEnvironment(),
    skillbook=skillbook4,
)

results4 = ace4.run(samples, epochs=2)

print(f"Total results across 2 epochs: {len(results4)}")
print(f"Skills learned: {skillbook4.stats()}")

# Print per-epoch accuracy
for epoch in range(1, 3):
    epoch_results = [r for r in results4 if r.output and r.output.epoch == epoch]
    correct = sum(
        1
        for r in epoch_results
        if r.output
        and r.output.agent_output
        and (r.sample.ground_truth or "").lower()
        in r.output.agent_output.final_answer.lower()
    )
    print(f"  Epoch {epoch}: {correct}/{len(epoch_results)} correct")

# %% [markdown]
# ---
# ## 7. Manual Step-by-Step Pipeline
#
# Under the hood, runners compose `Pipeline` objects from individual steps.
# Here we build one by hand to see exactly what each step does.

# %%
from pipeline import Pipeline
from ace_next.steps import (
    AgentStep,
    EvaluateStep,
    ReflectStep,
    TagStep,
    UpdateStep,
    ApplyStep,
    learning_tail,
)

skillbook5 = Skillbook()
env = SimpleEnvironment()

# Build the full pipeline manually
pipe = Pipeline(
    [
        AgentStep(Agent(client)),
        EvaluateStep(env),
        *learning_tail(Reflector(client), SkillManager(client), skillbook5),
    ]
)

print(f"Pipeline steps: {len(pipe._steps)}")
print(f"  requires: {pipe.requires}")
print(f"  provides: {pipe.provides}")

# %% [markdown]
# ### Run a single sample through the manual pipeline

# %%
sample = samples[0]

# Build the context the same way ACE._build_context() does
ctx = ACEStepContext(
    sample=sample,
    skillbook=SkillbookView(skillbook5),
    epoch=1,
    total_epochs=1,
    step_index=0,
    total_steps=1,
    global_sample_index=0,
)

print(f"Before pipeline:")
print(f"  Skills: {skillbook5.stats()}")
print(f"  agent_output: {ctx.agent_output}")

# Run the full pipeline on a single context
from pipeline.protocol import SampleResult

results_manual = pipe.run([ctx])

print(f"\nAfter pipeline:")
for r in results_manual:
    if r.error:
        print(f"  ERROR: {r.error}")
    elif r.output:
        out: ACEStepContext = r.output
        print(
            f"  Agent answer:      {out.agent_output.final_answer if out.agent_output else 'N/A'}"
        )
        print(
            f"  Reflector insight:  {out.reflection.key_insight if out.reflection else 'N/A'}"
        )
        print(f"  Skills now:        {skillbook5.stats()}")

# %% [markdown]
# ### Using `learning_tail()` as a building block
#
# `learning_tail()` returns the standard learning steps:
# `[ReflectStep, TagStep, UpdateStep, ApplyStep]` with optional
# deduplication and checkpoint steps appended.

# %%
skillbook6 = Skillbook()

tail = learning_tail(
    Reflector(client),
    SkillManager(client),
    skillbook6,
)

print(f"learning_tail() returns {len(tail)} steps:")
for step in tail:
    print(f"  - {type(step).__name__}")

# %% [markdown]
# ---
# ## 8. Checkpointing
#
# Save the skillbook every N successful samples so you can resume after
# interruption or compare skillbook evolution over time.

# %%
skillbook7 = Skillbook()

with tempfile.TemporaryDirectory() as tmpdir:
    ace7 = ACE.from_roles(
        agent=Agent(client),
        reflector=Reflector(client),
        skill_manager=SkillManager(client),
        environment=SimpleEnvironment(),
        skillbook=skillbook7,
        checkpoint_dir=tmpdir,
        checkpoint_interval=2,  # save every 2 successful samples
    )

    results7 = ace7.run(samples, epochs=1)

    saved = sorted(Path(tmpdir).glob("*.json"))
    print("Checkpoint files:")
    for f in saved:
        print(f"  {f.name}  ({f.stat().st_size} bytes)")

# %% [markdown]
# ---
# ## 9. Deduplication
#
# Merge near-duplicate skills to keep the skillbook compact. The
# `DeduplicationManager` runs periodically during training.

# %%
from ace_next import DeduplicationManager, SimilarityDetector
from ace_next.protocols import DeduplicationConfig

skillbook8 = Skillbook()

dedup = DeduplicationManager(DeduplicationConfig(similarity_threshold=0.85))

ace8 = ACE.from_roles(
    agent=Agent(client),
    reflector=Reflector(client),
    skill_manager=SkillManager(client),
    environment=SimpleEnvironment(),
    skillbook=skillbook8,
    dedup_manager=dedup,
    dedup_interval=3,  # run dedup every 3 samples
)

results8 = ace8.run(samples, epochs=1)

print(f"Skills after training with dedup: {skillbook8.stats()}")

# %% [markdown]
# ---
# ## 10. Observability with Opik
#
# `OpikStep` is an explicit, opt-in pipeline step that logs traces to Opik.
# It is **not** wired into `learning_tail()` — you append it yourself.
#
# Three usage patterns:
# 1. **Pipeline traces + LLM cost tracking** — append `OpikStep()` (default)
# 2. **Pipeline traces only** — `OpikStep(register_litellm_callback=False)`
# 3. **LLM cost tracking only** — `register_opik_litellm_callback()` (no step)

# %%
from ace_next import OpikStep, OPIK_AVAILABLE, register_opik_litellm_callback

print(f"Opik available: {OPIK_AVAILABLE}")

# %% [markdown]
# ### 10a. Append OpikStep to a custom pipeline
#
# Place it at the end — after the learning tail.

# %%
if OPIK_AVAILABLE:
    skillbook_opik = Skillbook()

    pipe_with_opik = Pipeline(
        [
            AgentStep(Agent(client)),
            EvaluateStep(SimpleEnvironment()),
            *learning_tail(Reflector(client), SkillManager(client), skillbook_opik),
            OpikStep(project_name="ace-demo"),
        ]
    )
    print(f"Pipeline steps (with Opik): {len(pipe_with_opik._steps)}")
    for step in pipe_with_opik._steps:
        print(f"  - {type(step).__name__}")
else:
    print("Opik not installed — skipping pipeline example")

# %% [markdown]
# ### 10b. LLM-level cost tracking only
#
# If you only want per-LLM-call token/cost logging without pipeline traces,
# use the standalone helper. This registers an `OpikLogger` callback on
# `litellm.callbacks`.

# %%
if OPIK_AVAILABLE:
    registered = register_opik_litellm_callback(project_name="ace-demo")
    print(f"LiteLLM Opik callback registered: {registered}")
else:
    print("Opik not installed — skipping callback example")

# %% [markdown]
# ---
# ## 11. Skillbook Persistence — Save & Reload
#
# Save the learned skillbook to disk and reload it in a future session.

# %%
with tempfile.TemporaryDirectory() as tmpdir:
    path = Path(tmpdir) / "learned_skillbook.json"

    # Save
    skillbook.save_to_file(str(path))
    print(f"Saved to {path.name}  ({path.stat().st_size} bytes)")

    # Reload
    reloaded = Skillbook.from_file(str(path))
    print(f"Reloaded: {reloaded.stats()}")
    print(f"Stats match: {reloaded.stats() == skillbook.stats()}")

# %% [markdown]
# ---
# ## 12. TraceAnalyser — Learning from Pre-Recorded Traces
#
# `TraceAnalyser` runs the learning tail only — no Agent, no Evaluate.
# Feed it raw trace dicts (the same shape ReflectStep expects) and it
# builds a skillbook from historical data.

# %%
# Simulate some pre-recorded traces (e.g., from browser-use history logs)
traces = [
    {
        "question": "Book a flight from NYC to London",
        "reasoning": "Step 1: Opened booking site. Step 2: Searched flights. Step 3: Selected cheapest option.",
        "answer": "Booked flight AA100 for $450",
        "skill_ids": [],
        "feedback": "Task succeeded in 3 steps",
        "ground_truth": None,
    },
    {
        "question": "Find the cheapest hotel in Paris",
        "reasoning": "Step 1: Opened hotel site. Step 2: Set filters. Step 3: Sorted by price. Step 4: Cookie popup blocked view.",
        "answer": "Failed: could not dismiss cookie popup",
        "skill_ids": [],
        "feedback": "Task failed — cookie popup blocked interaction after step 3",
        "ground_truth": None,
    },
    {
        "question": "Check weather in Tokyo",
        "reasoning": "Step 1: Navigated to weather.com. Step 2: Searched Tokyo. Step 3: Read forecast.",
        "answer": "Tokyo: 22C, partly cloudy",
        "skill_ids": [],
        "feedback": "Task succeeded in 3 steps — fast and accurate",
        "ground_truth": None,
    },
]

skillbook9 = Skillbook()

analyser = TraceAnalyser.from_roles(
    reflector=Reflector(client),
    skill_manager=SkillManager(client),
    skillbook=skillbook9,
)

results9 = analyser.run(traces, epochs=1)

print(f"Analysed {len(results9)} traces")
print(f"Skills learned: {skillbook9.stats()}")
for skill in skillbook9.skills()[:5]:
    print(f"  - [{skill.section}] {skill.content}")

# %% [markdown]
# ### Multi-epoch trace analysis
#
# Each epoch re-processes all traces with the evolving skillbook.
# Early epochs extract obvious patterns; later epochs refine.

# %%
skillbook10 = Skillbook()

analyser2 = TraceAnalyser.from_roles(
    reflector=Reflector(client),
    skill_manager=SkillManager(client),
    skillbook=skillbook10,
)

results10 = analyser2.run(traces, epochs=2)

print(f"Total results across 2 epochs: {len(results10)}")
print(f"Skills after 2 epochs: {skillbook10.stats()}")

# %% [markdown]
# ---
# ## 13. Mixed Workflow — TraceAnalyser then ACE
#
# A common pattern: build an initial skillbook from historical traces,
# then deploy with live learning.

# %%
# Phase 1: Build skillbook from historical data
shared_skillbook = Skillbook()

analyser_phase1 = TraceAnalyser.from_roles(
    reflector=Reflector(client),
    skill_manager=SkillManager(client),
    skillbook=shared_skillbook,
)
analyser_phase1.run(traces, epochs=1)

print(f"Phase 1 — TraceAnalyser:")
print(f"  Skills from traces: {shared_skillbook.stats()}")

# Phase 2: Deploy with live ACE learning (reuse the evolved skillbook)
ace_phase2 = ACE.from_roles(
    agent=Agent(client),
    reflector=Reflector(client),
    skill_manager=SkillManager(client),
    environment=SimpleEnvironment(),
    skillbook=shared_skillbook,
)

results_phase2 = ace_phase2.run(samples[:3], epochs=1)

print(f"\nPhase 2 — ACE live learning:")
print(f"  Processed {len(results_phase2)} samples")
print(f"  Skills after live learning: {shared_skillbook.stats()}")

# %% [markdown]
# ---
# ## 14. Error Handling
#
# Failed samples are captured in `SampleResult.error` — the pipeline
# never drops a sample silently. Other samples continue processing.

# %%
bad_samples = [
    samples[0],
    Sample(question="", ground_truth=""),  # edge case: empty question
    samples[1],
]

skillbook11 = Skillbook()
ace11 = ACE.from_roles(
    agent=Agent(client),
    reflector=Reflector(client),
    skill_manager=SkillManager(client),
    environment=SimpleEnvironment(),
    skillbook=skillbook11,
)

results11 = ace11.run(bad_samples, epochs=1)

for i, r in enumerate(results11, 1):
    status = "OK" if r.error is None else f"FAIL ({r.failed_at})"
    if r.output and r.output.agent_output:
        answer = r.output.agent_output.final_answer
    else:
        answer = "N/A"
    print(f"  [{i}] {status:20s}  answer={answer}")

# %% [markdown]
# ---
# ## 15. Inspecting the SkillbookView
#
# Steps receive a read-only `SkillbookView` on the context.
# This prevents accidental mutations from within pipeline steps.

# %%
sb = Skillbook()
view = SkillbookView(sb)

print(f"SkillbookView: {view}")
print(f"  len:    {len(view)}")
print(f"  stats:  {view.stats()}")
print(f"  prompt: {view.as_prompt()[:200]}...")

# Iterate over skills in the view
for skill in view:
    print(f"  - {skill.id}: {skill.content}")

# %% [markdown]
# ---
# ## Summary
#
# | What | How |
# |------|-----|
# | Full pipeline | `ACE.from_roles(agent=..., reflector=..., skill_manager=...)` |
# | With environment | `ACE.from_roles(..., environment=SimpleEnvironment())` |
# | Without environment | `ACE.from_roles(...)` — EvaluateStep is a no-op |
# | Multi-epoch | `ace.run(samples, epochs=3)` |
# | Checkpointing | `ACE.from_roles(..., checkpoint_dir="./ckpts", checkpoint_interval=10)` |
# | Deduplication | `ACE.from_roles(..., dedup_manager=dedup, dedup_interval=5)` |
# | Opik tracing | `Pipeline([...steps..., OpikStep(project_name="my-project")])` |
# | LLM cost tracking | `register_opik_litellm_callback()` |
# | Trace analysis | `TraceAnalyser.from_roles(reflector=..., skill_manager=...)` |
# | Save skillbook | `ace.save("path.json")` or `skillbook.save_to_file("path.json")` |
# | Load skillbook | `Skillbook.from_file("path.json")` |
# | Manual steps | `Pipeline([AgentStep(a), EvaluateStep(e), *learning_tail(r, sm, sb)])` |
# | Learning tail | `learning_tail(reflector, skill_manager, skillbook)` |
#
# **Pipeline:**
# ```
# ACE:            Agent → Evaluate → Reflect → Tag → Update → Apply → [Dedup] → [Checkpoint] → [Opik]
# TraceAnalyser:                     Reflect → Tag → Update → Apply → [Dedup] → [Checkpoint] → [Opik]
# ```
