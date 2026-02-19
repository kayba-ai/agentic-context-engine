#!/usr/bin/env python3
# %% [markdown]
# # Pipeline Engine — Interactive Demo
#
# This notebook walks through every feature of the generic pipeline engine.
# Each cell is self-contained — run them top to bottom.
#
# **No external dependencies** — only the `pipeline/` package.

# %% [markdown]
# ## Setup & Imports

# %%
import sys, os, time
from types import MappingProxyType

# ensure the project root is on the path (needed when running from examples/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..") if "__file__" in dir() else "..")

from pipeline import (
    Pipeline,
    Branch,
    MergeStrategy,
    StepContext,
    SampleResult,
    PipelineOrderError,
    BranchError,
)

def show(results: list[SampleResult]) -> None:
    """Pretty-print a list of SampleResult."""
    for r in results:
        tag = "OK" if r.error is None else f"FAIL @ {r.failed_at}"
        print(f"  [{tag}] sample={r.sample!r}")
        if r.output:
            named = {k: getattr(r.output, k) for k in ("agent_output", "environment_result", "reflection") if getattr(r.output, k) is not None}
            if named:
                print(f"         fields:   {named}")
            meta = dict(r.output.metadata)
            if meta:
                print(f"         metadata: {meta}")
        if r.error:
            print(f"         error:    {r.error}")

# %% [markdown]
# ## Step Definitions
#
# A **step** is any object with:
# - `requires: frozenset[str]` — metadata keys it reads
# - `provides: frozenset[str]` — metadata keys it writes
# - `__call__(ctx: StepContext) -> StepContext`
#
# No base class — pure duck typing via `StepProtocol`.

# %%
class Tokenize:
    """Split sample text into words, store token list and count."""
    requires = frozenset()
    provides = frozenset({"tokens", "word_count"})

    def __call__(self, ctx: StepContext) -> StepContext:
        tokens = str(ctx.sample).split()
        print(f"    [Tokenize]  '{ctx.sample}' → {len(tokens)} tokens")
        return ctx.replace(metadata=MappingProxyType({
            **ctx.metadata, "tokens": tokens, "word_count": len(tokens),
        }))


class Uppercase:
    """Uppercase each token. Requires 'tokens' in metadata."""
    requires = frozenset({"tokens"})
    provides = frozenset({"upper_tokens"})

    def __call__(self, ctx: StepContext) -> StepContext:
        upper = [t.upper() for t in ctx.metadata["tokens"]]
        print(f"    [Uppercase] {ctx.metadata['tokens']} → {upper}")
        return ctx.replace(metadata=MappingProxyType({**ctx.metadata, "upper_tokens": upper}))


class Reverse:
    """Reverse each token. Designed to run in parallel with Uppercase."""
    requires = frozenset({"tokens"})
    provides = frozenset({"reversed_tokens"})

    def __call__(self, ctx: StepContext) -> StepContext:
        rev = [t[::-1] for t in ctx.metadata["tokens"]]
        print(f"    [Reverse]   {ctx.metadata['tokens']} → {rev}")
        return ctx.replace(metadata=MappingProxyType({**ctx.metadata, "reversed_tokens": rev}))


class Summarize:
    """Combine processed metadata into a final agent_output string."""
    requires = frozenset({"upper_tokens", "reversed_tokens", "word_count"})
    provides = frozenset({"agent_output"})

    def __call__(self, ctx: StepContext) -> StepContext:
        summary = (
            f"{ctx.metadata['word_count']} words | "
            f"upper={ctx.metadata['upper_tokens']} | "
            f"rev={ctx.metadata['reversed_tokens']}"
        )
        print(f"    [Summarize] → {summary}")
        return ctx.replace(agent_output=summary)


class Boom:
    """Always fails — used to demonstrate error handling."""
    requires = frozenset()
    provides = frozenset()

    def __call__(self, ctx: StepContext) -> StepContext:
        raise RuntimeError(f"Boom on sample={ctx.sample!r}!")

# %% [markdown]
# ---
# ## 1. Basic Linear Pipeline
#
# Chain steps with `.then()`. The pipeline infers its **contracts**
# (`requires` / `provides`) from the step chain automatically.

# %%
pipe = Pipeline().then(Tokenize()).then(Uppercase())

print("Pipeline contracts:")
print(f"  requires = {pipe.requires}   ← external inputs the caller must provide")
print(f"  provides = {pipe.provides}   ← everything the pipeline writes")
print()

results = pipe.run(["hello world", "pipeline engine demo"])
show(results)

# %% [markdown]
# ---
# ## 2. Contract Validation
#
# The engine validates step ordering at **construction time**.
# If a step needs a field that a *later* step provides → `PipelineOrderError`.
#
# Fields not provided by *any* step are treated as **external inputs** — no error.

# %%
# Wrong order: Uppercase needs 'tokens', but Tokenize comes after
print("Trying: Pipeline().then(Uppercase()).then(Tokenize())\n")

try:
    Pipeline().then(Uppercase()).then(Tokenize())
except PipelineOrderError as e:
    print(f"  Caught PipelineOrderError:\n  {e}\n")

# External input: 'tokens' not produced by anyone → valid, caller must provide it
p = Pipeline().then(Uppercase())
print(f"External input is OK:  requires = {p.requires}")

# %% [markdown]
# ---
# ## 3. Branch — Parallel Fork/Join
#
# `.branch()` fans out to N child pipelines in parallel, then **merges**
# their outputs back into a single `StepContext`.
#
# ```
#                  ┌── Uppercase ──┐
#   Tokenize ──►──┤               ├──► Summarize
#                  └── Reverse  ──┘
# ```

# %%
pipe = (
    Pipeline()
    .then(Tokenize())
    .branch(
        Pipeline().then(Uppercase()),
        Pipeline().then(Reverse()),
        merge=MergeStrategy.RAISE_ON_CONFLICT,
    )
    .then(Summarize())
)

print(f"requires = {pipe.requires}")
print(f"provides = {pipe.provides}\n")

results = pipe.run(["fork join"])
show(results)

# %% [markdown]
# ---
# ## 4. Merge Strategies
#
# When branches write the **same named field**, the merge strategy decides what happens:
#
# | Strategy | Behaviour |
# |---|---|
# | `RAISE_ON_CONFLICT` | `ValueError` if any named field differs (metadata always LWW) |
# | `LAST_WRITE_WINS` | Last branch's value wins for every field |
# | `NAMESPACED` | Each branch stored at `metadata["branch_N"]`, no conflict possible |

# %%
class WriteAnswer:
    requires = frozenset()
    provides = frozenset({"agent_output"})
    def __init__(self, val: str):
        self.val = val
    def __call__(self, ctx):
        return ctx.replace(agent_output=self.val)

ctx = StepContext(sample="q")

# %%
# a) RAISE_ON_CONFLICT — two branches write different values → error
print("a) RAISE_ON_CONFLICT with conflict:\n")

b = Branch(
    Pipeline().then(WriteAnswer("yes")),
    Pipeline().then(WriteAnswer("no")),
    merge=MergeStrategy.RAISE_ON_CONFLICT,
)
try:
    b(ctx)
except ValueError as e:
    print(f"   Caught ValueError: {e}")

# %%
# b) LAST_WRITE_WINS — second branch always wins
print("b) LAST_WRITE_WINS:\n")

b = Branch(
    Pipeline().then(WriteAnswer("yes")),
    Pipeline().then(WriteAnswer("no")),
    merge=MergeStrategy.LAST_WRITE_WINS,
)
out = b(ctx)
print(f"   agent_output = {out.agent_output!r}   ← second branch wins")

# %%
# c) NAMESPACED — each branch isolated, accessible via metadata key
print("c) NAMESPACED:\n")

b = Branch(
    Pipeline().then(WriteAnswer("yes")),
    Pipeline().then(WriteAnswer("no")),
    merge=MergeStrategy.NAMESPACED,
)
out = b(ctx)
print(f"   agent_output          = {out.agent_output!r}   ← from first branch")
print(f"   branch_0.agent_output = {out.metadata['branch_0'].agent_output!r}")
print(f"   branch_1.agent_output = {out.metadata['branch_1'].agent_output!r}")

# %% [markdown]
# ---
# ## 5. Error Handling
#
# Every sample produces a `SampleResult` — nothing is dropped silently.
# A failing sample sets `error` and `failed_at`; other samples continue.

# %%
print("All samples fail:\n")
results = Pipeline().then(Tokenize()).then(Boom()).run(["good luck"])
show(results)

# %%
print("Mixed success / failure:\n")

class MaybeBoom:
    requires = frozenset()
    provides = frozenset()
    def __call__(self, ctx):
        if "bad" in str(ctx.sample):
            raise RuntimeError("bad sample!")
        return ctx

results = Pipeline().then(Tokenize()).then(MaybeBoom()).run(["ok", "bad input", "fine"])
show(results)

# %% [markdown]
# ---
# ## 6. Async Boundary — Fire-and-Forget Background
#
# Set `async_boundary = True` on a step. Everything from that step onward
# runs in a **background thread**. `run()` returns immediately.
#
# ```
#   Foreground (fast)         Background (slow)
#   ┌──────────┐             ┌───────────┐
#   │ Tokenize │ ──────►──── │ SlowScore │
#   └──────────┘             └───────────┘
#        │                        │
#     run() returns          updated later
# ```
#
# Call `wait_for_background()` to join all background threads.

# %%
class SlowScore:
    """Expensive scoring step that runs in background."""
    requires = frozenset()
    provides = frozenset({"score"})
    async_boundary = True
    max_workers = 2

    def __call__(self, ctx: StepContext) -> StepContext:
        time.sleep(0.1)
        score = ctx.metadata.get("word_count", 0) * 10
        print(f"    [SlowScore] sample={ctx.sample!r} score={score}  (background)")
        return ctx.replace(metadata=MappingProxyType({**ctx.metadata, "score": score}))

pipe = Pipeline().then(Tokenize()).then(SlowScore())

t0 = time.monotonic()
results = pipe.run(["fast return", "also fast"], workers=2)
elapsed = time.monotonic() - t0

print(f"\nrun() returned in {elapsed:.3f}s — background still scoring\n")
show(results)

# %%
# Now wait for background to finish and inspect the updated results
print("Waiting for background...\n")
pipe.wait_for_background(timeout=5.0)
print("Done!\n")
show(results)

# %% [markdown]
# ---
# ## 7. Nested Pipelines
#
# A `Pipeline` satisfies `StepProtocol`, so it can be used as a step
# inside another pipeline. Contracts are inferred recursively.

# %%
inner = Pipeline().then(Tokenize()).then(Uppercase())
outer = Pipeline().then(inner).then(Reverse())

print(f"Inner: requires={inner.requires}, provides={inner.provides}")
print(f"Outer: requires={outer.requires}, provides={outer.provides}\n")

results = outer.run(["nested demo"])
show(results)

# %% [markdown]
# ---
# ## 8. Workers — Concurrent Sample Processing
#
# The `workers` parameter on `run()` controls how many samples are processed
# in parallel (via an `asyncio.Semaphore` in the foreground event loop).
# This is independent of `max_workers` on individual steps.

# %%
class SlowStep:
    requires = frozenset()
    provides = frozenset({"done"})
    def __call__(self, ctx):
        time.sleep(0.1)
        return ctx.replace(metadata=MappingProxyType({**ctx.metadata, "done": True}))

samples = [f"s{i}" for i in range(6)]
pipe = Pipeline().then(SlowStep())

t0 = time.monotonic()
pipe.run(samples, workers=1)
seq = time.monotonic() - t0

t0 = time.monotonic()
pipe.run(samples, workers=6)
par = time.monotonic() - t0

print(f"  workers=1 : {seq:.2f}s")
print(f"  workers=6 : {par:.2f}s")
print(f"  speedup   : {seq / par:.1f}x")

# %% [markdown]
# ---
# ## Summary
#
# | Concept | API |
# |---|---|
# | Linear chain | `Pipeline().then(A()).then(B())` |
# | Contract inference | `pipe.requires`, `pipe.provides` |
# | Parallel fork/join | `.branch(Pipeline().then(A()), Pipeline().then(B()))` |
# | Merge control | `merge=MergeStrategy.RAISE_ON_CONFLICT / LAST_WRITE_WINS / NAMESPACED` |
# | Error isolation | `SampleResult.error`, `SampleResult.failed_at` |
# | Background execution | `async_boundary = True` on a step class |
# | Background join | `pipe.wait_for_background(timeout=...)` |
# | Nesting | Use a `Pipeline` as a step inside another `Pipeline` |
# | Sample concurrency | `pipe.run(samples, workers=N)` |
#
# The pipeline engine is **domain-agnostic** — it knows nothing about ACE.
# The `ace2/` package will add domain-specific steps (Agent, Evaluate,
# Reflect, Update) on top of this engine.
