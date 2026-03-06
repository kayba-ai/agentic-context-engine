# Insight Source Tracing

Insight source tracing tracks the **provenance of every skill** in an ACE skillbook. When the framework learns a new strategy or updates an existing one, metadata is attached describing *why* the skill was created: which sample triggered the learning, what error was observed, and the reflector's diagnosis.

## What's Captured

Every skill that originates from the ACE learning loop carries an `InsightSource` dict in its `sources` list:

| Field | Type | Description |
|-------|------|-------------|
| `sample_question` | `str` | The question/task that triggered the learning (capped at 200 chars) |
| `epoch` | `int` | Training epoch number |
| `step` | `int` | Step index within the epoch |
| `trace_refs` | `list` | Trace references pointing to agent reasoning (see below) |
| `learning_text` | `str?` | The matched `ExtractedLearning.learning` text, resolved via `learning_index` |
| `error_identification` | `str?` | The reflector's raw error diagnosis text (capped at 200 chars) |
| `sample_id` | `str?` | Caller-defined identifier (e.g., trace filename, session ID) |

### How `learning_text` is Populated

The SkillManager outputs a `learning_index` (0-based int) on each ADD/UPDATE operation, referencing which `extracted_learning` from the reflector's output produced the operation. `build_insight_source()` uses this index to look up the exact learning text:

```
Reflector outputs extracted_learnings: [
  {learning: "Verify multiplication", ...},    // index 0
  {learning: "Check boundary conditions", ...} // index 1
]

SkillManager outputs operations: [
  {type: "ADD", content: "Double-check math", learning_index: 0, ...},
]

build_insight_source() reads op.learning_index=0
  → extracted_learnings[0].learning = "Verify multiplication"
  → stores as learning_text on InsightSource
```

If `learning_index` is `None` or out of range, `learning_text` is set to `None`.

### Trace References

Each `TraceReference` in `trace_refs` uses one of two formats:

- **Structured** (when `TraceContext` is available): `step_indices` and `action_types` pointing to specific execution steps.
- **Text fallback** (default): A text excerpt from the agent's reasoning (up to 200 chars) with `excerpt_location` set to `"reasoning"`.

## Querying Sources

### `source_map()` — Raw source data

Returns a dict mapping skill IDs to their source lists. Only includes skills that have sources.

```python
source_map = skillbook.source_map()
for skill_id, sources in source_map.items():
    print(f"{skill_id}: {len(sources)} source(s)")
    for src in sources:
        print(f"  epoch={src['epoch']}")
        if src.get("error_identification"):
            print(f"  diagnosis: {src['error_identification'][:80]}...")
```

### `source_summary()` — Aggregated statistics

Returns distributions of epochs and sample questions across all sources.

```python
summary = skillbook.source_summary()
print(f"Total sources: {summary['total_sources']}")
print(f"Epochs: {summary['epochs']}")
print(f"Top questions: {summary['sample_questions']}")
```

### `source_filter()` — Query by criteria

Filter sources by epoch or sample question (substring match).

```python
# All skills learned in epoch 2
epoch_2 = skillbook.source_filter(epoch=2)

# All skills triggered by a specific question
math_q = skillbook.source_filter(sample_question="2+2")

# Combined criteria (AND logic)
specific = skillbook.source_filter(epoch=1, sample_question="capital")
```

> **Tip for prompt optimization**: When `sample_question` is a placeholder (e.g., `"-"`), use `sample_id` via `source_map()` to identify which trace file produced each skill:
>
> ```python
> for skill_id, sources in skillbook.source_map().items():
>     for src in sources:
>         print(f"{skill_id} ← {src.get('sample_id', 'unknown')}")
> ```

## How to Interpret Sources

Reading an insight source tells you the full story of a skill's creation:

1. **`sample_question`** — What task was being attempted?
2. **`error_identification`** — What went wrong (or right)? The reflector's raw diagnosis.
3. **`trace_refs`** — What was the agent thinking? The text excerpt shows the agent's reasoning at the time.
4. **`learning_text`** — What specific learning was extracted by the reflector? (resolved via `learning_index`)
5. **`sample_id`** — Caller-defined identifier for the source (e.g., trace filename). Useful for prompt optimization workflows where `sample_question` may be a placeholder.

Skills with multiple entries in `sources` were updated across multiple learning iterations. Check `epoch` and `step` to see the evolution.

## Architecture

Insight sources are attached during the ACE learning loop, after the SkillManager produces update operations but before they are applied to the skillbook:

```
Sample (carries optional .id for caller-defined identification)
  → Agent.generate()        produces AgentOutput (with reasoning)
  → Environment.evaluate()  produces EnvironmentResult (with feedback)
  → Reflector.reflect()     produces ReflectorOutput (with error_identification, learnings)
  → SkillManager.update()   produces UpdateOperations (ADD, UPDATE, TAG, REMOVE)
                             ADD/UPDATE ops carry learning_index
  → build_insight_source()  attaches InsightSource metadata to ADD/UPDATE ops
                             resolves learning_index → learning_text
                             copies Sample.id → InsightSource.sample_id
  → Skillbook.apply_update() stores ops with their sources
```

The `build_insight_source()` function in `ace/insight_source.py` handles:
- Building trace references from agent output
- Resolving each operation's `learning_index` to its `learning_text`
- Storing the reflector's raw error identification

This metadata is preserved through serialization (`save_to_file` / `load_from_file`), so you can analyze provenance of skills trained in previous sessions.
