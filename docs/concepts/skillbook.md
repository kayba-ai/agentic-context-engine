# The Skillbook

The **Skillbook** is ACE's knowledge store — a structured collection of learned issues and insights. Each entry is called a **skill**.

## What Is a Skill?

A skill is a single learned entry with:

| Field | Description |
|-------|-------------|
| `id` | Unique identifier (e.g., `context-00001`) |
| `section` | Pipeline-facing split: `context` or `harness` |
| `keywords` | Structured topic labels (domain, subsystem, API, behavior) |
| `issue` | The problem this skill captures |
| `insight` | The recommended action; required for `context`, optional for `harness` |
| `active` | Whether the skill participates in normal active views |
| `helpful_count` / `harmful_count` / `neutral_count` | Effectiveness counters |
| `occurrences` | Provenance records linking the skill back to supporting traces |

Example skill:

```json
{
  "id": "context-00001",
  "section": "context",
  "keywords": ["math", "decomposition"],
  "issue": "Complex arithmetic questions are easier to solve when the work is decomposed into smaller verified steps.",
  "insight": "Break the problem into smaller steps before computing.",
  "active": true,
  "helpful_count": 5,
  "harmful_count": 0,
  "neutral_count": 1
}
```

## Skill Lifecycle

Skills go through four stages:

1. **Created** — the SkillManager adds a new skill after a reflection
2. **Tagged** — each time the Agent cites a skill, the Reflector tags it as helpful, harmful, or neutral
3. **Updated** — the SkillManager may refine a skill's issue, insight, or keywords based on new learnings
4. **Removed** — skills are soft-removed by setting `active=False`

These correspond to four [update operations](updates.md): `ADD`, `TAG`, `UPDATE`, `REMOVE`.

## Prompt Format

When the skillbook is rendered as text, it is grouped by section and shows keywords plus issue/insight:

```python
skillbook.as_prompt()  # Markdown format for LLM consumption
```

```
## context
- [context-00001]
  Keywords: math, decomposition
  Issue: Complex arithmetic questions are easier to solve when the work is decomposed into smaller verified steps.
  Insight: Break the problem into smaller steps before computing.

## harness
- [harness-00001]
  Keywords: retries, rate_limit
  Issue: The runtime can stall for long periods when provider 429 retries fan out.
```

## Sections

Skills are organized into exactly two sections:

- `context`: learnings the agent should apply while solving the task
- `harness`: environment or runtime learnings that affect the pipeline itself

Fine-grained categorization lives in `keywords`:

```python
from ace import Skillbook

skillbook = Skillbook()

# Add a context skill with explicit keywords and an action insight
skillbook.add_skill(
    section="context",
    issue="Complex arithmetic questions are easier to solve when the work is decomposed into smaller verified steps.",
    keywords=["math", "decomposition"],
    insight="Break complex problems into smaller steps before computing.",
)
```

## Persistence

```python
# Save
skillbook.save_to_file("strategies.json")
# Writes `strategies.json` plus `strategies.embeddings.npz`

# Load
skillbook = Skillbook.load_from_file("strategies.json")
```

## Statistics

```python
stats = skillbook.stats()
# {"sections": 2, "skills": 15, "active_skills": 14, "by_section": {"context": 11, "harness": 3}}
```

## Deduplication

As the skillbook grows, similar skills can accumulate. The `DeduplicationManager` detects and consolidates them using embedding similarity:

```python
from ace import DeduplicationConfig, DeduplicationManager

config = DeduplicationConfig(
    enabled=True,
    embedding_model="text-embedding-3-small",
    similarity_threshold=0.85,
    within_section_only=True,
    consolidation_model="gpt-4o-mini",
)
dedup = DeduplicationManager(config)
```

`consolidation_model` decides, for each similar pair, whether to merge, delete, keep or update. Without it, deduplication only logs a similarity report and leaves the skillbook unchanged.

When used with a runner, deduplication runs automatically at a configurable interval:

```python
runner = ACE.from_roles(
    ...,
    dedup_manager=dedup,
    dedup_interval=10,  # Every 10 samples
)
```

## Insight Source Tracing

Each skill tracks where it came from using structured provenance records.
Each source stores a stable trace identity (`trace_uid`, `source_system`,
`trace_id`, `display_name`) plus optional learning metadata such as
`epoch`, `step`, `learning_text`, and `trace_refs`.

One skill can carry multiple source records. This is especially useful for
skills synthesized from several traces, where ACE can attach one primary
source plus additional supporting sources instead of collapsing everything
onto a single trace.

Trace identity is exact. In-trace anchors are best-effort: when the trace is
structured enough, `trace_refs` can include `json_path`, `step_indices`,
`message_indices`, and `span_ids`; otherwise ACE falls back to excerpt-only
references.

Typical source record:

```json
{
  "trace_uid": "kayba-hosted:conv-123",
  "source_system": "kayba-hosted",
  "trace_id": "conv-123",
  "display_name": "checkout-failure.md",
  "epoch": 1,
  "step": 3,
  "learning_text": "Check for a next-page token before stopping",
  "trace_refs": [
    {
      "text_excerpt": "The API response included next_page_token.",
      "excerpt_location": "operation.evidence"
    }
  ]
}
```

Query provenance with:

```python
sources = skillbook.source_map()     # skill_id -> source info
summary = skillbook.source_summary() # Aggregated statistics
```

## What to Read Next

- [Update Operations](updates.md) — how ADD, UPDATE, TAG, REMOVE work
- [Three Roles](roles.md) — which role creates, tags, and updates skills
- [Full Pipeline Guide](../guides/full-pipeline.md) — see the skillbook in action
