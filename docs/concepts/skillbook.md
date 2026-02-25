# The Skillbook

The **Skillbook** is ACE's knowledge store — a structured collection of strategies that the agent has learned from experience. Each strategy is called a **skill**.

## What Is a Skill?

A skill is a single strategy entry with:

| Field | Description |
|-------|-------------|
| `id` | Unique identifier (e.g., `math-00001`) |
| `section` | Category grouping (e.g., `"Math Strategies"`) |
| `content` | The strategy text |
| `helpful` | Times this skill contributed to a correct answer |
| `harmful` | Times this skill led to an incorrect answer |
| `neutral` | Times cited but had no clear effect |

Example skill:

```json
{
  "id": "math-00001",
  "section": "Math Strategies",
  "content": "Break complex problems into smaller steps before computing",
  "helpful": 5,
  "harmful": 0,
  "neutral": 1
}
```

## Skill Lifecycle

Skills go through four stages:

1. **Created** — the SkillManager adds a new skill after a reflection
2. **Tagged** — each time the Agent cites a skill, the Reflector tags it as helpful, harmful, or neutral
3. **Updated** — the SkillManager may refine a skill's content based on new learnings
4. **Removed** — skills that are consistently harmful get pruned

These correspond to four [update operations](updates.md): `ADD`, `TAG`, `UPDATE`, `REMOVE`.

## TOON Format

When the skillbook is included in LLM prompts, it uses **TOON** (Token-Oriented Object Notation) — a compact format that saves 16-62% tokens compared to JSON:

```python
skillbook.as_prompt()  # TOON format for LLM consumption
```

```
skills[3]{id	section	content	helpful	harmful	neutral}:
  math-00001	Math Strategies	Break complex problems into smaller steps	5	0	1
  math-00002	Math Strategies	Verify answers by working backwards	3	1	0
  logic-00001	Logic	Check edge cases before concluding	2	0	0
```

For human debugging, use the string representation:

```python
str(skillbook)  # Markdown format for readability
```

## Sections

Skills are organized into sections. Sections emerge naturally from the SkillManager's categorization during learning:

```python
from ace_next import Skillbook

skillbook = Skillbook()

# Add skills to specific sections
skillbook.add_skill(
    section="Math Strategies",
    content="Break complex problems into smaller steps",
    metadata={"helpful": 5, "harmful": 0, "neutral": 1},
)
```

## Persistence

```python
# Save
skillbook.save_to_file("strategies.json")

# Load
skillbook = Skillbook.load_from_file("strategies.json")
```

## Statistics

```python
stats = skillbook.stats()
# {"sections": 3, "skills": 15, "tags": {"helpful": 45, "harmful": 5, "neutral": 10}}
```

## Deduplication

As the skillbook grows, similar skills can accumulate. The `DeduplicationManager` detects and consolidates them using embedding similarity:

```python
from ace_next import DeduplicationConfig, DeduplicationManager

config = DeduplicationConfig(
    enabled=True,
    embedding_model="text-embedding-3-small",
    similarity_threshold=0.85,
    within_section_only=True,
)
dedup = DeduplicationManager(config)
```

When used with a runner, deduplication runs automatically at a configurable interval:

```python
runner = ACE.from_roles(
    ...,
    dedup_manager=dedup,
    dedup_interval=10,  # Every 10 samples
)
```

## Insight Source Tracing

Each skill tracks where it came from — which sample, epoch, and step produced it. Query this with:

```python
sources = skillbook.source_map()     # skill_id -> source info
summary = skillbook.source_summary() # Aggregated statistics
```

## What to Read Next

- [Update Operations](updates.md) — how ADD, UPDATE, TAG, REMOVE work
- [Three Roles](roles.md) — which role creates, tags, and updates skills
- [Full Pipeline Guide](../guides/full-pipeline.md) — see the skillbook in action
