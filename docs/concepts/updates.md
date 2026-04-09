# Update Operations

The SkillManager communicates changes to the skillbook through **update operations**. Each operation is a structured instruction to modify the skillbook in a specific way.

## Operation Types

| Type | Description | Required Fields |
|------|-------------|----------------|
| `ADD` | Create a new skill | `section`, `content` |
| `UPDATE` | Modify an existing skill's content | `skill_id`, `content` |
| `REMOVE` | Delete a skill from the skillbook | `skill_id` |

## Examples

### ADD

Adds a new strategy learned from experience:

```json
{
  "type": "ADD",
  "section": "Math Strategies",
  "content": "Break complex problems into smaller steps before computing"
}
```

### UPDATE

Refines an existing strategy:

```json
{
  "type": "UPDATE",
  "skill_id": "math-00001",
  "content": "Break complex problems into smaller steps. Verify each step before proceeding."
}
```

### REMOVE

Prunes a strategy that is consistently harmful:

```json
{
  "type": "REMOVE",
  "skill_id": "math-00003"
}
```

## Update Batches

The SkillManager emits operations as an `UpdateBatch` — one or more operations applied atomically:

```python
from ace import UpdateOperation, UpdateBatch

batch = UpdateBatch(operations=[
    UpdateOperation(type="ADD", section="Debugging", content="Log inputs before errors"),
    UpdateOperation(type="REMOVE", skill_id="debug-00003"),
])

skillbook.apply_update(batch)
```

In batch reflection mode, `ADD` and `UPDATE` operations may also include
`reflection_index` to indicate which reflection in the input tuple primarily
produced the operation.

When an operation is synthesized from multiple reflections, it may instead use
`reflection_indices` to list all contributing reflections. This lets downstream
provenance attach multiple trace sources to one learned skill.

## Skill Tagging

Skill effectiveness is tracked by the **Reflector**, not the SkillManager. In online mode, the Reflector:

1. Scans the agent's reasoning for skill ID citations (e.g. `[general-00042]`)
2. Verifies each cited skill exists in the skillbook
3. Classifies each as `helpful`, `harmful`, or `neutral`
4. Populates `skill_tags` on `ReflectorOutput`

The SkillManager receives these tags and uses them to inform its operations — e.g., consistently harmful skills may be UPDATEd or REMOVEd.

In offline mode (batch analysis of historical traces), skill tagging is skipped since traces may not contain skill citations.

## How Updates Flow

```
Agent cites skill_ids --> Reflector evaluates them (online) --> SkillManager emits ADD/UPDATE/REMOVE
```

1. The **Agent** cites skill IDs it used in its reasoning
2. The **Reflector** classifies each cited skill as helpful/harmful/neutral (`skill_tags`)
3. The **SkillManager** uses the reflection analysis and skill tags to ADD new strategies, UPDATE existing ones, or REMOVE harmful ones

## What to Read Next

- [The Skillbook](skillbook.md) — where operations are applied
- [Three Roles](roles.md) — which role emits which operations
