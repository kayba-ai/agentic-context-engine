# Update Operations

The SkillManager communicates changes to the skillbook through **update operations**. Each operation is a structured instruction to modify the skillbook in a specific way.

## Operation Types

| Type | Description | Required Fields |
|------|-------------|----------------|
| `ADD` | Create a new skill | `section`, `content` |
| `UPDATE` | Modify an existing skill's content | `skill_id`, `content` |
| `TAG` | Record a skill as helpful, harmful, or neutral | `skill_id`, `tag` |
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

### TAG

Records whether a strategy helped or hurt:

```json
{
  "type": "TAG",
  "skill_id": "math-00001",
  "tag": "helpful"
}
```

Tags are one of: `helpful`, `harmful`, `neutral`.

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
from ace_next import UpdateOperation, UpdateBatch

batch = UpdateBatch(operations=[
    UpdateOperation(type="ADD", section="Debugging", content="Log inputs before errors"),
    UpdateOperation(type="TAG", skill_id="debug-00001", tag="helpful"),
])

skillbook.apply_update(batch)
```

## How Updates Flow

```
Agent cites skill_ids --> Reflector tags them --> SkillManager emits ADD/UPDATE/REMOVE
```

1. The **Agent** cites skill IDs it used in its reasoning
2. The **Reflector** classifies each cited skill as helpful/harmful/neutral (TAG operations)
3. The **SkillManager** may also ADD new strategies or UPDATE/REMOVE existing ones based on the reflection

## What to Read Next

- [The Skillbook](skillbook.md) — where operations are applied
- [Three Roles](roles.md) — which role emits which operations
