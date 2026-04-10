# Trace Batch Classification Task

You are preparing 129 CAR-bench agent traces for analysis by a Recursive Reflector (RR) — an iterative code-execution LLM that will read each batch, explore the traces programmatically, and extract reusable strategies into a skillbook.

## Goal

Assign each trace to exactly one batch. Each batch will be analyzed independently by the RR agent. The quality of the resulting skillbook depends on batches being **semantically coherent** — the RR agent should be able to find cross-trace patterns within each batch.

## Constraints

- **Max 30 traces per batch** (hard limit — the RR context window can't handle more)
- **Min 10 traces per batch** (fewer than this doesn't give the RR enough signal)
- Every trace must be assigned to exactly one batch
- Aim for 5-8 batches total

## Classification Criteria (in priority order)

### 1. Task type grouping
The traces have three task types:
- **base** (50 traces): Standard tasks the agent should handle correctly
- **hallucination** (48 traces): Tasks involving requests the agent should recognize as impossible/unsupported and refuse
- **disambiguation** (31 traces): Tasks with ambiguous requests requiring the agent to clarify before acting

**Do NOT mix task types within a batch.** This is the primary split. The previous study showed that mixing task types during training is the #1 failure mode for skillbook quality.

### 2. Domain area grouping (within each task type)
Group traces that involve similar car subsystems:
- **Vehicle controls**: windows, sunroof/sunshade, lights (fog, headlights, low/high beam), trunk
- **Climate**: temperature, AC, fan speed/direction, seat heating, steering wheel heating, defrost, air circulation
- **Navigation**: location search, route planning, waypoints, POI search, multi-stop routes
- **Charging/EV**: battery status, charging specs, charging stations, range, charging time
- **Productivity**: calendar, contacts, phone calls, email
- **Weather**: weather queries (often a prerequisite for vehicle control decisions)

Multi-domain traces (e.g., "navigate to X and check weather") should go with the **primary** domain (the one the user's core intent maps to).

### 3. Semantic similarity of user requests
Within the same (task_type, domain) group, prefer clustering traces where the user requests are semantically similar. For example:
- "Open the sunroof" and "Close the rear window" are both window/sunroof control
- "Set temperature to 22" and "Turn on AC" are both climate
- "Navigate to Berlin" and "Add a stop at the gas station" are both navigation

### 4. Complexity balancing
Try to keep batch complexity somewhat balanced. Don't put all the simple 1-turn traces in one batch and all the complex multi-turn traces in another. Mix within the domain grouping.

## Handling small groups

If a (task_type, domain) group has fewer than 10 traces, merge it with the most semantically related group of the same task type. For example:
- Weather + vehicle controls (weather checks are often prerequisites for vehicle actions)
- Charging + navigation (charging stations involve route/location awareness)
- Productivity is small and can merge with whatever fits best

## Output format

Return a JSON object:

```json
{
  "batches": {
    "batch_name": {
      "description": "Brief description of what unifies this batch",
      "task_type": "base|hallucination|disambiguation",
      "domains": ["primary_domain", "secondary_domain"],
      "trace_ids": ["task_id_1", "task_id_2", ...]
    }
  },
  "summary": {
    "total_traces": 129,
    "num_batches": <N>,
    "batch_sizes": {"batch_name": <size>, ...}
  }
}
```

Use descriptive batch names like `base_vehicle_controls`, `halluc_navigation`, `disambig_climate_nav`, etc.

## Trace metadata

Below is the metadata for all 129 traces. Each entry has:
- `task_id`: unique identifier (prefix = task type)
- `task_type`: base, hallucination, or disambiguation
- `user_request`: what the user asked (first message)
- `tools`: which tools the agent called
- `domains`: auto-detected domain tags (note: many show as "unknown:tool_name" — use the tool name to infer the domain)
- `turns`: number of user messages (proxy for conversation complexity)
- `tool_calls`: total tool calls made

```json
<PASTE trace_summary_for_classification.json HERE>
```
