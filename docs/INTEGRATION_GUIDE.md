# Integration Guide: Building Custom ACE Integrations

Quick guide for integrating ACE learning with your custom agentic system.

## Overview

ACE integration pattern:
1. **Your agent executes** (no ACE Generator needed)
2. **Inject playbook context** before execution (optional)
3. **Learn from results** after execution (Reflector + Curator)

See [`ace/integrations/browser_use.py`](../ace/integrations/browser_use.py) for reference implementation.

---

## Quick Start (5 min)

### Installation
```bash
pip install ace-framework
```

### Minimal Example (50 lines)

```python
from ace import Playbook, Reflector, Curator, LiteLLMClient
from ace.integrations.base import wrap_playbook_context
from ace.roles import GeneratorOutput

# Setup
playbook = Playbook()
llm = LiteLLMClient(model="gpt-4o-mini", max_tokens=2048)
reflector = Reflector(llm)
curator = Curator(llm)

# Your agent's task
task = "Process user request"

# 1. BEFORE: Inject learned strategies (optional)
if playbook.bullets():
    enhanced_task = f"{task}\n\n{wrap_playbook_context(playbook)}"
else:
    enhanced_task = task

# 2. EXECUTE: Your agent runs
result = your_agent.execute(enhanced_task)  # Your custom agent

# 3. AFTER: Learn from execution
# Build adapter for Reflector
generator_output = GeneratorOutput(
    reasoning=f"Task: {task}",
    final_answer=result.output,
    bullet_ids=[],  # Not using Generator
    raw={"success": result.success}
)

# Build feedback
feedback = f"Task {'succeeded' if result.success else 'failed'}"

# Reflect
reflection = reflector.reflect(
    question=task,
    generator_output=generator_output,
    playbook=playbook,
    ground_truth=None,
    feedback=feedback
)

# Curate
curator_output = curator.curate(
    reflection=reflection,
    playbook=playbook,
    question_context=f"task: {task}",
    progress=f"Task: {task}"
)

# Update playbook
playbook.apply_delta(curator_output.delta)

# Save for reuse
playbook.save_to_file("learned.json")
```

---

## Core Components

### 1. `wrap_playbook_context(playbook)`
Formats learned strategies for your agent.

```python
from ace.integrations.base import wrap_playbook_context

context = wrap_playbook_context(playbook)
# Returns formatted markdown with strategies + usage instructions
```

### 2. Reflector
Analyzes what went right/wrong.

```python
reflection = reflector.reflect(
    question=task,
    generator_output=GeneratorOutput(...),  # Adapter pattern
    playbook=playbook,
    ground_truth=None,  # Optional
    feedback="Execution succeeded/failed because..."
)
```

### 3. Curator
Generates playbook updates.

```python
curator_output = curator.curate(
    reflection=reflection,
    playbook=playbook,
    question_context="Domain info",
    progress="Current task description"
)

playbook.apply_delta(curator_output.delta)
```

### 4. Rich Feedback for External Agents

When integrating with agents that provide detailed execution traces (like browser-use), ACE can extract and use this information for better learning.

**Example: Browser-Use Integration**

The browser-use `AgentHistoryList` object provides:
- `model_thoughts()` - Agent's reasoning at each step
- `model_actions()` - Actions taken with parameters
- `action_results()` - Detailed results for each action
- `urls()` - URLs visited during execution
- `errors()` - Per-step error information
- `total_duration_seconds()` - Execution timing

**Building Rich Feedback:**

```python
def build_rich_feedback(history, success, error=None):
    """Extract comprehensive trace from agent history."""
    feedback_parts = []

    # Basic info
    steps = history.number_of_steps()
    feedback_parts.append(f"Task {'succeeded' if success else 'failed'} in {steps} steps")

    # Agent reasoning
    if hasattr(history, 'model_thoughts'):
        thoughts = history.model_thoughts()
        feedback_parts.append("\nAgent Reasoning:")
        for i, thought in enumerate(thoughts[:3], 1):  # First 3 steps
            feedback_parts.append(f"  Step {i}: {thought.next_goal}")

    # Actions taken
    if hasattr(history, 'model_actions'):
        actions = history.model_actions()
        action_names = [list(a.keys())[0] for a in actions[:5]]
        feedback_parts.append(f"\nActions: {', '.join(action_names)}")

    # URLs and errors
    if hasattr(history, 'urls'):
        urls = history.urls()
        feedback_parts.append(f"URLs: {', '.join(filter(None, urls[:3]))}")

    return "\n".join(feedback_parts)

# Use in Reflector
feedback = build_rich_feedback(history, success=True)
reflection = reflector.reflect(
    question=task,
    generator_output=generator_output,
    playbook=playbook,
    feedback=feedback  # Rich feedback enables better learning
)
```

**Benefits of Rich Feedback:**
- Learns action sequencing patterns ("Wait after goto")
- Understands timing requirements ("Page loads slowly")
- Recognizes error patterns ("Element not visible before scroll")
- Captures domain-specific knowledge ("news.ycombinator.com requires 2s wait")

**See also:** `ace/integrations/browser_use.py` for complete implementation.

---

## Wrapper Class Pattern

```python
class MyACEAgent:
    def __init__(self, agent, ace_model="gpt-4o-mini"):
        self.agent = agent
        self.playbook = Playbook()
        self.llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(self.llm)
        self.curator = Curator(self.llm)
        self.is_learning = True

    def run(self, task):
        # Inject context
        if self.is_learning and self.playbook.bullets():
            task = f"{task}\n\n{wrap_playbook_context(self.playbook)}"

        # Execute
        result = self.agent.execute(task)

        # Learn
        if self.is_learning:
            self._learn(task, result)

        return result

    def _learn(self, task, result):
        # Create adapter
        gen_output = GeneratorOutput(
            reasoning=f"Task: {task}",
            final_answer=result.output,
            bullet_ids=[],
            raw={"success": result.success}
        )

        # Reflect + Curate
        reflection = self.reflector.reflect(
            question=task,
            generator_output=gen_output,
            playbook=self.playbook,
            feedback=f"Result: {result.output}"
        )

        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=f"task: {task}",
            progress=task
        )

        self.playbook.apply_delta(curator_output.delta)
```

---

## Best Practices

### Token Limits
```python
# Reflector needs 400-800 tokens, Curator needs 300-1000
llm = LiteLLMClient(model="gpt-4o-mini", max_tokens=2048)  # Recommended
```

### Error Handling
```python
try:
    reflection = reflector.reflect(...)
except Exception as e:
    logger.error(f"Reflection failed: {e}")
    # Continue without learning
```

### Learning Control
```python
agent.is_learning = False  # Disable for debugging
agent.is_learning = True   # Re-enable
```

### Playbook Persistence
```python
# Save after each session
agent.playbook.save_to_file("my_agent.json")

# Load at startup
agent.playbook = Playbook.load_from_file("my_agent.json")
```

---

## Common Patterns

### API-Based Agents
```python
# After API call
result = api_client.call(enhanced_task)

gen_output = GeneratorOutput(
    reasoning=f"API call: {endpoint}",
    final_answer=result.json(),
    bullet_ids=[],
    raw={"status_code": result.status_code}
)
```

### Multi-Step Agents
```python
# Learn from entire workflow
feedback = f"Completed {len(steps)} steps. Final: {final_result}"
```

### Async Agents
```python
async def _learn_async(self, task, result):
    # Reflector/Curator are sync, wrap if needed
    await asyncio.to_thread(self._learn_sync, task, result)
```

---

## API Reference

### `wrap_playbook_context(playbook: Playbook) -> str`
Returns formatted strategies or empty string if no bullets.

### `GeneratorOutput`
```python
GeneratorOutput(
    reasoning: str,        # What the agent did
    final_answer: str,     # Agent's output
    bullet_ids: List[str], # Empty for external agents
    raw: Dict[str, Any]    # Custom metadata
)
```

### `Reflector.reflect()`
```python
reflect(
    question: str,              # Task description
    generator_output: GeneratorOutput,
    playbook: Playbook,
    ground_truth: Optional[str] = None,
    feedback: Optional[str] = None
) -> ReflectorOutput
```

### `Curator.curate()`
```python
curate(
    reflection: ReflectorOutput,
    playbook: Playbook,
    question_context: str,  # Domain/task info
    progress: str           # Current state
) -> CuratorOutput
```

---

## Troubleshooting

**Q: JSON parsing errors from Curator?**
A: Increase `max_tokens=2048` or higher.

**Q: Not learning anything?**
A: Check `is_learning=True` and verify Curator output has operations.

**Q: Too many bullets?**
A: Curator auto-manages via TAG operations. Review with `playbook.bullets()`.

**Q: High costs?**
A: Disable learning for simple tasks, use cheaper model (gpt-4o-mini).

---

## Examples

See working implementations:
- [`ace/integrations/browser_use.py`](../ace/integrations/browser_use.py) - Browser automation
- [`examples/browser-use/simple_ace_agent.py`](../examples/browser-use/simple_ace_agent.py) - Complete example
- [`examples/custom_integration_example.py`](../examples/custom_integration_example.py) - Minimal custom agent

---

## Next Steps

1. Start with the minimal example above
2. Adapt `_learn()` method to your agent's output format
3. Test with `is_learning=False` first
4. Enable learning and monitor playbook growth
5. Save/load playbooks for persistence

Questions? See [ACE Complete Guide](COMPLETE_GUIDE_TO_ACE.md) or [Discord](https://discord.gg/mqCqH7sTyK).
