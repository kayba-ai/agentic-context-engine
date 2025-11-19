# ACE Integration Guide

Comprehensive guide for integrating ACE learning with your agentic system.

---

## Table of Contents

1. [Integration vs Full Pipeline](#integration-vs-full-pipeline)
2. [The Base Integration Pattern](#the-base-integration-pattern)
3. [Building a Custom Integration](#building-a-custom-integration)
4. [Reference Implementations](#reference-implementations)
5. [Advanced Topics](#advanced-topics)
6. [Troubleshooting](#troubleshooting)

---

## Integration vs Full Pipeline

### Decision Tree: Which Approach Should You Use?

```
Do you have an existing agentic system?
│
├─ YES → Use INTEGRATION PATTERN
│   │
│   ├─ Browser automation? → Use ACEAgent (browser-use)
│   ├─ LangChain chains/agents? → Use ACELangChain
│   └─ Custom agent? → Follow this guide
│
└─ NO → Use FULL ACE PIPELINE
    │
    ├─ Simple tasks (Q&A, classification)? → Use ACELiteLLM
    └─ Complex tasks (tools, workflows)? → Consider LangChain + ACELangChain
```

### What's the Difference?

**INTEGRATION PATTERN** (this guide):
- Your agent executes tasks (browser-use, LangChain, custom API)
- ACE **learns** from results (doesn't execute)
- Components: Playbook + Reflector + Curator (NO Generator)
- Use case: Wrapping existing agents with learning

**FULL ACE PIPELINE** (not this guide):
- ACE Generator executes tasks
- Full ACE components: Playbook + Generator + Reflector + Curator
- Use case: Building new agents from scratch
- See: `ace.integrations.ACELiteLLM` class

---

## The Base Integration Pattern

All ACE integrations follow a three-step pattern:

### Step 1: INJECT (Optional but Recommended)

Add learned strategies from the playbook to your agent's input.

```python
from ace.integrations.base import wrap_playbook_context
from ace import Playbook

playbook = Playbook()  # or load existing: Playbook.load_from_file("expert.json")
task = "Process user request"

# Inject playbook context
if playbook.bullets():
    enhanced_task = f"{task}\n\n{wrap_playbook_context(playbook)}"
else:
    enhanced_task = task  # No learned strategies yet
```

**What does `wrap_playbook_context()` do?**
- Formats learned strategies with success rates
- Adds usage instructions for the agent
- Returns empty string if no bullets (safe to call always)

### Step 2: EXECUTE

Your agent runs normally - ACE doesn't interfere.

```python
# Your agent (any framework/API)
result = your_agent.execute(enhanced_task)

# Examples:
# - Browser-use: await agent.run(task=enhanced_task)
# - LangChain: chain.invoke({"input": enhanced_task})
# - API: requests.post("/execute", json={"task": enhanced_task})
# - Custom: my_agent.run(enhanced_task)
```

### Step 3: LEARN

ACE analyzes the result and updates the playbook.

```python
from ace import LiteLLMClient, Reflector, Curator
from ace.roles import GeneratorOutput

# Setup ACE learning components (do this once)
llm = LiteLLMClient(model="gpt-4o-mini", max_tokens=2048)
reflector = Reflector(llm)
curator = Curator(llm)

# Create adapter for Reflector interface
generator_output = GeneratorOutput(
    reasoning=f"Task: {task}",  # What happened
    final_answer=result.output,  # Agent's output
    bullet_ids=[],  # External agents don't cite bullets
    raw={"success": result.success, "steps": result.steps}  # Metadata
)

# Build feedback string
feedback = f"Task {'succeeded' if result.success else 'failed'}. Output: {result.output}"

# Reflect: Analyze what worked/failed
reflection = reflector.reflect(
    question=task,
    generator_output=generator_output,
    playbook=playbook,
    ground_truth=None,  # Optional: expected output
    feedback=feedback
)

# Curate: Generate playbook updates
curator_output = curator.curate(
    reflection=reflection,
    playbook=playbook,
    question_context=f"task: {task}",
    progress=f"Executing: {task}"
)

# Apply updates
playbook.apply_delta(curator_output.delta)

# Save for next time
playbook.save_to_file("learned_strategies.json")
```

---

## Building a Custom Integration

### Wrapper Class Pattern (Recommended)

Create a wrapper class that bundles your agent with ACE learning:

```python
from ace import Playbook, LiteLLMClient, Reflector, Curator
from ace.integrations.base import wrap_playbook_context
from ace.roles import GeneratorOutput

class ACEWrapper:
    """Wraps your custom agent with ACE learning."""

    def __init__(
        self,
        agent,
        ace_model: str = "gpt-4o-mini",
        playbook_path: str = None,
        is_learning: bool = True
    ):
        """
        Args:
            agent: Your agent instance
            ace_model: Model for ACE learning (Reflector/Curator)
            playbook_path: Path to existing playbook (optional)
            is_learning: Enable/disable learning
        """
        self.agent = agent
        self.is_learning = is_learning

        # Load or create playbook
        if playbook_path:
            self.playbook = Playbook.load_from_file(playbook_path)
        else:
            self.playbook = Playbook()

        # Setup ACE learning components
        self.llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(self.llm)
        self.curator = Curator(self.llm)

    def run(self, task: str):
        """Execute task with ACE learning."""
        # STEP 1: Inject playbook context
        enhanced_task = self._inject_context(task)

        # STEP 2: Execute
        result = self.agent.execute(enhanced_task)

        # STEP 3: Learn (if enabled)
        if self.is_learning:
            self._learn(task, result)

        return result

    def _inject_context(self, task: str) -> str:
        """Add playbook strategies to task."""
        if self.playbook.bullets():
            return f"{task}\n\n{wrap_playbook_context(self.playbook)}"
        return task

    def _learn(self, task: str, result):
        """Run ACE learning pipeline."""
        # Adapt result to ACE interface
        generator_output = GeneratorOutput(
            reasoning=f"Task: {task}",
            final_answer=result.output,
            bullet_ids=[],
            raw={"success": result.success}
        )

        # Build feedback
        feedback = f"Task {'succeeded' if result.success else 'failed'}"

        # Reflect
        reflection = self.reflector.reflect(
            question=task,
            generator_output=generator_output,
            playbook=self.playbook,
            feedback=feedback
        )

        # Curate
        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=f"task: {task}",
            progress=task
        )

        # Update playbook
        self.playbook.apply_delta(curator_output.delta)

    def save_playbook(self, path: str):
        """Save learned strategies."""
        self.playbook.save_to_file(path)

    def load_playbook(self, path: str):
        """Load existing strategies."""
        self.playbook = Playbook.load_from_file(path)

    def enable_learning(self):
        """Enable learning."""
        self.is_learning = True

    def disable_learning(self):
        """Disable learning (execution only)."""
        self.is_learning = False
```

### Usage Example

```python
# Your custom agent
class MyAgent:
    def execute(self, task: str):
        # Your agent logic
        return {"output": "result", "success": True}

# Wrap with ACE
my_agent = MyAgent()
ace_agent = ACEWrapper(my_agent, is_learning=True)

# Use it
result = ace_agent.run("Process data")
print(f"Result: {result.output}")
print(f"Learned {len(ace_agent.playbook.bullets())} strategies")

# Save learned knowledge
ace_agent.save_playbook("my_agent_learned.json")

# Next session: Load previous knowledge
ace_agent = ACEWrapper(MyAgent(), playbook_path="my_agent_learned.json")
```

---

## Reference Implementations

### Browser-Use Integration

See [`ace/integrations/browser_use.py`](../ace/integrations/browser_use.py) for a complete reference implementation.

**Key Design Decisions:**

1. **Context Injection** (line 182-189):
```python
if self.is_learning and self.playbook.bullets():
    playbook_context = wrap_playbook_context(self.playbook)
    enhanced_task = f"{current_task}\n\n{playbook_context}"
```

2. **Rich Feedback Extraction** (line 234-403):
- Extracts chronological execution trace
- Includes agent thoughts, actions, results
- Provides detailed context for Reflector

3. **Citation Extraction** (line 405-434):
- Parses agent's reasoning for bullet citations
- Filters invalid IDs (graceful degradation)

4. **Learning Pipeline** (line 436-510):
- Creates GeneratorOutput adapter
- Passes full trace to Reflector in `reasoning` field
- Updates playbook via Curator

**Why Browser-Use is a Good Reference:**
- Shows rich feedback extraction
- Handles async execution
- Robust error handling
- Learning toggle
- Playbook persistence

---

## Advanced Topics

### Rich Feedback Extraction

The quality of ACE learning depends on the feedback you provide. The more detailed, the better.

**Basic Feedback (Minimal):**
```python
feedback = f"Task {'succeeded' if success else 'failed'}"
```

**Good Feedback (Contextual):**
```python
feedback = f"""
Task {'succeeded' if success else 'failed'} in {steps} steps.
Duration: {duration}s
Final output: {output[:200]}...
"""
```

**Rich Feedback (Detailed Trace):**
```python
# For agents with step-by-step execution
feedback_parts = []
feedback_parts.append(f"Task {status} in {steps} steps")

# Add execution trace
for i, step in enumerate(execution_steps, 1):
    feedback_parts.append(f"\nStep {i}:")
    feedback_parts.append(f"  Thought: {step.thought}")
    feedback_parts.append(f"  Action: {step.action}")
    feedback_parts.append(f"  Result: {step.result}")

feedback = "\n".join(feedback_parts)
```

**Benefits of Rich Feedback:**
- Learns action sequencing patterns
- Understands timing requirements
- Recognizes error patterns
- Captures domain-specific knowledge

### Citation-Based Strategy Tracking

ACE uses citations to track which strategies were used:

**How It Works:**
1. Strategies are formatted with IDs: `[section-00001]`
2. Agent cites them in reasoning: `"Following [navigation-00042], I will..."`
3. ACE extracts citations automatically

**Extracting Citations:**
```python
from ace.roles import extract_cited_bullet_ids

# Agent's reasoning with citations
reasoning = """
Step 1: Following [navigation-00042], navigate to main page.
Step 2: Using [extraction-00003], extract title element.
"""

# Extract citations
cited_ids = extract_cited_bullet_ids(reasoning)
# Returns: ['navigation-00042', 'extraction-00003']

# Pass to GeneratorOutput
generator_output = GeneratorOutput(
    reasoning=reasoning,
    final_answer=result,
    bullet_ids=cited_ids,
    raw={}
)
```

**For External Agents:**
```python
# Extract from agent's thought process
if hasattr(history, 'model_thoughts'):
    thoughts = history.model_thoughts()
    thoughts_text = "\n".join(t.thinking for t in thoughts)
    cited_ids = extract_cited_bullet_ids(thoughts_text)
```

### Handling Async Agents

If your agent is async, wrap the learning in a sync function:

```python
async def run(self, task: str):
    # Inject context
    enhanced_task = self._inject_context(task)

    # Execute (async)
    result = await self.agent.execute(enhanced_task)

    # Learn (sync Reflector/Curator)
    if self.is_learning:
        await asyncio.to_thread(self._learn, task, result)

    return result
```

### Error Handling

Always wrap learning in try/except to prevent crashes:

```python
def _learn(self, task: str, result):
    try:
        # Reflection
        reflection = self.reflector.reflect(...)

        # Curation
        curator_output = self.curator.curate(...)

        # Update
        self.playbook.apply_delta(curator_output.delta)

    except Exception as e:
        logger.error(f"ACE learning failed: {e}")
        # Continue without learning - don't crash!
```

### Token Limits

ACE learning components need sufficient tokens:

```python
# Reflector: 400-800 tokens typical
# Curator: 300-1000 tokens typical
llm = LiteLLMClient(model="gpt-4o-mini", max_tokens=2048)  # Recommended

# For complex tasks with long traces:
llm = LiteLLMClient(model="gpt-4o-mini", max_tokens=4096)
```

---

## Troubleshooting

### Problem: JSON Parsing Errors from Curator

**Cause:** Insufficient `max_tokens` for structured output

**Solution:**
```python
llm = LiteLLMClient(model="gpt-4o-mini", max_tokens=2048)  # or higher
```

### Problem: Not Learning Anything

**Checks:**
1. Is `is_learning=True`?
2. Is Curator output non-empty? `print(curator_output.delta)`
3. Is playbook being saved? `playbook.save_to_file(...)`

### Problem: Too Many Bullets

**Solution:** Curator automatically manages bullets via TAG operations. Review with:
```python
bullets = playbook.bullets()
print(f"Total: {len(bullets)}")
for b in bullets[:10]:
    print(f"[{b.id}] +{b.helpful}/-{b.harmful}: {b.content}")
```

### Problem: High API Costs

**Solutions:**
- Use cheaper model: `ace_model="gpt-4o-mini"`
- Disable learning for simple tasks: `is_learning=False`
- Batch learning: Learn only every N tasks

### Problem: Agent Ignores Playbook Strategies

**Checks:**
1. Are you actually injecting context? `print(enhanced_task)`
2. Does playbook have bullets? `print(len(playbook.bullets()))`
3. Is context clear enough for your agent?

---

## Common Integration Patterns

### REST API-Based Agents

```python
class APIAgent:
    def execute(self, task: str):
        response = requests.post(
            "https://api.example.com/execute",
            json={"task": task}
        )
        return {
            "output": response.json()["result"],
            "success": response.status_code == 200
        }

# Wrap with ACE
ace_agent = ACEWrapper(APIAgent())
```

### Multi-Step Workflow Agents

```python
def _learn(self, task, result):
    # Build rich feedback from all steps
    feedback = f"Workflow completed {len(result.steps)} steps:\n"
    for i, step in enumerate(result.steps, 1):
        feedback += f"Step {i}: {step.action} → {step.outcome}\n"

    generator_output = GeneratorOutput(
        reasoning=feedback,  # Full workflow trace
        final_answer=result.final_output,
        bullet_ids=[],
        raw={"steps": len(result.steps)}
    )
    # ... rest of learning pipeline
```

### Tool-Using Agents

```python
def _inject_context(self, task: str) -> str:
    """Inject into system message instead of task."""
    if self.playbook.bullets():
        context = wrap_playbook_context(self.playbook)
        # Add to system message or tool descriptions
        self.agent.system_message = f"{self.agent.system_message}\n\n{context}"
    return task
```

---

## Next Steps

1. **Start Simple:** Use the wrapper class template above
2. **Adapt `_learn()`:** Customize for your agent's output format
3. **Test Without Learning:** Set `is_learning=False` first
4. **Enable Learning:** Turn on and monitor playbook growth
5. **Iterate:** Improve feedback extraction for better learning

---

## See Also

- **Out-of-box integrations:** ACELiteLLM, ACEAgent (browser-use), ACELangChain
- **Integration patterns:** [INTEGRATION_PATTERNS.md](INTEGRATION_PATTERNS.md)
- **Full ACE guide:** [COMPLETE_GUIDE_TO_ACE.md](COMPLETE_GUIDE_TO_ACE.md)
- **API reference:** [API_REFERENCE.md](API_REFERENCE.md)

Questions? Join our [Discord](https://discord.gg/mqCqH7sTyK)
