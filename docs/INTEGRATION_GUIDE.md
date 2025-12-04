# ACE Integration Guide

Comprehensive guide for integrating ACE learning with your agentic system.

---

## Table of Contents

1. [Integration vs Full Pipeline](#integration-vs-full-pipeline)
2. [The Base Integration Pattern](#the-base-integration-pattern)
3. [Building a Custom Integration](#building-a-custom-integration)
4. [Reference Implementations](#reference-implementations)
5. [Integration Patterns](#integration-patterns)
6. [Advanced Topics](#advanced-topics)
7. [Troubleshooting](#troubleshooting)

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
- Components: Skillbook + Reflector + SkillManager (NO Agent)
- Use case: Wrapping existing agents with learning

**FULL ACE PIPELINE** (not this guide):
- ACE Agent executes tasks
- Full ACE components: Skillbook + Agent + Reflector + SkillManager
- Use case: Building new agents from scratch
- See: `ace.integrations.ACELiteLLM` class

---

## The Base Integration Pattern

All ACE integrations follow a three-step pattern:

### Step 1: INJECT (Optional but Recommended)

Add learned strategies from the skillbook to your agent's input.

```python
from ace.integrations.base import wrap_skillbook_context
from ace import Skillbook

skillbook = Skillbook()  # or load existing: Skillbook.load_from_file("expert.json")
task = "Process user request"

# Inject skillbook context
if skillbook.skills():
    enhanced_task = f"{task}\n\n{wrap_skillbook_context(skillbook)}"
else:
    enhanced_task = task  # No learned strategies yet
```

**What does `wrap_skillbook_context()` do?**
- Formats learned strategies with success rates
- Adds usage instructions for the agent
- Returns empty string if no skills (safe to call always)

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

ACE analyzes the result and updates the skillbook.

```python
from ace import LiteLLMClient, Reflector, SkillManager
from ace.roles import AgentOutput

# Setup ACE learning components (do this once)
llm = LiteLLMClient(model="gpt-4o-mini", max_tokens=2048)
reflector = Reflector(llm)
skill_manager = SkillManager(llm)

# Create adapter for Reflector interface
agent_output = AgentOutput(
    reasoning=f"Task: {task}",  # What happened
    final_answer=result.output,  # Agent's output
    skill_ids=[],  # External agents don't cite skills
    raw={"success": result.success, "steps": result.steps}  # Metadata
)

# Build feedback string
feedback = f"Task {'succeeded' if result.success else 'failed'}. Output: {result.output}"

# Reflect: Analyze what worked/failed
reflection = reflector.reflect(
    question=task,
    agent_output=agent_output,
    skillbook=skillbook,
    ground_truth=None,  # Optional: expected output
    feedback=feedback
)

# Curate: Generate skillbook updates
skill_manager_output = skill_manager.curate(
    reflection=reflection,
    skillbook=skillbook,
    question_context=f"task: {task}",
    progress=f"Executing: {task}"
)

# Apply updates
skillbook.apply_update(skill_manager_output.update)

# Save for next time
skillbook.save_to_file("learned_strategies.json")
```

---

## Building a Custom Integration

### Wrapper Class Pattern (Recommended)

Create a wrapper class that bundles your agent with ACE learning:

```python
from ace import Skillbook, LiteLLMClient, Reflector, SkillManager
from ace.integrations.base import wrap_skillbook_context
from ace.roles import AgentOutput

class ACEWrapper:
    """Wraps your custom agent with ACE learning."""

    def __init__(
        self,
        agent,
        ace_model: str = "gpt-4o-mini",
        skillbook_path: str = None,
        is_learning: bool = True
    ):
        """
        Args:
            agent: Your agent instance
            ace_model: Model for ACE learning (Reflector/SkillManager)
            skillbook_path: Path to existing skillbook (optional)
            is_learning: Enable/disable learning
        """
        self.agent = agent
        self.is_learning = is_learning

        # Load or create skillbook
        if skillbook_path:
            self.skillbook = Skillbook.load_from_file(skillbook_path)
        else:
            self.skillbook = Skillbook()

        # Setup ACE learning components
        self.llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(self.llm)
        self.skill_manager = SkillManager(self.llm)

    def run(self, task: str):
        """Execute task with ACE learning."""
        # STEP 1: Inject skillbook context
        enhanced_task = self._inject_context(task)

        # STEP 2: Execute
        result = self.agent.execute(enhanced_task)

        # STEP 3: Learn (if enabled)
        if self.is_learning:
            self._learn(task, result)

        return result

    def _inject_context(self, task: str) -> str:
        """Add skillbook strategies to task."""
        if self.skillbook.skills():
            return f"{task}\n\n{wrap_skillbook_context(self.skillbook)}"
        return task

    def _learn(self, task: str, result):
        """Run ACE learning pipeline."""
        # Adapt result to ACE interface
        agent_output = AgentOutput(
            reasoning=f"Task: {task}",
            final_answer=result.output,
            skill_ids=[],
            raw={"success": result.success}
        )

        # Build feedback
        feedback = f"Task {'succeeded' if result.success else 'failed'}"

        # Reflect
        reflection = self.reflector.reflect(
            question=task,
            agent_output=agent_output,
            skillbook=self.skillbook,
            feedback=feedback
        )

        # Curate
        skill_manager_output = self.skill_manager.curate(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context=f"task: {task}",
            progress=task
        )

        # Update skillbook
        self.skillbook.apply_update(skill_manager_output.update)

    def save_skillbook(self, path: str):
        """Save learned strategies."""
        self.skillbook.save_to_file(path)

    def load_skillbook(self, path: str):
        """Load existing strategies."""
        self.skillbook = Skillbook.load_from_file(path)

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
print(f"Learned {len(ace_agent.skillbook.skills())} strategies")

# Save learned knowledge
ace_agent.save_skillbook("my_agent_learned.json")

# Next session: Load previous knowledge
ace_agent = ACEWrapper(MyAgent(), skillbook_path="my_agent_learned.json")
```

---

## Reference Implementations

### Browser-Use Integration

See [`ace/integrations/browser_use.py`](../ace/integrations/browser_use.py) for a complete reference implementation.

**Key Design Decisions:**

1. **Context Injection** (line 182-189):
```python
if self.is_learning and self.skillbook.skills():
    skillbook_context = wrap_skillbook_context(self.skillbook)
    enhanced_task = f"{current_task}\n\n{skillbook_context}"
```

2. **Rich Feedback Extraction** (line 234-403):
- Extracts chronological execution trace
- Includes agent thoughts, actions, results
- Provides detailed context for Reflector

3. **Citation Extraction** (line 405-434):
- Parses agent's reasoning for skill citations
- Filters invalid IDs (graceful degradation)

4. **Learning Pipeline** (line 436-510):
- Creates AgentOutput adapter
- Passes full trace to Reflector in `reasoning` field
- Updates skillbook via SkillManager

**Why Browser-Use is a Good Reference:**
- Shows rich feedback extraction
- Handles async execution
- Robust error handling
- Learning toggle
- Skillbook persistence

### Runnable Examples

See these working examples in the repository:

- **Browser automation**: [`examples/browser-use/simple_ace_agent.py`](../examples/browser-use/simple_ace_agent.py)
- **Custom integration**: [`examples/custom_integration_example.py`](../examples/custom_integration_example.py)
- **LangChain chains/agents**: [`examples/langchain/simple_chain_example.py`](../examples/langchain/simple_chain_example.py)

Full list: [`examples/README.md`](../examples/README.md)

---

## Integration Patterns

Common patterns for integrating ACE with different types of agents. Each pattern includes complete code examples, when to use it, and key considerations.

---

### REST API-Based Agents

#### When to Use
- Your agent is a REST API service
- Remote execution (cloud-based agents)
- Stateless request/response pattern

#### Pattern

```python
from ace import Skillbook, LiteLLMClient, Reflector, SkillManager
from ace.integrations.base import wrap_skillbook_context
from ace.roles import AgentOutput
import requests

class ACEAPIAgent:
    """Wraps REST API agent with ACE learning."""

    def __init__(self, api_url: str, api_key: str = None, ace_model: str = "gpt-4o-mini"):
        self.api_url = api_url
        self.api_key = api_key
        self.skillbook = Skillbook()

        # ACE components
        llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(llm)
        self.skill_manager = SkillManager(llm)

    def execute(self, task: str):
        """Execute task via API with ACE learning."""
        # Inject context
        if self.skillbook.skills():
            task = f"{task}\n\n{wrap_skillbook_context(self.skillbook)}"

        # API call
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = requests.post(
            f"{self.api_url}/execute",
            json={"task": task},
            headers=headers,
            timeout=60
        )

        # Extract result
        success = response.status_code == 200
        output = response.json().get("result", "") if success else response.text

        # Learn
        self._learn(task, output, success)

        return {"output": output, "success": success}

    def _learn(self, task: str, output: str, success: bool):
        # Create adapter
        agent_output = AgentOutput(
            reasoning=f"API call for task: {task}",
            final_answer=output,
            skill_ids=[],
            raw={"success": success}
        )

        # Feedback
        feedback = f"API call {'succeeded' if success else 'failed'}. Output: {output[:200]}"

        # Reflect + Curate
        reflection = self.reflector.reflect(
            question=task,
            agent_output=agent_output,
            skillbook=self.skillbook,
            feedback=feedback
        )

        skill_manager_output = self.skill_manager.curate(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context=f"API task: {task}",
            progress=task
        )

        self.skillbook.apply_update(skill_manager_output.update)

# Usage
agent = ACEAPIAgent(api_url="https://api.example.com", api_key="...")
result = agent.execute("Process user data")
agent.skillbook.save_to_file("api_agent_learned.json")
```

#### Key Considerations
- Handle timeouts and retries
- Parse API error messages for better feedback
- Consider rate limiting (don't learn on every call if high volume)

---

### Multi-Step Workflow Agents

#### When to Use
- Agent executes multiple sequential steps
- Each step has its own outcome
- Want to learn from entire workflow

#### Pattern

```python
from dataclasses import dataclass
from typing import List

@dataclass
class WorkflowStep:
    action: str
    outcome: str
    success: bool
    duration: float

@dataclass
class WorkflowResult:
    steps: List[WorkflowStep]
    final_output: str
    overall_success: bool

class ACEWorkflowAgent:
    """Wraps multi-step workflow agent with rich trace learning."""

    def __init__(self, workflow_agent, ace_model: str = "gpt-4o-mini"):
        self.agent = workflow_agent
        self.skillbook = Skillbook()

        llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(llm)
        self.skill_manager = SkillManager(llm)

    def run(self, task: str) -> WorkflowResult:
        """Execute workflow with ACE learning."""
        # Inject context
        if self.skillbook.skills():
            task = f"{task}\n\n{wrap_skillbook_context(self.skillbook)}"

        # Execute workflow (returns WorkflowResult)
        result = self.agent.execute_workflow(task)

        # Learn from entire workflow
        self._learn(task, result)

        return result

    def _learn(self, task: str, result: WorkflowResult):
        """Learn from complete workflow trace."""
        # Build rich feedback with all steps
        feedback_parts = [
            f"Workflow {'succeeded' if result.overall_success else 'failed'} "
            f"in {len(result.steps)} steps\n"
        ]

        for i, step in enumerate(result.steps, 1):
            status = "✓" if step.success else "✗"
            feedback_parts.append(
                f"Step {i} [{status}]: {step.action}\n"
                f"  → Outcome: {step.outcome}\n"
                f"  → Duration: {step.duration:.2f}s"
            )

        feedback = "\n".join(feedback_parts)

        # Create adapter with full trace
        agent_output = AgentOutput(
            reasoning=feedback,  # Full workflow trace
            final_answer=result.final_output,
            skill_ids=[],
            raw={
                "total_steps": len(result.steps),
                "successful_steps": sum(1 for s in result.steps if s.success),
                "total_duration": sum(s.duration for s in result.steps)
            }
        )

        # Reflect + Curate
        reflection = self.reflector.reflect(
            question=task,
            agent_output=agent_output,
            skillbook=self.skillbook,
            feedback=feedback
        )

        skill_manager_output = self.skill_manager.curate(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context=f"Multi-step workflow: {task}",
            progress=f"Completed {len(result.steps)} steps"
        )

        self.skillbook.apply_update(skill_manager_output.update)

# Usage
workflow_agent = MyWorkflowAgent()
ace_agent = ACEWorkflowAgent(workflow_agent)
result = ace_agent.run("Complete data pipeline")
```

#### Key Considerations
- Include step-by-step trace in feedback for better learning
- Track timing information to learn performance patterns
- Distinguish partial failures (some steps succeed) from total failures

---

### Tool-Using Agents

#### When to Use
- Agent has access to external tools/functions
- Tool selection and usage is part of learning
- Want to inject context into system message or tool descriptions

#### Pattern

```python
class ACEToolAgent:
    """Wraps tool-using agent with ACE learning."""

    def __init__(self, agent, ace_model: str = "gpt-4o-mini"):
        self.agent = agent
        self.skillbook = Skillbook()
        self.original_system_message = agent.system_message

        llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(llm)
        self.skill_manager = SkillManager(llm)

    def run(self, task: str):
        """Execute with tool access and ACE learning."""
        # Inject skillbook into system message (not task)
        if self.skillbook.skills():
            context = wrap_skillbook_context(self.skillbook)
            self.agent.system_message = f"{self.original_system_message}\n\n{context}"

        # Execute (agent selects and uses tools)
        result = self.agent.execute(task)

        # Restore original system message
        self.agent.system_message = self.original_system_message

        # Learn
        self._learn(task, result)

        return result

    def _learn(self, task: str, result):
        """Learn from tool usage patterns."""
        # Extract tool usage information
        tools_used = result.get("tools_used", [])
        tool_results = result.get("tool_results", [])

        # Build rich feedback
        feedback_parts = [
            f"Task {'succeeded' if result['success'] else 'failed'}",
            f"Tools used: {', '.join(t['name'] for t in tools_used)}"
        ]

        for tool, tool_result in zip(tools_used, tool_results):
            feedback_parts.append(
                f"  {tool['name']}({tool['args']}) → {tool_result['outcome']}"
            )

        feedback = "\n".join(feedback_parts)

        # Adapter
        agent_output = AgentOutput(
            reasoning=feedback,
            final_answer=result["output"],
            skill_ids=[],
            raw={"tools_used": [t["name"] for t in tools_used]}
        )

        # Reflect + Curate
        reflection = self.reflector.reflect(
            question=task,
            agent_output=agent_output,
            skillbook=self.skillbook,
            feedback=feedback
        )

        skill_manager_output = self.skill_manager.curate(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context=f"Tool-using task: {task}",
            progress=f"Used {len(tools_used)} tools"
        )

        self.skillbook.apply_update(skill_manager_output.update)

# Usage
tool_agent = MyToolUsingAgent(tools=[...])
ace_agent = ACEToolAgent(tool_agent)
result = ace_agent.run("Analyze data and send report")
```

#### Key Considerations
- Inject context into system message (not task) for better tool selection
- Track which tools were used for learning tool selection patterns
- Include tool outcomes in feedback

---

### Async Agents

#### When to Use
- Agent operations are async (browser automation, async APIs)
- Need non-blocking execution
- Want to maintain async interface

#### Pattern

```python
import asyncio

class ACEAsyncAgent:
    """Wraps async agent with ACE learning."""

    def __init__(self, async_agent, ace_model: str = "gpt-4o-mini"):
        self.agent = async_agent
        self.skillbook = Skillbook()

        llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(llm)
        self.skill_manager = SkillManager(llm)

    async def run(self, task: str):
        """Async execution with ACE learning."""
        # Inject context
        if self.skillbook.skills():
            task = f"{task}\n\n{wrap_skillbook_context(self.skillbook)}"

        # Execute (async)
        result = await self.agent.execute(task)

        # Learn (sync operations in thread)
        await asyncio.to_thread(self._learn, task, result)

        return result

    def _learn(self, task: str, result):
        """Sync learning pipeline (runs in thread)."""
        agent_output = AgentOutput(
            reasoning=f"Async task: {task}",
            final_answer=result["output"],
            skill_ids=[],
            raw={"success": result["success"]}
        )

        feedback = f"Async task {'succeeded' if result['success'] else 'failed'}"

        reflection = self.reflector.reflect(
            question=task,
            agent_output=agent_output,
            skillbook=self.skillbook,
            feedback=feedback
        )

        skill_manager_output = self.skill_manager.curate(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context=task,
            progress=task
        )

        self.skillbook.apply_update(skill_manager_output.update)

# Usage
async def main():
    async_agent = MyAsyncAgent()
    ace_agent = ACEAsyncAgent(async_agent)

    result = await ace_agent.run("Fetch and process data")
    print(f"Result: {result}")

asyncio.run(main())
```

#### Key Considerations
- Use `asyncio.to_thread()` to run sync Reflector/SkillManager in background
- Don't block async event loop with sync ACE operations
- Consider batching learning for high-throughput async systems

---

### Chat-Based Agents

#### When to Use
- Agent maintains conversation history
- Multi-turn interactions
- Want to learn from entire conversation

#### Pattern

```python
class ACEChatAgent:
    """Wraps chat agent with per-conversation learning."""

    def __init__(self, chat_agent, ace_model: str = "gpt-4o-mini"):
        self.agent = chat_agent
        self.skillbook = Skillbook()
        self.conversation_history = []

        llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(llm)
        self.skill_manager = SkillManager(llm)

    def chat(self, message: str) -> str:
        """Single chat turn with context injection."""
        # Inject skillbook on first message
        if len(self.conversation_history) == 0 and self.skillbook.skills():
            system_context = wrap_skillbook_context(self.skillbook)
            self.agent.add_system_message(system_context)

        # Chat
        response = self.agent.chat(message)

        # Track conversation
        self.conversation_history.append({"user": message, "assistant": response})

        return response

    def end_conversation(self, success: bool = True, feedback: str = ""):
        """Learn from entire conversation at the end."""
        if not self.conversation_history:
            return

        # Build conversation summary
        conversation = "\n".join(
            f"User: {turn['user']}\nAssistant: {turn['assistant']}"
            for turn in self.conversation_history
        )

        # Learn from full conversation
        agent_output = AgentOutput(
            reasoning=conversation,
            final_answer=self.conversation_history[-1]["assistant"],
            skill_ids=[],
            raw={"turns": len(self.conversation_history)}
        )

        feedback_text = (
            f"Conversation {'succeeded' if success else 'failed'} "
            f"over {len(self.conversation_history)} turns. {feedback}"
        )

        reflection = self.reflector.reflect(
            question=self.conversation_history[0]["user"],
            agent_output=agent_output,
            skillbook=self.skillbook,
            feedback=feedback_text
        )

        skill_manager_output = self.skill_manager.curate(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context=f"Multi-turn conversation ({len(self.conversation_history)} turns)",
            progress="Conversation completed"
        )

        self.skillbook.apply_update(skill_manager_output.update)

        # Reset for next conversation
        self.conversation_history = []

# Usage
chat_agent = MyChatAgent()
ace_agent = ACEChatAgent(chat_agent)

# Multi-turn conversation
ace_agent.chat("Hello, I need help with X")
ace_agent.chat("Can you clarify Y?")
ace_agent.chat("Thanks, that works!")

# Learn from entire conversation
ace_agent.end_conversation(success=True, feedback="User satisfied")
ace_agent.skillbook.save_to_file("chat_agent_learned.json")
```

#### Key Considerations
- Learn from complete conversation (not individual turns)
- Inject skillbook context at conversation start
- Allow manual feedback at conversation end

---

### Batch Processing Agents

#### When to Use
- Processing large batches of similar tasks
- Want to amortize learning costs
- Need high throughput

#### Pattern

```python
class ACEBatchAgent:
    """Wraps agent with batched learning."""

    def __init__(self, agent, ace_model: str = "gpt-4o-mini", learn_every: int = 10):
        self.agent = agent
        self.skillbook = Skillbook()
        self.learn_every = learn_every
        self.pending_results = []

        llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(llm)
        self.skill_manager = SkillManager(llm)

    def process(self, task: str):
        """Process single task (learn in batches)."""
        # Inject context
        if self.skillbook.skills():
            task = f"{task}\n\n{wrap_skillbook_context(self.skillbook)}"

        # Execute
        result = self.agent.execute(task)

        # Add to pending
        self.pending_results.append((task, result))

        # Learn when batch is full
        if len(self.pending_results) >= self.learn_every:
            self._learn_from_batch()

        return result

    def _learn_from_batch(self):
        """Learn from accumulated results."""
        if not self.pending_results:
            return

        # Aggregate feedback
        successes = sum(1 for _, r in self.pending_results if r["success"])
        failures = len(self.pending_results) - successes

        # Learn from batch summary
        feedback = (
            f"Batch of {len(self.pending_results)} tasks: "
            f"{successes} succeeded, {failures} failed"
        )

        # Use first task as representative
        task, result = self.pending_results[0]

        agent_output = AgentOutput(
            reasoning=f"Batch processing: {feedback}",
            final_answer=result["output"],
            skill_ids=[],
            raw={"batch_size": len(self.pending_results), "success_rate": successes / len(self.pending_results)}
        )

        reflection = self.reflector.reflect(
            question=task,
            agent_output=agent_output,
            skillbook=self.skillbook,
            feedback=feedback
        )

        skill_manager_output = self.skill_manager.curate(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context=f"Batch processing ({len(self.pending_results)} items)",
            progress="Batch completed"
        )

        self.skillbook.apply_update(skill_manager_output.update)

        # Clear pending
        self.pending_results = []

    def flush(self):
        """Force learning from remaining pending results."""
        self._learn_from_batch()

# Usage
agent = MyBatchAgent()
ace_agent = ACEBatchAgent(agent, learn_every=10)

# Process many tasks
for task in tasks:
    ace_agent.process(task)

# Learn from remainder
ace_agent.flush()
ace_agent.skillbook.save_to_file("batch_learned.json")
```

#### Key Considerations
- Balance learning frequency vs cost (learn_every parameter)
- Call `flush()` at end to learn from remaining items
- Consider success rate in batch feedback

---

### Streaming Agents

#### When to Use
- Agent streams responses token-by-token
- Want to maintain streaming interface
- Learn after complete stream

#### Pattern

```python
class ACEStreamingAgent:
    """Wraps streaming agent with post-stream learning."""

    def __init__(self, streaming_agent, ace_model: str = "gpt-4o-mini"):
        self.agent = streaming_agent
        self.skillbook = Skillbook()

        llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(llm)
        self.skill_manager = SkillManager(llm)

    def stream(self, task: str):
        """Stream response with learning after completion."""
        # Inject context
        if self.skillbook.skills():
            task = f"{task}\n\n{wrap_skillbook_context(self.skillbook)}"

        # Collect full response while streaming
        full_response = []

        for chunk in self.agent.stream(task):
            full_response.append(chunk)
            yield chunk  # Stream to caller

        # Learn after stream completes
        complete_response = "".join(full_response)
        self._learn(task, complete_response)

    def _learn(self, task: str, response: str):
        """Learn from complete streamed response."""
        agent_output = AgentOutput(
            reasoning=f"Streamed response for: {task}",
            final_answer=response,
            skill_ids=[],
            raw={"response_length": len(response)}
        )

        feedback = f"Streamed {len(response)} characters"

        reflection = self.reflector.reflect(
            question=task,
            agent_output=agent_output,
            skillbook=self.skillbook,
            feedback=feedback
        )

        skill_manager_output = self.skill_manager.curate(
            reflection=reflection,
            skillbook=self.skillbook,
            question_context=task,
            progress="Streaming completed"
        )

        self.skillbook.apply_update(skill_manager_output.update)

# Usage
streaming_agent = MyStreamingAgent()
ace_agent = ACEStreamingAgent(streaming_agent)

for chunk in ace_agent.stream("Generate report"):
    print(chunk, end="", flush=True)
```

#### Key Considerations
- Collect full response before learning
- Don't block streaming (learn after completion)
- Maintain streaming interface for caller

---

### Error-Prone Agents

#### When to Use
- Agent frequently fails or throws exceptions
- Want to learn from failures
- Need robust error handling

#### Pattern

```python
class ACERobustAgent:
    """Wraps agent with error handling and failure learning."""

    def __init__(self, agent, ace_model: str = "gpt-4o-mini", max_retries: int = 3):
        self.agent = agent
        self.skillbook = Skillbook()
        self.max_retries = max_retries

        llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(llm)
        self.skill_manager = SkillManager(llm)

    def run(self, task: str):
        """Execute with retries and error learning."""
        # Inject context
        if self.skillbook.skills():
            task = f"{task}\n\n{wrap_skillbook_context(self.skillbook)}"

        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = self.agent.execute(task)
                # Success - learn from it
                self._learn(task, result, success=True)
                return result

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    # Retry
                    continue
                else:
                    # Final failure - learn from it
                    self._learn(task, None, success=False, error=last_error)
                    raise

    def _learn(self, task: str, result, success: bool, error: str = None):
        """Learn from both successes and failures."""
        try:
            # Build feedback
            if success:
                feedback = f"Task succeeded. Output: {result['output']}"
                final_answer = result["output"]
            else:
                feedback = f"Task failed after {self.max_retries} attempts. Error: {error}"
                final_answer = ""

            # Adapter
            agent_output = AgentOutput(
                reasoning=f"Task: {task}. {feedback}",
                final_answer=final_answer,
                skill_ids=[],
                raw={"success": success, "error": error}
            )

            # Reflect + Curate
            reflection = self.reflector.reflect(
                question=task,
                agent_output=agent_output,
                skillbook=self.skillbook,
                feedback=feedback
            )

            skill_manager_output = self.skill_manager.curate(
                reflection=reflection,
                skillbook=self.skillbook,
                question_context=f"Task ({'success' if success else 'failure'}): {task}",
                progress="Execution completed"
            )

            self.skillbook.apply_update(skill_manager_output.update)

        except Exception as learning_error:
            # Never crash due to learning failures
            print(f"Learning failed: {learning_error}")

# Usage
error_prone_agent = MyUnreliableAgent()
ace_agent = ACERobustAgent(error_prone_agent, max_retries=3)

try:
    result = ace_agent.run("Risky task")
except Exception as e:
    print(f"Task failed: {e}")
    # But skillbook learned from the failure!
```

#### Key Considerations
- Learn from both successes AND failures
- Wrap learning in try/except (never crash from learning)
- Include error details in feedback for failure pattern learning

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
from ace.roles import extract_cited_skill_ids

# Agent's reasoning with citations
reasoning = """
Step 1: Following [navigation-00042], navigate to main page.
Step 2: Using [extraction-00003], extract title element.
"""

# Extract citations
cited_ids = extract_cited_skill_ids(reasoning)
# Returns: ['navigation-00042', 'extraction-00003']

# Pass to AgentOutput
agent_output = AgentOutput(
    reasoning=reasoning,
    final_answer=result,
    skill_ids=cited_ids,
    raw={}
)
```

**For External Agents:**
```python
# Extract from agent's thought process
if hasattr(history, 'model_thoughts'):
    thoughts = history.model_thoughts()
    thoughts_text = "\n".join(t.thinking for t in thoughts)
    cited_ids = extract_cited_skill_ids(thoughts_text)
```

### Handling Async Agents

If your agent is async, wrap the learning in a sync function:

```python
async def run(self, task: str):
    # Inject context
    enhanced_task = self._inject_context(task)

    # Execute (async)
    result = await self.agent.execute(enhanced_task)

    # Learn (sync Reflector/SkillManager)
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
        self.skillbook.apply_update(skill_manager_output.update)

    except Exception as e:
        logger.error(f"ACE learning failed: {e}")
        # Continue without learning - don't crash!
```

### Token Limits

ACE learning components need sufficient tokens:

```python
# Reflector: 400-800 tokens typical
# SkillManager: 300-1000 tokens typical
llm = LiteLLMClient(model="gpt-4o-mini", max_tokens=2048)  # Recommended

# For complex tasks with long traces:
llm = LiteLLMClient(model="gpt-4o-mini", max_tokens=4096)
```

---

## Troubleshooting

### Problem: JSON Parsing Errors from SkillManager

**Cause:** Insufficient `max_tokens` for structured output

**Solution:**
```python
llm = LiteLLMClient(model="gpt-4o-mini", max_tokens=2048)  # or higher
```

### Problem: Not Learning Anything

**Checks:**
1. Is `is_learning=True`?
2. Is SkillManager output non-empty? `print(skill_manager_output.update)`
3. Is skillbook being saved? `skillbook.save_to_file(...)`

### Problem: Too Many Skills

**Solution:** SkillManager automatically manages skills via TAG operations. Review with:
```python
skills = skillbook.skills()
print(f"Total: {len(skills)}")
for s in skills[:10]:
    print(f"[{s.id}] +{s.helpful}/-{s.harmful}: {s.content}")
```

### Problem: High API Costs

**Solutions:**
- Use cheaper model: `ace_model="gpt-4o-mini"`
- Disable learning for simple tasks: `is_learning=False`
- Batch learning: Learn only every N tasks

### Problem: Agent Ignores Skillbook Strategies

**Checks:**
1. Are you actually injecting context? `print(enhanced_task)`
2. Does skillbook have skills? `print(len(skillbook.skills()))`
3. Is context clear enough for your agent?

---

## Next Steps

1. **Start Simple:** Use the wrapper class template above
2. **Adapt `_learn()`:** Customize for your agent's output format
3. **Test Without Learning:** Set `is_learning=False` first
4. **Enable Learning:** Turn on and monitor skillbook growth
5. **Iterate:** Improve feedback extraction for better learning

---

## See Also

- **Out-of-box integrations:** ACELiteLLM, ACEAgent (browser-use), ACELangChain
- **Full ACE guide:** [COMPLETE_GUIDE_TO_ACE.md](COMPLETE_GUIDE_TO_ACE.md)
- **API reference:** [API_REFERENCE.md](API_REFERENCE.md)

Questions? Join our [Discord](https://discord.gg/mqCqH7sTyK)
