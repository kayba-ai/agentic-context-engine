# ACE Integration Patterns

Common patterns for integrating ACE with different types of agents.

Each pattern includes:
- Complete code example
- When to use it
- Key considerations

---

## Table of Contents

1. [REST API-Based Agents](#rest-api-based-agents)
2. [Multi-Step Workflow Agents](#multi-step-workflow-agents)
3. [Tool-Using Agents](#tool-using-agents)
4. [Async Agents](#async-agents)
5. [Chat-Based Agents](#chat-based-agents)
6. [Batch Processing Agents](#batch-processing-agents)
7. [Streaming Agents](#streaming-agents)
8. [Error-Prone Agents](#error-prone-agents)

---

## REST API-Based Agents

### When to Use
- Your agent is a REST API service
- Remote execution (cloud-based agents)
- Stateless request/response pattern

### Pattern

```python
from ace import Playbook, LiteLLMClient, Reflector, Curator
from ace.integrations.base import wrap_playbook_context
from ace.roles import GeneratorOutput
import requests

class ACEAPIAgent:
    """Wraps REST API agent with ACE learning."""

    def __init__(self, api_url: str, api_key: str = None, ace_model: str = "gpt-4o-mini"):
        self.api_url = api_url
        self.api_key = api_key
        self.playbook = Playbook()

        # ACE components
        llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(llm)
        self.curator = Curator(llm)

    def execute(self, task: str):
        """Execute task via API with ACE learning."""
        # Inject context
        if self.playbook.bullets():
            task = f"{task}\n\n{wrap_playbook_context(self.playbook)}"

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
        generator_output = GeneratorOutput(
            reasoning=f"API call for task: {task}",
            final_answer=output,
            bullet_ids=[],
            raw={"success": success}
        )

        # Feedback
        feedback = f"API call {'succeeded' if success else 'failed'}. Output: {output[:200]}"

        # Reflect + Curate
        reflection = self.reflector.reflect(
            question=task,
            generator_output=generator_output,
            playbook=self.playbook,
            feedback=feedback
        )

        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=f"API task: {task}",
            progress=task
        )

        self.playbook.apply_delta(curator_output.delta)

# Usage
agent = ACEAPIAgent(api_url="https://api.example.com", api_key="...")
result = agent.execute("Process user data")
agent.playbook.save_to_file("api_agent_learned.json")
```

### Key Considerations
- Handle timeouts and retries
- Parse API error messages for better feedback
- Consider rate limiting (don't learn on every call if high volume)

---

## Multi-Step Workflow Agents

### When to Use
- Agent executes multiple sequential steps
- Each step has its own outcome
- Want to learn from entire workflow

### Pattern

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
        self.playbook = Playbook()

        llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(llm)
        self.curator = Curator(llm)

    def run(self, task: str) -> WorkflowResult:
        """Execute workflow with ACE learning."""
        # Inject context
        if self.playbook.bullets():
            task = f"{task}\n\n{wrap_playbook_context(self.playbook)}"

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
        generator_output = GeneratorOutput(
            reasoning=feedback,  # Full workflow trace
            final_answer=result.final_output,
            bullet_ids=[],
            raw={
                "total_steps": len(result.steps),
                "successful_steps": sum(1 for s in result.steps if s.success),
                "total_duration": sum(s.duration for s in result.steps)
            }
        )

        # Reflect + Curate
        reflection = self.reflector.reflect(
            question=task,
            generator_output=generator_output,
            playbook=self.playbook,
            feedback=feedback
        )

        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=f"Multi-step workflow: {task}",
            progress=f"Completed {len(result.steps)} steps"
        )

        self.playbook.apply_delta(curator_output.delta)

# Usage
workflow_agent = MyWorkflowAgent()
ace_agent = ACEWorkflowAgent(workflow_agent)
result = ace_agent.run("Complete data pipeline")
```

### Key Considerations
- Include step-by-step trace in feedback for better learning
- Track timing information to learn performance patterns
- Distinguish partial failures (some steps succeed) from total failures

---

## Tool-Using Agents

### When to Use
- Agent has access to external tools/functions
- Tool selection and usage is part of learning
- Want to inject context into system message or tool descriptions

### Pattern

```python
class ACEToolAgent:
    """Wraps tool-using agent with ACE learning."""

    def __init__(self, agent, ace_model: str = "gpt-4o-mini"):
        self.agent = agent
        self.playbook = Playbook()
        self.original_system_message = agent.system_message

        llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(llm)
        self.curator = Curator(llm)

    def run(self, task: str):
        """Execute with tool access and ACE learning."""
        # Inject playbook into system message (not task)
        if self.playbook.bullets():
            context = wrap_playbook_context(self.playbook)
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
        generator_output = GeneratorOutput(
            reasoning=feedback,
            final_answer=result["output"],
            bullet_ids=[],
            raw={"tools_used": [t["name"] for t in tools_used]}
        )

        # Reflect + Curate
        reflection = self.reflector.reflect(
            question=task,
            generator_output=generator_output,
            playbook=self.playbook,
            feedback=feedback
        )

        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=f"Tool-using task: {task}",
            progress=f"Used {len(tools_used)} tools"
        )

        self.playbook.apply_delta(curator_output.delta)

# Usage
tool_agent = MyToolUsingAgent(tools=[...])
ace_agent = ACEToolAgent(tool_agent)
result = ace_agent.run("Analyze data and send report")
```

### Key Considerations
- Inject context into system message (not task) for better tool selection
- Track which tools were used for learning tool selection patterns
- Include tool outcomes in feedback

---

## Async Agents

### When to Use
- Agent operations are async (browser automation, async APIs)
- Need non-blocking execution
- Want to maintain async interface

### Pattern

```python
import asyncio

class ACEAsyncAgent:
    """Wraps async agent with ACE learning."""

    def __init__(self, async_agent, ace_model: str = "gpt-4o-mini"):
        self.agent = async_agent
        self.playbook = Playbook()

        llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(llm)
        self.curator = Curator(llm)

    async def run(self, task: str):
        """Async execution with ACE learning."""
        # Inject context
        if self.playbook.bullets():
            task = f"{task}\n\n{wrap_playbook_context(self.playbook)}"

        # Execute (async)
        result = await self.agent.execute(task)

        # Learn (sync operations in thread)
        await asyncio.to_thread(self._learn, task, result)

        return result

    def _learn(self, task: str, result):
        """Sync learning pipeline (runs in thread)."""
        generator_output = GeneratorOutput(
            reasoning=f"Async task: {task}",
            final_answer=result["output"],
            bullet_ids=[],
            raw={"success": result["success"]}
        )

        feedback = f"Async task {'succeeded' if result['success'] else 'failed'}"

        reflection = self.reflector.reflect(
            question=task,
            generator_output=generator_output,
            playbook=self.playbook,
            feedback=feedback
        )

        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=task,
            progress=task
        )

        self.playbook.apply_delta(curator_output.delta)

# Usage
async def main():
    async_agent = MyAsyncAgent()
    ace_agent = ACEAsyncAgent(async_agent)

    result = await ace_agent.run("Fetch and process data")
    print(f"Result: {result}")

asyncio.run(main())
```

### Key Considerations
- Use `asyncio.to_thread()` to run sync Reflector/Curator in background
- Don't block async event loop with sync ACE operations
- Consider batching learning for high-throughput async systems

---

## Chat-Based Agents

### When to Use
- Agent maintains conversation history
- Multi-turn interactions
- Want to learn from entire conversation

### Pattern

```python
class ACEChatAgent:
    """Wraps chat agent with per-conversation learning."""

    def __init__(self, chat_agent, ace_model: str = "gpt-4o-mini"):
        self.agent = chat_agent
        self.playbook = Playbook()
        self.conversation_history = []

        llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(llm)
        self.curator = Curator(llm)

    def chat(self, message: str) -> str:
        """Single chat turn with context injection."""
        # Inject playbook on first message
        if len(self.conversation_history) == 0 and self.playbook.bullets():
            system_context = wrap_playbook_context(self.playbook)
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
        generator_output = GeneratorOutput(
            reasoning=conversation,
            final_answer=self.conversation_history[-1]["assistant"],
            bullet_ids=[],
            raw={"turns": len(self.conversation_history)}
        )

        feedback_text = (
            f"Conversation {'succeeded' if success else 'failed'} "
            f"over {len(self.conversation_history)} turns. {feedback}"
        )

        reflection = self.reflector.reflect(
            question=self.conversation_history[0]["user"],
            generator_output=generator_output,
            playbook=self.playbook,
            feedback=feedback_text
        )

        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=f"Multi-turn conversation ({len(self.conversation_history)} turns)",
            progress="Conversation completed"
        )

        self.playbook.apply_delta(curator_output.delta)

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
ace_agent.playbook.save_to_file("chat_agent_learned.json")
```

### Key Considerations
- Learn from complete conversation (not individual turns)
- Inject playbook context at conversation start
- Allow manual feedback at conversation end

---

## Batch Processing Agents

### When to Use
- Processing large batches of similar tasks
- Want to amortize learning costs
- Need high throughput

### Pattern

```python
class ACEBatchAgent:
    """Wraps agent with batched learning."""

    def __init__(self, agent, ace_model: str = "gpt-4o-mini", learn_every: int = 10):
        self.agent = agent
        self.playbook = Playbook()
        self.learn_every = learn_every
        self.pending_results = []

        llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(llm)
        self.curator = Curator(llm)

    def process(self, task: str):
        """Process single task (learn in batches)."""
        # Inject context
        if self.playbook.bullets():
            task = f"{task}\n\n{wrap_playbook_context(self.playbook)}"

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

        generator_output = GeneratorOutput(
            reasoning=f"Batch processing: {feedback}",
            final_answer=result["output"],
            bullet_ids=[],
            raw={"batch_size": len(self.pending_results), "success_rate": successes / len(self.pending_results)}
        )

        reflection = self.reflector.reflect(
            question=task,
            generator_output=generator_output,
            playbook=self.playbook,
            feedback=feedback
        )

        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=f"Batch processing ({len(self.pending_results)} items)",
            progress="Batch completed"
        )

        self.playbook.apply_delta(curator_output.delta)

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
ace_agent.playbook.save_to_file("batch_learned.json")
```

### Key Considerations
- Balance learning frequency vs cost (learn_every parameter)
- Call `flush()` at end to learn from remaining items
- Consider success rate in batch feedback

---

## Streaming Agents

### When to Use
- Agent streams responses token-by-token
- Want to maintain streaming interface
- Learn after complete stream

### Pattern

```python
class ACEStreamingAgent:
    """Wraps streaming agent with post-stream learning."""

    def __init__(self, streaming_agent, ace_model: str = "gpt-4o-mini"):
        self.agent = streaming_agent
        self.playbook = Playbook()

        llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(llm)
        self.curator = Curator(llm)

    def stream(self, task: str):
        """Stream response with learning after completion."""
        # Inject context
        if self.playbook.bullets():
            task = f"{task}\n\n{wrap_playbook_context(self.playbook)}"

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
        generator_output = GeneratorOutput(
            reasoning=f"Streamed response for: {task}",
            final_answer=response,
            bullet_ids=[],
            raw={"response_length": len(response)}
        )

        feedback = f"Streamed {len(response)} characters"

        reflection = self.reflector.reflect(
            question=task,
            generator_output=generator_output,
            playbook=self.playbook,
            feedback=feedback
        )

        curator_output = self.curator.curate(
            reflection=reflection,
            playbook=self.playbook,
            question_context=task,
            progress="Streaming completed"
        )

        self.playbook.apply_delta(curator_output.delta)

# Usage
streaming_agent = MyStreamingAgent()
ace_agent = ACEStreamingAgent(streaming_agent)

for chunk in ace_agent.stream("Generate report"):
    print(chunk, end="", flush=True)
```

### Key Considerations
- Collect full response before learning
- Don't block streaming (learn after completion)
- Maintain streaming interface for caller

---

## Error-Prone Agents

### When to Use
- Agent frequently fails or throws exceptions
- Want to learn from failures
- Need robust error handling

### Pattern

```python
class ACERobustAgent:
    """Wraps agent with error handling and failure learning."""

    def __init__(self, agent, ace_model: str = "gpt-4o-mini", max_retries: int = 3):
        self.agent = agent
        self.playbook = Playbook()
        self.max_retries = max_retries

        llm = LiteLLMClient(model=ace_model, max_tokens=2048)
        self.reflector = Reflector(llm)
        self.curator = Curator(llm)

    def run(self, task: str):
        """Execute with retries and error learning."""
        # Inject context
        if self.playbook.bullets():
            task = f"{task}\n\n{wrap_playbook_context(self.playbook)}"

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
            generator_output = GeneratorOutput(
                reasoning=f"Task: {task}. {feedback}",
                final_answer=final_answer,
                bullet_ids=[],
                raw={"success": success, "error": error}
            )

            # Reflect + Curate
            reflection = self.reflector.reflect(
                question=task,
                generator_output=generator_output,
                playbook=self.playbook,
                feedback=feedback
            )

            curator_output = self.curator.curate(
                reflection=reflection,
                playbook=self.playbook,
                question_context=f"Task ({'success' if success else 'failure'}): {task}",
                progress="Execution completed"
            )

            self.playbook.apply_delta(curator_output.delta)

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
    # But playbook learned from the failure!
```

### Key Considerations
- Learn from both successes AND failures
- Wrap learning in try/except (never crash from learning)
- Include error details in feedback for failure pattern learning

---

## See Also

- [Integration Guide](INTEGRATION_GUIDE.md) - Comprehensive integration documentation
- [ACE Browser-Use Integration](../ace/integrations/browser_use.py) - Reference implementation
- **Out-of-Box Integrations:** ACELiteLLM, ACEAgent (browser-use), ACELangChain
