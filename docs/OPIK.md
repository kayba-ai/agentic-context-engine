# Opik Integration Guide

This guide covers how to use [Opik](https://github.com/comet-ml/opik) for tracing, monitoring, and cost tracking with the ACE framework.

## Overview

ACE framework includes built-in Opik integration for:
- **Tracing**: Track Agent, Reflector, and SkillManager interactions
- **Cost Tracking**: Automatic token usage and LLM cost monitoring
- **Performance Analysis**: Visualize learning loops and skill evolution
- **Debugging**: Inspect detailed traces of the ACE pipeline

## Installation

```bash
# Install ACE with observability features
pip install ace-framework[observability]

# Or for development (includes Opik in optional deps)
uv sync
```

## Starting Opik

### Local Development (Recommended)

Run the local Opik server using Docker:

```bash
# Start Opik server
docker run -d -p 5173:5173 --name opik ghcr.io/comet-ml/opik:latest

# View traces at http://localhost:5173
```

### Comet Cloud

For production, use [Comet's hosted Opik](https://www.comet.com/opik):

```bash
# Set your Comet API key
export COMET_API_KEY="your-api-key"
```

## Quick Start

### Script Initialization

Add this at the start of your `main()` function:

```python
import os
from ace.observability.opik_integration import configure_opik

def main():
    # Initialize Opik with project name
    project_name = os.environ.get("OPIK_PROJECT_NAME", "ace-default")
    opik_integration = configure_opik(project_name=project_name)

    if opik_integration.is_available():
        # Register LiteLLM callback for automatic token tracking
        opik_integration.setup_litellm_callback()
        print(f"Opik tracing enabled for project: {project_name}")

    # Your ACE code here...
```

### Running with Tracing

```bash
# Run with custom project name
OPIK_PROJECT_NAME="my-experiment" uv run python my_script.py

# View traces at http://localhost:5173
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPIK_PROJECT_NAME` | Project name for organizing traces | `ace-framework` |
| `OPIK_DISABLED=true` | Disable all Opik tracing | Not set |
| `OPIK_ENABLED=false` | Alternative way to disable tracing | Not set |
| `OPIK_URL_OVERRIDE` | Custom Opik server URL | `http://localhost:5173/api` |
| `OPIK_WORKSPACE` | Opik workspace name | `default` |

## Component Tracing

### Automatically Traced Components

The following ACE components have built-in tracing:

| Component | Trace Name | Tags |
|-----------|------------|------|
| `Agent.generate()` | `agent_generate` | `["agent"]` |
| `Reflector.reflect()` | `reflector_reflect` | `["reflector"]` |
| `SkillManager.update_skills()` | `skill_manager_update_skills` | `["skill_manager"]` |
| `RecursiveReflector.reflect()` | `recursive_reflector` | `["reflector", "recursive"]` |
| `ReplayAgent.generate()` | `replay_agent_generate` | `["agent", "replay"]` |

### Adding Tracing to Custom Components

Use the `@maybe_track` decorator to add tracing to your own components:

```python
from ace.observability.tracers import maybe_track

class MyCustomComponent:
    @maybe_track(name="my_component_process", tags=["custom", "processing"])
    def process(self, input_data):
        # Your processing logic here
        result = self._do_work(input_data)
        return result
```

The `@maybe_track` decorator:
- Only applies tracing when Opik is installed and enabled
- Gracefully degrades to no-op when Opik is unavailable
- Respects `OPIK_DISABLED` environment variable

## Automatic Token & Cost Tracking

When you call `setup_litellm_callback()`, all LiteLLM calls are automatically tracked with:

- Input/output tokens
- Model used
- Cost per call
- Latency

```python
from ace.observability.opik_integration import configure_opik
from ace.llm_providers.litellm_client import LiteLLMClient

# Initialize Opik
opik_integration = configure_opik(project_name="cost-tracking-demo")
if opik_integration.is_available():
    opik_integration.setup_litellm_callback()

# All LLM calls are now tracked
client = LiteLLMClient(model="gpt-4")
response = client.complete("What is 2+2?")
# Token usage and cost automatically logged to Opik
```

## Viewing Traces

### Local Opik UI

1. Start the Opik server (see above)
2. Open http://localhost:5173 in your browser
3. Select your project from the dropdown
4. Browse traces, spans, and metrics

### Trace Hierarchy

ACE traces are organized hierarchically:

```
[Project: my-experiment]
  └── [Trace: script_run_12345]
        ├── [Span: agent_generate]
        │     └── [LLM Call: gpt-4]
        ├── [Span: reflector_reflect]
        │     └── [LLM Call: gpt-4]
        └── [Span: skill_manager_update_skills]
              └── [LLM Call: gpt-4]
```

## Advanced Usage

### Logging Custom Metrics

Use the `OpikIntegration` class to log custom metrics:

```python
from ace.observability.opik_integration import get_integration

opik = get_integration()

# Log skill evolution
opik.log_skill_evolution(
    skill_id="skill_001",
    skill_content="Always validate input...",
    helpful_count=5,
    harmful_count=1,
    neutral_count=2,
    section="error_handling"
)

# Log adaptation metrics
opik.log_adaptation_metrics(
    epoch=2,
    step=15,
    performance_score=0.85,
    skill_count=12,
    successful_predictions=85,
    total_predictions=100
)
```

### Disabling Tracing for Tests

```bash
# Disable in CI/tests
OPIK_DISABLED=true pytest tests/

# Or in code
import os
os.environ["OPIK_DISABLED"] = "true"
```

### Using with Async Code

The `@maybe_track` decorator works with both sync and async functions:

```python
from ace.observability.tracers import maybe_track

class AsyncComponent:
    @maybe_track(name="async_process", tags=["async"])
    async def process(self, data):
        result = await self._async_work(data)
        return result
```

## Troubleshooting

### "Opik not available"

Install the observability extras:
```bash
pip install ace-framework[observability]
# or
uv add opik
```

### Traces not appearing

1. Check Opik server is running: `curl http://localhost:5173/api/health`
2. Verify `OPIK_DISABLED` is not set
3. Ensure `configure_opik()` is called before any traced functions

### Async warnings

Warnings like "coroutine was never awaited" in synchronous contexts are harmless and can be ignored. They occur when Opik's async internals interact with sync code.

### High memory usage

For long-running scripts, traces accumulate in memory. Consider:
- Running in batches with separate processes
- Flushing traces periodically (if supported by Opik)

## Example: Full ACE Pipeline with Tracing

```python
import os
from ace import Skillbook, Agent, Reflector, SkillManager, OfflineACE
from ace.llm_providers.litellm_client import LiteLLMClient
from ace.observability.opik_integration import configure_opik

def main():
    # 1. Initialize Opik
    project_name = os.environ.get("OPIK_PROJECT_NAME", "ace-training")
    opik_integration = configure_opik(project_name=project_name)
    if opik_integration.is_available():
        opik_integration.setup_litellm_callback()
        print(f"Tracing enabled: {project_name}")

    # 2. Create ACE components (all LLM calls will be traced)
    llm = LiteLLMClient(model="gpt-4")
    skillbook = Skillbook()
    agent = Agent(llm)
    reflector = Reflector(llm)
    skill_manager = SkillManager(llm)

    # 3. Run adaptation (traces appear in Opik)
    adapter = OfflineACE(
        skillbook=skillbook,
        agent=agent,
        reflector=reflector,
        skill_manager=skill_manager
    )

    results = adapter.run(samples, environment, epochs=3)

    # 4. View traces at http://localhost:5173

if __name__ == "__main__":
    main()
```

Run with:
```bash
OPIK_PROJECT_NAME="training-run-001" uv run python train.py
```

## Related Documentation

- [Quick Start Guide](./QUICK_START.md)
- [Integration Guide](./INTEGRATION_GUIDE.md)
- [API Reference](./API_REFERENCE.md)
