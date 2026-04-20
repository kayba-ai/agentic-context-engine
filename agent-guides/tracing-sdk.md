# Kayba Tracing SDK

Use this guide when you need to instrument agent code with Kayba tracing.

## When To Use This Guide

Read this guide before you:

- add tracing to new or existing agent code
- create examples that send traces to Kayba
- debug why traces are not appearing in the dashboard

## Module Location

All tracing code lives in `ace/tracing/`. The public API is re-exported from
`ace/tracing/__init__.py`. The implementation is in `ace/tracing/_wrapper.py`.

## Installation

Tracing requires the optional `tracing` extra:

```bash
pip install ace-framework[tracing]
```

This pulls in `mlflow` as the underlying tracing backend.

## Configuration

```python
from ace.tracing import configure

configure(
    api_key="...",          # or set KAYBA_SDK_KEY / KAYBA_API_KEY env var
    base_url="...",         # optional, defaults to https://use.kayba.ai
    experiment="my-exp",    # optional MLflow experiment name
    folder="production",    # optional dashboard folder
)
```

The `configure()` function sets the MLflow tracking URI to
`{base_url}/api/mlflow` and stores the API key in
`MLFLOW_TRACKING_TOKEN`.

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `KAYBA_SDK_KEY` or `KAYBA_API_KEY` | API key (alternative to `api_key=`) |
| `KAYBA_API_URL` | Base URL override |

## Core API

### `@trace` decorator

Wraps a function to create a trace span. Supports bare and parameterized forms:

```python
from ace.tracing import trace

@trace
def my_agent(query: str) -> str: ...

@trace(name="custom", span_type="LLM", attributes={"model": "glm-4-plus"})
def llm_call(messages): ...
```

### `start_span()` context manager

Creates a child span within an active trace:

```python
from ace.tracing import start_span

with start_span("retrieval") as span:
    span.set_inputs({"query": query})
    results = search(query)
    span.set_outputs({"count": len(results)})
```

### Other functions

- `set_folder(name)` / `get_folder()` — change/read the dashboard folder
- `enable()` / `disable()` — toggle tracing on/off
- `get_trace(trace_id)` — fetch a trace by ID
- `search_traces(experiment_names=[...])` — search traces

## Using with OpenAI-Compatible Endpoints

The tracing SDK is LLM-agnostic. Use any OpenAI-compatible client (Zhipu GLM,
vLLM, Ollama, LiteLLM, etc.) and wrap calls with `@trace`:

```python
from openai import OpenAI
from ace.tracing import configure, trace

configure(api_key=os.environ["KAYBA_SDK_KEY"])

client = OpenAI(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
)

@trace(name="llm_call", span_type="LLM")
def llm_call(messages):
    return client.chat.completions.create(
        model="glm-5.1",
        messages=messages,
    ).choices[0].message.content
```

The `OPENAI_BASE_URL` in `.env` points to `https://api.z.ai/api/coding/paas/v4`
(Zhipu AI). Any model served there (e.g. `glm-4-plus`) works.

## Span Nesting

Decorated functions called within other decorated functions produce a nested
trace tree automatically:

```
@trace pipeline
├── @trace research_agent
│   ├── start_span("build_prompt")
│   └── @trace llm_call
└── @trace summariser_agent
    ├── start_span("build_prompt")
    └── @trace llm_call
```

## Current Limitations

- **No async support**: the `@trace` decorator only wraps sync functions. Async
  functions will return a coroutine instead of awaiting it.
- **No cross-process context propagation**: each `@trace` root creates an
  independent trace. There is no mechanism to link traces across agents running
  in separate processes.
- **No agent identity tagging**: spans are not automatically tagged with an
  agent name or ID.

## Example

See `examples/tracing_glm_example.py` for a full runnable two-agent pipeline
(research + summarise) instrumented with the tracing SDK.
