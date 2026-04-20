# Tracing

Send agent traces to Kayba with a few lines of code. The `ace.tracing` module wraps all tracing functionality behind a Kayba-native API — just configure your API key and instrument your functions.

## Installation

```bash
pip install ace-framework[tracing]
```

## Quick Start

```python
from ace.tracing import configure, trace, start_span

configure(api_key="kb-...")

@trace
def my_agent(query: str) -> str:
    with start_span("retrieval") as span:
        span.set_inputs({"query": query})
        results = search(query)
        span.set_outputs(results)
    return synthesize(results)

my_agent("What is the capital of France?")
```

Every call to `my_agent` now produces a trace visible in your Kayba dashboard.

## Configuration

### configure()

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str` | `None` | Kayba API key. Falls back to `KAYBA_API_KEY` env var |
| `base_url` | `str` | `None` | API base URL. Falls back to `KAYBA_API_URL`, then `https://use.kayba.ai` |
| `experiment` | `str` | `None` | Optional experiment name for grouping traces |
| `folder` | `str` | `None` | Optional folder name — traces are filed into this folder in the dashboard |

### Environment Variables

| Variable | Description |
|----------|-------------|
| `KAYBA_API_KEY` | API key (alternative to passing `api_key=` directly) |
| `KAYBA_API_URL` | Base URL override (default: `https://use.kayba.ai`) |

### Minimal Configuration

If `KAYBA_API_KEY` is set in your environment, configuration is a single line:

```python
from ace.tracing import configure
configure()
```

Or skip the import entirely and configure from `ace`:

```python
from ace import configure_tracing
configure_tracing(api_key="kb-...")
```

## Instrumenting Your Code

### @trace decorator

Wrap any function to automatically capture its inputs, outputs, and duration:

```python
from ace.tracing import trace

@trace
def classify(text: str) -> str:
    return call_llm(f"Classify: {text}")
```

Add metadata with optional parameters:

```python
@trace(name="custom-name", span_type="LLM", attributes={"model": "gpt-4o"})
def classify(text: str) -> str:
    return call_llm(f"Classify: {text}")
```

### start_span context manager

For finer-grained control within a function:

```python
from ace.tracing import trace, start_span

@trace
def my_agent(query: str) -> str:
    with start_span("retrieve") as span:
        span.set_inputs({"query": query})
        docs = vector_search(query)
        span.set_outputs({"count": len(docs)})

    with start_span("generate") as span:
        span.set_inputs({"docs": docs})
        answer = llm_generate(docs, query)
        span.set_outputs({"answer": answer})

    return answer
```

Spans nest automatically — child spans created inside a parent span are linked in the trace tree.

### Nested function tracing

Decorated functions called within other decorated functions produce a nested trace:

```python
from ace.tracing import trace

@trace
def retrieve(query: str) -> list[str]:
    return vector_search(query)

@trace
def generate(docs: list[str], query: str) -> str:
    return llm_call(docs, query)

@trace
def agent(query: str) -> str:
    docs = retrieve(query)      # child span
    return generate(docs, query) # child span
```

Calling `agent("...")` produces a single trace with three spans in a tree.

## Folders

Traces can be organized into folders in the Kayba dashboard. Set the folder at configuration time or change it dynamically:

```python
from ace.tracing import configure, set_folder, trace

# Set folder at configure time
configure(api_key="kb-...", folder="production")

@trace
def my_agent(query: str) -> str:
    ...

# Change folder mid-session
set_folder("staging")

# Clear folder (traces go to Unfiled)
set_folder(None)
```

All traces created after `set_folder()` are tagged with the new folder. Previously sent traces are not affected.

## Enabling / Disabling

```python
from ace.tracing import enable, disable

disable()  # temporarily stop sending traces
# ... untraced code ...
enable()   # resume
```

## Retrieving Traces

```python
from ace.tracing import get_trace, search_traces

# Fetch a specific trace by ID
t = get_trace("abc123")

# Search recent traces
traces = search_traces()

# Search within a specific experiment
traces = search_traces(experiment_names=["my-experiment"])
```

## Full API Reference

| Function | Description |
|----------|-------------|
| `configure()` | Set API key, base URL, experiment, and folder |
| `trace` | Decorator — auto-instruments a function |
| `start_span()` | Context manager — create a child span with manual inputs/outputs |
| `set_folder()` | Change the target folder for subsequent traces |
| `get_folder()` | Return the currently configured folder |
| `enable()` | Re-enable tracing after disabling |
| `disable()` | Temporarily stop sending traces |
| `get_trace()` | Retrieve a trace by ID |
| `search_traces()` | Search for traces by experiment |
