# Logfire Trace Querying

Use this guide when you need to inspect or analyze traces already collected in
the `kayba/ace` Logfire project.

This is an operational guide for coding agents and scripts. It focuses on how
to authenticate, query the Logfire API, and safely turn the API response into
trace records you can analyze locally.

## When To Use This Guide

Read this guide before you:

- inspect the latest production or benchmark traces in Logfire
- fetch a specific trace by `trace_id`
- summarize failures, exceptions, latency, or model usage from Logfire data
- export collected traces from Logfire into local analysis code

If you are adding runtime instrumentation, also read the existing Logfire
observability notes in `ace/observability/__init__.py` and the Claude SDK guide.

## Project And Credentials

The Logfire project used in this repository is:

- organization: `kayba`
- project: `ace`

### Authentication Rules

The Logfire query API officially requires a **read token**.

Preferred auth order:

1. `LOGFIRE_READ_TOKEN` if it is already available
2. a fresh read token created for `kayba/ace`
3. `LOGFIRE_TOKEN` from the repo-local `.env` only as a bootstrap path when you
   need to create or recover a read token

Do not assume `LOGFIRE_TOKEN` can query traces. In this repository it is
usually the project write token used by the SDK for emission, and the query API
will reject it with `401 Invalid token`.

Do not hardcode token values into source files, docs, tests, or shell history.
Do not print full tokens in logs or user-facing output.

### Repo-Local `.env`

This repository commonly stores Logfire credentials in the repo-root `.env`.
If the current shell does not already have `LOGFIRE_TOKEN`, load `.env` first.

Python:

```python
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(".env"))
```

Shell:

```bash
set -a
source .env
set +a
```

## Endpoint And Region Selection

Preferred query endpoint:

- `https://logfire-api.pydantic.dev/v1/query`

Regional endpoints also work:

- US: `https://logfire-us.pydantic.dev/v1/query`
- EU: `https://logfire-eu.pydantic.dev/v1/query`

Logfire tokens encode the region. If you need to derive the regional URL, the
token format is typically `pylf_v<version>_<region>_<secret>`, where `<region>`
is usually `us` or `eu`.

## Recommended Access Pattern In This Repo

Inside this repository, prefer **direct HTTP requests from `.venv/bin/python`**
over the Logfire CLI when running in a sandboxed agent environment.

Reasons:

- the Logfire CLI writes logs under `~/.logfire/`, which may be blocked
- `uv run ...` may try to use `~/.cache/uv/`, which may also be blocked
- direct HTTP requests with `requests` are simpler and more predictable

If you must use `uv` in a restricted sandbox, set:

```bash
UV_CACHE_DIR=/tmp/uv-cache
```

## Query API Basics

The query API accepts SQL against Logfire tables such as `records`.

Minimal HTTP shape:

```http
GET /v1/query?sql=SELECT%20... HTTP/1.1
Authorization: Bearer <read-token>
Accept: application/json
```

Use `GET` for the query API in this repo. `POST /v1/query` currently returns
`405 Method Not Allowed`.

Useful query parameters:

- `sql`: required SQL query
- `limit`: response row cap, default `500`, maximum `10000`
- `min_timestamp`: optional lower time bound
- `max_timestamp`: optional upper time bound
- `row_oriented`: may be accepted by the API, but agents in this repo should not
  rely on it to change the payload shape

Official Logfire query API docs:

- <https://logfire.pydantic.dev/docs/how-to-guides/query-api/>

## Important Response Shape

Treat Logfire responses as **column-oriented JSON**, not as a list of row dicts.
Even when `row_oriented` is provided, the safe assumption in this repo is still
that the payload will come back in `columns`.

Example shape:

```json
{
  "columns": [
    {"name": "trace_id", "values": ["abc", "def"]},
    {"name": "message", "values": ["agent run", "chat ..."]}
  ]
}
```

Convert it to rows before analysis:

```python
def columns_to_rows(payload: dict) -> list[dict]:
    columns = payload["columns"]
    names = [col["name"] for col in columns]
    values = [col["values"] for col in columns]
    return [dict(zip(names, row)) for row in zip(*values)]
```

Do not assume `response.json()` is already a list.

## Canonical Python Snippet

Use this as the default pattern for agents.

```python
from __future__ import annotations

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(".env"))

READ_TOKEN = os.environ.get("LOGFIRE_READ_TOKEN")
if not READ_TOKEN:
    raise RuntimeError("LOGFIRE_READ_TOKEN is required for Logfire queries")

BASE_URL = "https://logfire-api.pydantic.dev"


def query_logfire(sql: str, *, limit: int = 1000) -> list[dict]:
    resp = requests.get(
        f"{BASE_URL}/v1/query",
        params={"sql": sql, "limit": limit},
        headers={
            "Authorization": f"Bearer {READ_TOKEN}",
            "Accept": "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    columns = payload["columns"]
    names = [col["name"] for col in columns]
    values = [col["values"] for col in columns]
    return [dict(zip(names, row)) for row in zip(*values)]
```

## High-Value Queries

### 1. Latest Root Traces

Start here when the user asks for the latest traces.

```sql
SELECT
  start_timestamp,
  trace_id,
  span_id,
  service_name,
  message,
  span_name,
  level,
  duration
FROM records
WHERE parent_span_id IS NULL
ORDER BY start_timestamp DESC
LIMIT 20
```

Notes:

- root traces usually have `parent_span_id IS NULL`
- in this project, root messages are often `agent run`

### 2. Full Trace By `trace_id`

Use this after identifying a trace worth inspecting.

```sql
SELECT
  start_timestamp,
  span_id,
  parent_span_id,
  message,
  span_name,
  level,
  duration,
  service_name
FROM records
WHERE trace_id = '<TRACE_ID>'
ORDER BY start_timestamp ASC
LIMIT 500
```

### 3. Exceptions Inside One Trace

Use this to separate top-level failures from recoverable tool retries.

```sql
SELECT
  start_timestamp,
  message,
  span_name,
  level,
  exception_type,
  exception_message
FROM records
WHERE trace_id = '<TRACE_ID>' AND is_exception = true
ORDER BY start_timestamp ASC
LIMIT 100
```

### 4. Quick Trace Stats

Use this for a compact summary before drilling deeper.

```sql
SELECT
  COUNT(*) AS record_count,
  SUM(CASE WHEN is_exception THEN 1 ELSE 0 END) AS exception_count,
  MIN(start_timestamp) AS first_seen,
  MAX(start_timestamp) AS last_seen
FROM records
WHERE trace_id = '<TRACE_ID>'
```

### 5. Model Usage Within A Trace

Useful when RR or sub-agent activity is suspected.

```sql
SELECT
  span_name,
  COUNT(*) AS call_count,
  SUM(duration) AS total_duration
FROM records
WHERE trace_id = '<TRACE_ID>'
GROUP BY span_name
ORDER BY total_duration DESC
LIMIT 50
```

## How To Read Common Patterns

Typical messages you may see:

- `agent run`: root span for a run
- `chat <model>`: one LLM call
- `running tool: execute_code`: sandbox code execution
- `running tool: batch_analyze`: batch semantic analysis
- `sub run`: nested sub-agent execution

Interpretation guidance:

- a trace with `0` exceptions is usually a clean run, but still inspect duration
  and child spans if the user asked for performance analysis
- `ToolRetryError` inside `execute_code` often means the RR sandbox made a bad
  attempt and retried; this is not automatically a top-level pipeline failure
- `UsageLimitExceeded` indicates an internal request budget or tool budget issue,
  not necessarily a transport failure
- long traces with many `sub run` spans often indicate recursive reflection or
  batch analysis behavior

## Safety And Operational Rules

- Always keep SQL narrow. Add `LIMIT`, and add time filters when possible.
- Prefer starting from root traces, then drill into one `trace_id`.
- Never paste secret tokens into committed docs, code, or logs.
- Never claim the “latest” trace without actually querying Logfire first.
- When reporting times to users, include the full UTC timestamp.
- If the sandbox blocks network access, request escalation instead of guessing.

## Troubleshooting

### `401 Unauthorized`

Likely causes:

- token is expired
- token is the wrong token type
- token belongs to the wrong project or region
- you are trying to use `LOGFIRE_TOKEN` instead of `LOGFIRE_READ_TOKEN`

Action:

- use a valid read token for `kayba/ace`
- prefer the global API endpoint if region selection is unclear

### `429 Too Many Requests`

Likely causes:

- several query requests were fired back-to-back while drilling into the same trace
- one large query pulled too many records or wide JSON attributes

Action:

- pause briefly and retry with fewer queries
- prefer one narrow query over several broad exploratory ones
- fetch counts first, then ordered records for a single `trace_id`

### `200 OK` But No Rows

Likely causes:

- wrong project
- query window is too narrow
- data is older than the default filtered range in a helper you are using

Action:

- remove accidental timestamp filters
- query root traces first
- widen the time range explicitly

### CLI Fails In Sandbox

Likely causes:

- `logfire` CLI trying to write under `~/.logfire/`
- `uv` trying to write under `~/.cache/uv/`

Action:

- prefer `.venv/bin/python` plus `requests`
- if `uv` is required, set `UV_CACHE_DIR=/tmp/uv-cache`

## Recommended Workflow For Agents

When asked to inspect the latest traces:

1. Load `.env` if needed.
2. Confirm you have a valid read token path.
3. Query the latest root traces.
4. Pick the newest interesting `trace_id`.
5. Fetch trace stats.
6. Fetch exception rows.
7. Fetch ordered records for that trace.
8. Summarize:
   - exact UTC timestamp
   - total duration
   - record count
   - exception count
   - model/tool pattern
   - likely failure mode

This is the default procedure for “look at the latest traces in Logfire”.
