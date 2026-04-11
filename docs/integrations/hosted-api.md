# Kayba Hosted API

The Kayba hosted API lets you upload traces, generate insights, and pull optimised prompts without running ACE roles locally. The `kayba` CLI wraps every API endpoint.

## Prerequisites

1. A Kayba API key (set `KAYBA_API_KEY` or pass `--api-key` to every command).
2. An Anthropic API key (set `ANTHROPIC_API_KEY`) — used server-side for LLM calls when generating insights.
3. Install the `cloud` extra:

```bash
uv tool install 'ace-framework[cloud]' --python 3.12
```

Quote the extra in `zsh`/`bash` so `[cloud]` is not treated as a shell glob.

Or if you installed from source:

```bash
uv sync
```

## Authentication

Every command reads `KAYBA_API_KEY` from the environment. You can also pass it explicitly:

```bash
export KAYBA_API_KEY=your-key-here
kayba traces list
```

The default API endpoint is `https://use.kayba.ai/api`. Override it with `KAYBA_API_URL` or `--base-url`.

## Where do traces come from?

Kayba does **not** auto-ingest local transcripts from Claude Code, Codex, Cursor,
or other coding agents. The hosted API and web UI only show traces that you
explicitly upload.

- **Claude Code:** session transcripts are typically written under
  `~/.claude/projects/<slug>/*.jsonl`
- **Codex:** local session logs are discoverable under
  `~/.codex/sessions/YYYY/MM/DD/*.jsonl`
- **Cursor:** no auto-ingest; locate or export the transcript files from your
  setup first, then upload them manually

Copy-pasteable examples:

```bash
kayba traces upload ~/.claude/projects/<slug>/
kayba traces upload ~/.codex/sessions/2026/04/10/
```

You can point `kayba traces upload` at a single file, a glob-expanded file list,
or a directory. Directories are walked recursively, so uploading the project or
day folder is usually the simplest option.

## CLI Reference

### Trace management

```bash
# List uploaded traces
kayba traces list
kayba traces list --json          # machine-parseable output

# View a trace
kayba traces show TRACE_ID
kayba traces show TRACE_ID --meta  # metadata only, no content
kayba traces show TRACE_ID --json

# Upload traces
kayba traces upload trace.md
kayba traces upload session.jsonl  # common for Claude Code / OpenClaw
kayba traces upload traces/       # directory (recursive)
kayba traces upload --type json traces/  # force file type
cat trace.md | kayba upload -     # pipe from stdin (top-level alias)

# Delete traces
kayba traces delete ID1 ID2
kayba traces delete ID1 --force   # skip confirmation
```

Files larger than 350k characters are rejected by the API. The CLI skips them
locally and tells you to split or trim the trace first. Supported types are
auto-detected from the extension: `.md`/`.markdown` → `md`,
`.json`/`.jsonl` → `json`, everything else → `txt`.

### Run the pipeline

The `run` command combines trace selection and pipeline execution:

```bash
# Interactive mode (visual checkbox selector)
kayba run

# Select all traces
kayba run --all --wait

# Explicit trace IDs
kayba run --traces ID1 --traces ID2

# Custom model, epochs, and reflector mode
kayba run --all --model claude-opus-4-6 --epochs 3 --reflector-mode recursive --wait

# Machine-parseable output (for agents/scripts)
kayba run --all --json
```

In interactive mode (`kayba run` with no flags), a visual checkbox selector lets you pick traces with arrow keys, space to toggle, and enter to confirm. Requires the `questionary` package (included in the `cloud` extra).

In programmatic mode (`--traces`, `--all`, or `--json`), no prompts are shown — suitable for agents and scripts.

Options:

| Flag | Description |
|------|-------------|
| `--traces ID` | Trace IDs to analyse (repeatable) |
| `--all` | Select all uploaded traces |
| `--model` | `claude-sonnet-4-6` or `claude-opus-4-6` |
| `--epochs N` | Number of analysis epochs |
| `--reflector-mode` | `recursive` or `standard` |
| `--anthropic-key` | Anthropic API key for server-side LLM calls |
| `--wait` | Poll until the job completes |
| `--json` | Machine-parseable JSON output |

### Generate insights

```bash
# From all uploaded traces
kayba insights generate --wait

# Specific traces
kayba insights generate --traces ID1 --traces ID2

# Custom model and epochs
kayba insights generate --model claude-opus-4-6 --epochs 3 --wait
```

### List and triage insights

```bash
# List all
kayba insights list

# Filter by status
kayba insights list --status pending

# JSON output
kayba insights list --json

# Accept specific insights
kayba insights triage --accept ID1 --accept ID2

# Accept all pending
kayba insights triage --accept-all

# Reject with a note
kayba insights triage --reject ID1 --note "Too vague"
```

### Generate and pull prompts

```bash
# Generate a prompt from accepted insights
kayba prompts generate

# Generate with a label and save to file
kayba prompts generate --label "v2-coding" -o prompt.md

# List prompt versions
kayba prompts list

# Pull latest prompt
kayba prompts pull

# Pull specific version
kayba prompts pull --id PROMPT_ID -o skillbook-prompt.md

# Pretty-print full JSON
kayba prompts pull --pretty

# Install the latest prompt into Claude Code's instruction file
kayba prompts install --target claude-code

# Install a local prompt export into AGENTS.md
kayba prompts install --input prompt.md --target universal
```

To install the generated prompt into `CLAUDE.md`, `AGENTS.md`, or
`.cursorrules` without duplicating prior runs, see
[Using your generated prompt](#using-your-generated-prompt).

### Integrations

Manage connections to external trace platforms (MLflow, LangSmith).

```bash
# List configured integrations
kayba integrations list
kayba integrations list --json

# Interactively configure an integration
kayba integrations configure mlflow
kayba integrations configure langsmith

# Test a connection
kayba integrations test langsmith
kayba integrations test mlflow
```

The `configure` command prompts for each field interactively:

- **MLflow**: tracking URI, auth type (none/basic/bearer/databricks), token, username, experiment name
- **LangSmith**: API URL (defaults to `https://api.smith.langchain.com`, use `https://eu.api.smith.langchain.com` for EU), API key, project name

After saving, the connection is automatically tested. Credentials are stored in your Kayba account settings (DynamoDB), accessible from both the CLI and the web dashboard.

### Job status and materialisation

```bash
# Check job status
kayba status JOB_ID

# Poll until complete
kayba status JOB_ID --wait --interval 10

# Materialise results into the skillbook
kayba materialize JOB_ID
```

### Batch pre-processing

The `batch` command groups traces into batches before analysis. It works in two modes:

**Prepare mode** (default) — extracts trace metadata and prints a classification prompt:

```bash
kayba batch traces/
```

This writes a skeleton `batches.json` and prints a prompt to stdout. Pipe it to an LLM (e.g. Claude Code) to fill in the batch assignments.

**Apply mode** — validates and optionally uploads a batch plan:

```bash
# Validate only
kayba batch traces/ --apply batches.json

# Validate and upload each batch
kayba batch traces/ --apply batches.json --upload
```

Options:

| Flag | Description |
|------|-------------|
| `--prompt FILE` | Custom classification prompt template |
| `-o FILE` | Output batch plan file (default: `batches.json`) |
| `--apply FILE` | Apply an existing batch plan |
| `--upload` | Upload each batch (requires `--apply`) |
| `--min-batch-size N` | Minimum traces per batch (default: 10) |
| `--max-batch-size N` | Maximum traces per batch (default: 30) |

### Agent setup

```bash
# Print CLI instructions and install pipeline skills
kayba setup

# Append to a project agent file
kayba setup --append-to AGENTS.md

# Skip skill installation
kayba setup --no-skills

# Install into a different project
kayba setup --project-dir /path/to/project
```

Options:

| Flag | Description |
|------|-------------|
| `--append-to FILE` | Append instructions to file instead of printing (recommended: `AGENTS.md`) |
| `--skills/--no-skills` | Install Claude Code pipeline skills (default: enabled) |
| `--project-dir DIR` | Project root directory (default: current directory) |

By default `kayba setup` copies the **kayba-pipeline** skill into `.claude/skills/`. This skill orchestrates a 7-stage evaluation pipeline (analyze traces → compute metrics → build rubric → plan fixes → HITL review → apply fixes → verify). See [Claude Code](claude-code.md#pipeline-skill) for details.

## End-to-end workflows

### Interactive (human at terminal)

```bash
# 1. Upload traces
kayba traces upload traces/

# 2. Run the pipeline (interactive trace selector)
kayba run

# 3. Review insights
kayba insights list --status pending
kayba insights triage --accept-all

# 4. Generate a prompt
kayba prompts generate -o prompt.md

# 5. Install it into your agent
kayba prompts install --target claude-code
```

### Programmatic (agent or script)

```bash
# 1. Upload traces
kayba traces upload traces/

# 2. List what was uploaded
TRACES=$(kayba traces list --json | jq -r '.[].id')

# 3. Run the pipeline on all traces
JOB_ID=$(kayba run --all --json | jq -r '.jobId')

# 4. Wait for completion
kayba status $JOB_ID --wait

# 5. Accept all insights and generate prompt
kayba insights triage --accept-all
kayba prompts generate -o prompt.md

# 6. Install the latest prompt into your agent
kayba prompts install --target codex
```

## Python client

The `KaybaClient` class can be used directly in Python code:

```python
from ace.cli.client import KaybaClient

client = KaybaClient(api_key="your-key")

# Trace management
traces = client.list_traces()
trace = client.get_trace("conv-123")
client.delete_trace("conv-123")
result = client.upload_traces([
    {"filename": "trace.md", "content": "...", "fileType": "md"},
])

# Run pipeline
job = client.generate_insights(
    trace_ids=["conv-123", "conv-456"],
    model="claude-sonnet-4-6",
)

# Check status
status = client.get_job(job["jobId"])

# List and triage
insights = client.list_insights(status="pending")
client.triage_insight(insights["insights"][0]["id"], "accepted")

# Generate and pull prompts
client.generate_prompt()
prompts = client.list_prompts()
prompt = client.get_prompt(prompts[0]["id"])

# Integrations
integrations = client.get_integrations()
client.update_integration("langsmith", {
    "enabled": True,
    "apiUrl": "https://eu.api.smith.langchain.com",
    "apiKey": "lsv2_pt_...",
})
result = client.test_integration("langsmith")
```

## API endpoints

| Method | Path | Client method |
|--------|------|---------------|
| `GET` | `/traces` | `list_traces()` |
| `POST` | `/traces` | `upload_traces()` |
| `GET` | `/traces/:id` | `get_trace()` |
| `DELETE` | `/traces/:id` | `delete_trace()` |
| `POST` | `/traces/batch` | `get_traces()` |
| `POST` | `/insights/generate` | `generate_insights()` |
| `GET` | `/insights` | `list_insights()` |
| `PATCH` | `/insights/:id` | `triage_insight()` |
| `GET` | `/jobs/:id` | `get_job()` |
| `POST` | `/jobs/:id` | `materialize_job()` |
| `POST` | `/prompts/generate` | `generate_prompt()` |
| `GET` | `/prompts` | `list_prompts()` |
| `GET` | `/prompts/:id` | `get_prompt()` |
| `GET` | `/integrations` | `get_integrations()` |
| `PUT` | `/integrations/:name` | `update_integration()` |
| `POST` | `/integrations/:name/test` | `test_integration()` |

## Coding agent setup

**Quick (current session):** Tell your coding agent to run `kayba setup`. The agent will see
the full CLI reference in its context and know how to use every command. The pipeline skill is
also installed to `.claude/skills/`, giving Claude Code access to the 7-stage evaluation pipeline.

**Persistent (all future sessions):** Append instructions to your project's agent file:

```bash
kayba setup --append-to AGENTS.md      # universal (Claude Code, Cursor, Copilot, Windsurf, etc.)
kayba setup --append-to CLAUDE.md      # Claude Code only
kayba setup --append-to .cursorrules   # Cursor only
```

`AGENTS.md` is the recommended target — it's the universal standard supported by 20+ coding agents.

To skip skill installation (e.g. for non-Claude-Code agents), pass `--no-skills`.

This setup step is separate from prompt installation. Once you have accepted
insights and generated a prompt, use `kayba prompts install` to update
`AGENTS.md`, `CLAUDE.md`, or `.cursorrules` with the generated prompt content.

## Environment variables

| Variable | Description |
|----------|-------------|
| `KAYBA_API_KEY` | API key (required) |
| `KAYBA_API_URL` | Base URL (default: `https://use.kayba.ai/api`) |
| `ANTHROPIC_API_KEY` | Passed to server for LLM calls via `--anthropic-key` |

## Using your generated prompt

`kayba prompts generate -o prompt.md` writes a Markdown prompt block built from
your accepted insights. It does **not** update `CLAUDE.md`, `AGENTS.md`, or
`.cursorrules` automatically; you choose where to apply it.

Use the built-in installer to update the right file without duplicating old
blocks:

```bash
# Install the latest prompt from Kayba into Claude Code
kayba prompts install --target claude-code

# Install the latest prompt into a universal agent file
kayba prompts install --target universal

# Install a local export into Cursor
kayba prompts install --input prompt.md --target cursor
```

The installer manages a dedicated Kayba block, so re-running it replaces the
previous prompt instead of appending duplicates.
