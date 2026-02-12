# ACE Agent for AppWorld

This directory contains the ACE agent implementation for the [AppWorld](https://appworld.dev/) benchmark.

## Quick Start

```bash
# 1. Download AppWorld data
pip install appworld
appworld download data

# 2. Start AppWorld Docker servers
docker run -d --name appworld-env -p 8000:8000 \
  -v ~/.appworld/data:/appworld/data -w /appworld \
  ghcr.io/stonybrooknlp/appworld environment --port 8000

docker run -d --name appworld-apis -p 9000:9000 \
  -v ~/.appworld/data:/appworld/data -w /appworld \
  ghcr.io/stonybrooknlp/appworld apis --port 9000

# 3. Run the test
cd benchmarks
uv run python test_appworld_ace.py
```

## Prerequisites

- Docker
- Python 3.11+
- An LLM API key (OpenAI or Anthropic)

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Setup

### 1. Install AppWorld and Download Data

```bash
pip install appworld
appworld download data
```

This downloads ~2GB of task data to `~/.appworld/data/`.

### 2. Start AppWorld Docker Servers

AppWorld requires two servers running in Docker:

**Environment Server** (port 8000) - Handles task initialization, code execution, and evaluation:
```bash
docker run -d --name appworld-env -p 8000:8000 \
  -v ~/.appworld/data:/appworld/data -w /appworld \
  ghcr.io/stonybrooknlp/appworld environment --port 8000
```

**API Server** (port 9000) - Provides simulated app APIs (Venmo, Spotify, Gmail, etc.):
```bash
docker run -d --name appworld-apis -p 9000:9000 \
  -v ~/.appworld/data:/appworld/data -w /appworld \
  ghcr.io/stonybrooknlp/appworld apis --port 9000
```

### 3. Verify Servers

```bash
# Test connectivity
curl http://localhost:8000/
# Expected: {"message":"Welcome to AppWorld Server!..."}

curl http://localhost:9000/
# Expected: {"title":"AppWorld API Server"...}
```

Or use the built-in test:
```bash
uv run python agents/ace_agent/main.py
```

## Running ACE on AppWorld

### Direct Test (Recommended for Development)

```bash
cd benchmarks
uv run python test_appworld_ace.py
```

This runs ACE on 2 AppWorld tasks using Claude (if `ANTHROPIC_API_KEY` is set) or GPT-4o-mini.

### With HAL Harness (Full Evaluation)

For complete benchmark evaluation with metrics:

```bash
# Install HAL
git clone --recursive https://github.com/princeton-pli/hal-harness.git
cd hal-harness
pip install -e .

# Copy ACE agent
cp -r ../agents/ace_agent agents/

# Run evaluation
hal-eval --benchmark appworld_test_normal \
    --agent_dir agents/ace_agent \
    --agent_function main.run \
    -A model=gpt-4o \
    --limit 10
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `ACE_MODEL` | LLM model to use | `gpt-4o-mini` or `claude-3-5-haiku-latest` |
| `APPWORLD_ENV_URL` | Environment server URL | `http://localhost:8000` |
| `APPWORLD_API_URL` | API server URL | `http://localhost:9000` |

### Agent Parameters

When using HAL, pass via `-A` flags:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | LLM model name | `gpt-4o-mini` |
| `skillbook_path` | Path to skillbook JSON | None |
| `learning_enabled` | Enable ACE learning | `false` |
| `max_interactions` | Max execution steps | `30` |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ACE Agent                              │
│  ┌─────────┐  ┌───────────┐  ┌──────────────┐              │
│  │ Agent   │  │ Reflector │  │ SkillManager │              │
│  └────┬────┘  └─────┬─────┘  └──────┬───────┘              │
│       │             │               │                       │
│       └─────────────┴───────────────┘                       │
│                     │                                       │
│              ┌──────┴──────┐                                │
│              │  Skillbook  │                                │
│              └─────────────┘                                │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP
        ┌─────────────┴─────────────┐
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│ Environment   │           │ API Server    │
│ Server :8000  │           │ :9000         │
│               │           │               │
│ - /initialize │           │ - /venmo/*    │
│ - /execute    │           │ - /spotify/*  │
│ - /evaluate   │           │ - /gmail/*    │
└───────────────┘           └───────────────┘
        │                           │
        └───────────┬───────────────┘
                    ▼
            ┌───────────────┐
            │ AppWorld Data │
            │ ~/.appworld/  │
            └───────────────┘
```

## Troubleshooting

### "Task directory doesn't exist"
Data not mounted correctly. Ensure `-v ~/.appworld/data:/appworld/data -w /appworld` is set.

### "404 Not Found" on /initialize
Wrong server type. Use `environment --port 8000` (not `apis`).

### Platform mismatch warning
The AppWorld image is amd64. On ARM Macs, Docker runs it via emulation (slower but works).

### Stop and restart servers
```bash
docker rm -f appworld-env appworld-apis
# Then run the docker commands again
```

## AppWorld Task Format

Tasks are interactive coding challenges. Example:

```
Task: "Reset friends on venmo to be the same as my friends in my phone."

The agent must:
1. Query phone contacts API
2. Query current Venmo friends
3. Add missing friends
4. Remove extra friends
```

The ACE agent generates Python code that calls the AppWorld APIs to complete these tasks.

## Available Benchmarks

| Benchmark | Tasks | Description |
|-----------|-------|-------------|
| `test_normal` | 168 | Standard difficulty |
| `test_challenge` | 417 | Harder tasks |
| `dev` | 57 | Development set |
| `train` | 90 | Training set |
