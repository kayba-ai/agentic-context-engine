# Installation

## For Users

=== "pip"

    ```bash
    pip install ace-framework
    ```

=== "With extras"

    ```bash
    pip install ace-framework[all]            # All optional features
    pip install ace-framework[instructor]     # Structured outputs (Instructor)
    pip install ace-framework[langchain]      # LangChain integration
    pip install ace-framework[browser-use]    # Browser automation
    pip install ace-framework[claude-code]    # Claude Code CLI integration
    pip install ace-framework[observability]  # Opik monitoring + cost tracking
    pip install ace-framework[deduplication]  # Skill deduplication (embeddings)
    pip install ace-framework[transformers]   # Local model support
    ```

## For Contributors

=== "UV (Recommended)"

    ```bash
    git clone https://github.com/kayba-ai/agentic-context-engine
    cd agentic-context-engine
    uv sync  # Installs everything (10-100x faster than pip)
    ```

=== "pip"

    ```bash
    git clone https://github.com/kayba-ai/agentic-context-engine
    cd agentic-context-engine
    pip install -e .
    ```

## Requirements

- **Python 3.12**
- An API key for your LLM provider

## Configure Your LLM

The recommended way to set up your API keys and model selection:

```bash
ace setup
```

This interactive wizard validates your API key and model, then saves config to `ace.toml` (model names, safe to commit) and `.env` (API keys, gitignored). See [Setup](setup.md) for full details.

### Manual alternative

If you prefer not to use the wizard, set environment variables directly:

```bash
export OPENAI_API_KEY="sk-..."
```

Or create a `.env` file (add to `.gitignore`):

```bash
OPENAI_API_KEY=sk-...
```

## Verify Installation

```python
from ace_next import ACELiteLLM

# Uses ace.toml + .env from `ace setup`
agent = ACELiteLLM.from_setup()
print(agent.ask("Hello!"))
```

Or without `ace setup`:

```python
agent = ACELiteLLM.from_model("gpt-4o-mini")
print(agent.ask("Hello!"))
```

## Set Up Coding Agent Skills (Optional)

If you use Claude Code, install the Kayba pipeline skill:

```bash
kayba setup
```

This installs the evaluation pipeline skill to `.claude/skills/` and prints CLI instructions. See [Hosted API](../integrations/hosted-api.md) for details.

## What to Read Next

- [Setup](setup.md) — configure models and API keys
- [Quick Start](quick-start.md) — build your first self-learning agent
- [How ACE Works](../concepts/overview.md) — understand the architecture
