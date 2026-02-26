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

## API Key Setup

Set one of these environment variables depending on your provider:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="AIza..."
```

Or create a `.env` file:

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

```python
from dotenv import load_dotenv
load_dotenv()  # Loads from .env
```

## Supported Providers

ACE uses [LiteLLM](https://docs.litellm.ai/) for model access, supporting 100+ providers:

| Provider | Model Example | Env Variable |
|----------|--------------|--------------|
| OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-sonnet-4-5-20250929` | `ANTHROPIC_API_KEY` |
| Google | `gemini-pro` | `GOOGLE_API_KEY` |
| Ollama (local) | `ollama/llama2` | — |
| AWS Bedrock | `bedrock/anthropic.claude-v2` | AWS credentials |
| Azure | `azure/gpt-4` | `AZURE_API_KEY` |

## Verify Installation

```python
from ace_next import ACELiteLLM

agent = ACELiteLLM.from_model("gpt-4o-mini")
print(agent.ask("Hello!"))
```

## What to Read Next

- [Quick Start](quick-start.md) — build your first self-learning agent
- [How ACE Works](../concepts/overview.md) — understand the architecture
