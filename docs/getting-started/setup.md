# Setup

Configure your LLM provider and model selection for ACE.

## Guided Setup (Recommended)

The `ace setup` command walks you through configuration interactively — it validates the connection first, and only asks for credentials if needed.

```bash
ace setup
```

```
ACE Setup

Step 1: Choose your model

  Examples: gpt-4o-mini, claude-sonnet-4-20250514, ollama/llama2
  Search models: ace models <query>

  Default model: gpt-4o-mini
  v Connected! (gpt-4o-mini via openai, 203ms)
    Using OPENAI_API_KEY

Step 2: Role assignment

  ACE uses three roles. You can assign a different model to each,
  or use the same model for all (recommended to start).

  Use this model for all roles? [Y/n]: n

  Agent (executes tasks) [gpt-4o-mini]: claude-sonnet-4-20250514
  ! No credentials found for anthropic
  ANTHROPIC_API_KEY: sk-ant-...
  v Connected! (claude-sonnet-4-20250514 via anthropic, 347ms)
  v Saved credentials to .env

  Reflector (analyses results) [gpt-4o-mini]:
  Skill Manager (updates skillbook) [gpt-4o-mini]:

v Saved model config to ace.toml

  Configuration summary:
    default:        gpt-4o-mini
    agent:          claude-sonnet-4-20250514
```

The wizard tries the connection immediately — if your credentials are already in the environment (via `.env`, exported variables, or cloud auth like AWS), it just works. It only prompts for keys when the connection actually fails.

This creates two files:

| File | Contains | Commit to git? |
|------|----------|----------------|
| `.env` | API keys only | No (gitignore it) |
| `ace.toml` | Model names per role | Yes (no secrets) |

Then in your code:

```python
from ace_next import ACELiteLLM

ace = ACELiteLLM.from_setup()
answer = ace.ask("What is 2+2?")
```

## Manual Setup

If you prefer not to use the CLI, set environment variables directly.

### 1. Set API keys

=== "Shell"

    ```bash
    export OPENAI_API_KEY="sk-..."
    export ANTHROPIC_API_KEY="sk-ant-..."
    ```

=== ".env file"

    ```bash
    # .env (add to .gitignore)
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    ```

### 2. Use in code

```python
from ace_next import ACELiteLLM

# Single model for all roles
ace = ACELiteLLM.from_model("gpt-4o-mini")
```

```python
from ace_next import ACELiteLLM, ACEModelConfig, ModelConfig

# Different models per role
ace = ACELiteLLM.from_config(ACEModelConfig(
    default=ModelConfig(model="gpt-4o-mini"),
    agent=ModelConfig(model="claude-sonnet-4-20250514"),
))
```

## Per-Role Model Selection

ACE has three roles, each making LLM calls. You can assign different models to optimise cost vs quality:

| Role | What it does | Recommendation |
|------|-------------|----------------|
| **Agent** | Executes tasks, produces answers | Strong reasoning model |
| **Reflector** | Analyses results, extracts lessons | Good analysis, lower cost OK |
| **Skill Manager** | Updates the skillbook | Structured output reliability |

Example `ace.toml`:

```toml
[default]
model = "gpt-4o-mini"

[agent]
model = "claude-sonnet-4-20250514"
max_tokens = 4096

[reflector]
model = "gpt-4o-mini"
```

Roles without an explicit section use `[default]`.

## Discovering Models

### Search available models

Use multiple terms to narrow results — all terms must match:

```bash
ace models claude             # All Claude models
ace models haiku us           # Only US-region Haiku models
ace models gpt 4o             # GPT-4o variants
ace models --provider openai  # All OpenAI models
```

Output shows model name, provider, pricing, and whether your API key is configured:

```
Model                                         Provider        Input $/M  Output $/M  Key
------------------------------------------------------------------------------------------
us.anthropic.claude-haiku-4-5-20251001-v1:0   bedrock_converse $1.10      $5.50       v
claude-haiku-4-5-20251001                     anthropic        $1.00      $5.00       x

Showing 20 of 40 models. Narrow your search: ace models <query> or use --limit 40
```

### Validate a specific model

```bash
ace validate us.anthropic.claude-haiku-4-5-20251001-v1:0
```

Makes a tiny test call (3 tokens) to confirm the key, model, and network all work.

## Supported Providers

ACE uses [LiteLLM](https://docs.litellm.ai/) for model access. Any model string LiteLLM supports will work:

| Provider | Model Example | Env Variable |
|----------|--------------|--------------|
| OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| AWS Bedrock | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_REGION_NAME` |
| Google Gemini | `gemini/gemini-2.0-flash` | `GEMINI_API_KEY` |
| DeepSeek | `deepseek/deepseek-chat` | `DEEPSEEK_API_KEY` |
| Groq | `groq/llama-3.1-70b` | `GROQ_API_KEY` |
| Ollama (local) | `ollama/llama2` | --- |
| Azure OpenAI | `azure/gpt-4` | `AZURE_API_KEY` |
| OpenRouter | `openrouter/anthropic/claude-3.5-sonnet` | `OPENROUTER_API_KEY` |
| MiniMax | `openai/MiniMax-M2.5` | `MINIMAX_API_KEY` |

100+ providers supported. Run `ace models` to search the full catalog.

## Troubleshooting

### "No ace.toml found"

Run `ace setup` or use `ACELiteLLM.from_model("gpt-4o-mini")` instead of `from_setup()`.

### "Invalid API key"

```bash
# Re-validate
ace validate gpt-4o-mini

# Re-run setup to fix
ace setup
```

### "Model not found"

The model string may have a typo. `ace validate` and `ace setup` suggest alternatives:

```bash
ace validate claud-sonnet
# x Model 'claud-sonnet' not found at the provider.
# Did you mean:
#   - claude-sonnet-4-20250514
#   - claude-3-5-sonnet-20241022
```

### "Could not detect a provider"

Use the `provider/model-name` format:

```bash
# Instead of just "llama2":
ollama/llama2
groq/llama-3.1-70b
```

Search for the correct model string: `ace models llama`

## What to Read Next

- [Quick Start](quick-start.md) --- build your first self-learning agent
- [How ACE Works](../concepts/overview.md) --- understand the three-role architecture
- [Integrations](../integrations/index.md) --- LangChain, Browser-Use, Claude Code
