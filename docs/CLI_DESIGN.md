# CLI & Provider Configuration Design

Design document for the `ace` CLI, model configuration system, and provider registry.

---

## Implementation Status

| Component | Status | Location |
|---|---|---|
| `ModelConfig` / `ACEModelConfig` dataclasses | Done | `ace_next/providers/config.py` |
| TOML persistence (`ace.toml`) | Done | `ace_next/providers/config.py` |
| `.env` persistence (secrets) | Done | `ace_next/providers/config.py` |
| Provider registry (LiteLLM delegation) | Done | `ace_next/providers/registry.py` |
| Model search / discovery | Done | `ace_next/providers/registry.py` |
| Connection validation | Done | `ace_next/providers/registry.py` |
| `ace setup` interactive wizard | Done | `ace_next/cli/setup.py` |
| `ace models` search command | Done | `ace_next/cli/setup.py` |
| `ace validate` connection test | Done | `ace_next/cli/setup.py` |
| Lazy imports (fast CLI startup) | Done | `ace_next/__init__.py`, `ace_next/providers/__init__.py`, `ace_next/providers/registry.py` |

---

## Goals

1. **Zero-friction onboarding** — `ace setup` walks the user from nothing to a working config in under a minute.
2. **Secrets / config separation** — `ace.toml` (committable, no secrets) + `.env` (gitignored API keys). Teams share model choices without leaking credentials.
3. **Per-role model selection** — different models for Agent, Reflector, and Skill Manager to optimise cost vs quality.
4. **Provider agnosticism** — any model string LiteLLM supports works. No provider-specific code in the config layer.
5. **Fast CLI startup** — heavy dependencies (LiteLLM, instructor, openai) are lazily imported. The `ace` command starts in ~50ms, not ~2s.

---

## Architecture Overview

```
pyproject.toml
  [project.scripts]
  ace = "ace_next.cli.setup:main"        # entry point

ace_next/cli/
  __init__.py
  setup.py          # CLI commands + interactive wizard

ace_next/providers/
  __init__.py        # lazy re-exports
  config.py          # ModelConfig, ACEModelConfig, TOML + .env I/O
  registry.py        # provider detection, model search, connection validation
  litellm.py         # LiteLLMClient (runtime LLM calls — not CLI)
  instructor.py      # InstructorClient (structured outputs — not CLI)
```

The CLI layer (`ace_next/cli/`) depends on:
- `ace_next/providers/config.py` — always (lightweight, no heavy deps)
- `ace_next/providers/registry.py` — only when validating or searching (imports LiteLLM lazily)

The config layer has **zero heavy dependencies** — it uses only `tomllib` (stdlib), `pathlib`, and `dataclasses`.

---

## Configuration Model

### Two-file split

| File | Contains | Git | Written by |
|------|----------|-----|------------|
| `ace.toml` | Model names, temperature, max_tokens, extra_params | Commit | `save_config()` |
| `.env` | `OPENAI_API_KEY=sk-...`, `ANTHROPIC_API_KEY=sk-ant-...` | Gitignore | `save_env_var()` |

### `ace.toml` format

```toml
[default]
model = "gpt-4o-mini"

[agent]
model = "claude-sonnet-4-20250514"
max_tokens = 4096

[reflector]
model = "gpt-4o-mini"
temperature = 0.2
```

Roles without an explicit section inherit from `[default]`. Only non-default values are written (e.g. `temperature` is omitted when it equals `0.0`).

### Config discovery

`find_config(start)` walks up from `start` to the filesystem root looking for `ace.toml`. This supports monorepos where the config lives at the project root but commands run from subdirectories.

### Loading in code

```python
from ace_next import ACELiteLLM

# Option 1: Load from ace.toml + .env (created by `ace setup`)
ace = ACELiteLLM.from_setup()

# Option 2: Explicit model, keys from environment
ace = ACELiteLLM.from_model("gpt-4o-mini")

# Option 3: Full config object
from ace_next import ACEModelConfig, ModelConfig
ace = ACELiteLLM.from_config(ACEModelConfig(
    default=ModelConfig(model="gpt-4o-mini"),
    agent=ModelConfig(model="claude-sonnet-4-20250514"),
))
```

---

## Data Types

### ModelConfig

```python
@dataclass
class ModelConfig:
    model: str                              # LiteLLM model string
    temperature: float = 0.0
    max_tokens: int = 2048
    extra_params: dict[str, Any] | None = None
```

Serialises to/from a TOML section. `to_dict()` omits default values to keep the file clean.

### ACEModelConfig

```python
@dataclass
class ACEModelConfig:
    default: ModelConfig                    # required — used as fallback
    agent: ModelConfig | None = None        # overrides default for Agent role
    reflector: ModelConfig | None = None    # overrides default for Reflector role
    skill_manager: ModelConfig | None = None  # overrides default for Skill Manager role
```

`for_role(role)` returns the role-specific config or falls back to `default`.

### ValidationResult

```python
@dataclass
class ValidationResult:
    success: bool
    model: str = ""
    provider: str = ""
    latency_ms: int = 0
    error: str = ""
```

Returned by `validate_connection()`. On success, includes the provider name and round-trip latency. On failure, includes a human-readable error string.

### ModelInfo

```python
@dataclass
class ModelInfo:
    model: str
    provider: str
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    input_cost_per_m: float | None = None   # cost per million tokens
    output_cost_per_m: float | None = None
    key_found: bool = False                 # are the required env vars set?
```

Returned by `search_models()`. Pricing is per million tokens (converted from LiteLLM's per-token values).

---

## Provider Registry

All provider logic is delegated to LiteLLM. The registry module (`ace_next/providers/registry.py`) wraps LiteLLM with a stable API:

| Function | What it does | Makes API calls? |
|----------|-------------|-----------------|
| `get_provider(model)` | Detect provider from model string | No |
| `get_missing_keys(model)` | List required env vars that are unset | No |
| `keys_are_set(model)` | Check if all required env vars exist | No |
| `validate_connection(model, api_key?)` | Send a 3-token test call | Yes (tiny) |
| `search_models(query, provider?, limit?)` | Search LiteLLM's static model registry | No |
| `suggest_models(typo, limit?)` | Fuzzy-match model names for typo correction | No |

### LiteLLM lazy import

LiteLLM takes ~1.5s to import. The registry defers import until first use:

```python
def _litellm():
    global _litellm_mod
    try:
        return _litellm_mod
    except NameError:
        pass
    import litellm as _mod
    _litellm_mod = _mod
    return _mod
```

All registry functions call `_litellm()` instead of using a top-level import.

### Provider key mapping

`PROVIDER_KEY_ENV` maps provider names to the environment variables they require:

| Provider | Required env vars |
|----------|------------------|
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `azure` | `AZURE_API_KEY` |
| `gemini` | `GEMINI_API_KEY` |
| `deepseek` | `DEEPSEEK_API_KEY` |
| `groq` | `GROQ_API_KEY` |
| `bedrock` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_REGION_NAME` |
| `bedrock_converse` | `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_REGION_NAME` |
| `vertex_ai` | `GOOGLE_APPLICATION_CREDENTIALS` |
| `cohere` | `COHERE_API_KEY` |
| `mistral` | `MISTRAL_API_KEY` |
| `openrouter` | `OPENROUTER_API_KEY` |
| `together_ai` | `TOGETHERAI_API_KEY` |
| `fireworks_ai` | `FIREWORKS_AI_API_KEY` |
| `replicate` | `REPLICATE_API_KEY` |
| `huggingface` | `HUGGINGFACE_API_KEY` |
| `perplexity` | `PERPLEXITYAI_API_KEY` |
| `anyscale` | `ANYSCALE_API_KEY` |

For multi-key providers (bedrock), **all** listed vars must be set for `_quick_key_check` to report `True`.

This mapping is preferred over LiteLLM's `validate_environment()` because:
- LiteLLM's response can be inaccurate for certain providers (e.g. bedrock_converse)
- Our mapping is used for both the `Key` column in `ace models` and the key prompting in `ace setup`

---

## CLI Commands

### Entry point

Defined in `pyproject.toml`:

```toml
[project.scripts]
ace = "ace_next.cli.setup:main"
```

`main()` uses `argparse` with subcommands:

```
ace setup [--dir DIR]        Interactive configuration wizard
ace models [QUERY] [--provider P] [--limit N]   Search model catalog
ace validate MODEL           Test a model connection
ace config                   Show current configuration
```

### `ace setup`

Interactive wizard flow:

```
1. Load existing .env (if present)
2. Check for existing ace.toml
   - If found: show current config, ask "Reconfigure?"
   - If declined: return existing config
3. Step 1: Choose your model
   - Prompt for model name
   - Detect provider via get_provider()
   - Validate connection
     - If auth fails: prompt for missing keys, retry
     - If model not found: suggest alternatives, re-prompt
     - If success: continue
4. Step 2: Role assignment
   - Ask "Use this model for all roles?" (default: yes)
   - If no: prompt for each role (Agent, Reflector, Skill Manager)
     - Enter = keep default (skip validation)
     - Different model = validate it
5. Save ace.toml and .env
6. Print configuration summary
```

Key behaviours:
- **Validation-first**: the wizard tries the connection immediately. If credentials exist in the environment (env vars, `.env`, AWS profiles), no prompting needed.
- **Error recovery**: on failed validation, env vars set during prompting are rolled back.
- **Secret handling**: API keys are prompted via `getpass` (hidden input). Non-secret values like `AWS_REGION_NAME` and `GOOGLE_APPLICATION_CREDENTIALS` use regular `input()` so users can see what they type.
- **Typo correction**: when a model is not found, `suggest_models()` offers alternatives.

### `ace models`

Searches LiteLLM's static `model_cost` registry (no API calls):

```
$ ace models claude haiku

Model                                         Provider        Input $/M  Output $/M  Key
------------------------------------------------------------------------------------------
claude-haiku-4-5-20251001                     anthropic        $1.00      $5.00       ✓
us.anthropic.claude-haiku-4-5-20251001-v1:0   bedrock_converse $1.10      $5.50       ✗
```

- Multiple query terms are AND-matched (all must appear in the model name)
- `--provider` filters by LiteLLM provider name
- `--limit` caps results (default 20), shows total count
- `Key` column uses `_quick_key_check()` — checks env vars only, no API calls

### `ace validate`

Sends a minimal LLM call (3 tokens: "Say 'ok'") to verify:
- API key authentication
- Model availability at the provider
- Network connectivity

```
$ ace validate gpt-4o-mini
✓ Connected! (gpt-4o-mini via openai, 203ms)
```

On failure, suggests similar model names if the model wasn't found.

All subcommands (`models`, `validate`, `config`) use `_load_project_dotenv()` which finds `.env` relative to `ace.toml` (via `find_config()`), not just CWD.

### `ace config`

Displays the current configuration from `ace.toml`:

```
$ ace config
Configuration (/path/to/ace.toml)

  Role             Model
  ---------------- ---------------------------------------------
  default          gpt-4o-mini
  agent            claude-sonnet-4-20250514
  reflector        (default)
  skill_manager    (default)
```

Uses `find_config()` to locate `ace.toml` from the current directory upward.

### `ace models` (no query)

When called without arguments, shows usage examples instead of dumping arbitrary results:

```
$ ace models
Usage: ace models <query>

Examples:
  ace models claude          All Claude models
  ace models gpt 4o          GPT-4o variants
  ace models haiku us        US-region Haiku models
  ace models --provider openai  All OpenAI models
```

---

## Lazy Import Strategy

### Problem

`import ace_next` eagerly imported all submodules, including `litellm` (~1.5s). This made the CLI unusable (~2s startup for a simple `ace --help`).

### Solution

Three-layer lazy import:

1. **`ace_next/__init__.py`** — `__getattr__`-based lazy loading. `TYPE_CHECKING` block for IDE support, `_LAZY_IMPORTS` dict for runtime.

2. **`ace_next/providers/__init__.py`** — same pattern. Config imports are eager (lightweight), everything else is lazy.

3. **`ace_next/providers/registry.py`** — `_litellm()` helper defers `import litellm` until first function call.

Result: `ace --help` runs in ~50ms. LiteLLM is only imported when the user actually searches, validates, or sets up.

### Pattern

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .heavy_module import HeavyClass  # IDE autocomplete

_LAZY_IMPORTS = {
    "HeavyClass": ("package.heavy_module", "HeavyClass"),
}

def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value  # cache for subsequent access
        return value
    raise AttributeError(...)
```

---

## File Map

```
ace_next/
  __init__.py                     # lazy re-exports for all ace_next symbols
  cli/
    __init__.py                   # "ACE CLI — setup and management commands."
    setup.py                      # main(), run_setup(), _cmd_models(), _cmd_validate(), _cmd_config()
  providers/
    __init__.py                   # lazy re-exports (config eager, rest lazy)
    config.py                     # ModelConfig, ACEModelConfig, TOML/env I/O
    registry.py                   # provider detection, model search, validation
    litellm.py                    # LiteLLMClient (runtime, not CLI)
    instructor.py                 # InstructorClient (runtime, not CLI)
    langchain.py                  # LangChainLiteLLMClient (optional)
    claude_code.py                # ClaudeCodeLLMClient (optional)
```

Generated files:

```
project-root/
  ace.toml                        # model config (commit this)
  .env                            # API keys (gitignore this)
```

---

## Known Issues

### Resolved

| # | Issue | Fix |
|---|-------|-----|
| 1 | `_prompt_secret` used `getpass` for non-secrets like `AWS_REGION_NAME` | Non-secret vars (`AWS_REGION_NAME`, `GOOGLE_APPLICATION_CREDENTIALS`) now use visible `_prompt()` |
| 2 | Error classification used fragile substring matching | Simplified: only "not found" is non-recoverable; everything else offers key prompting |
| 3 | `save_env_var` didn't quote values | Values now written as `KEY="value"` |
| 4 | `_PROVIDER_KEY_ENV` was private but imported externally | Renamed to `PROVIDER_KEY_ENV` (public) |
| 5 | `v` / `x` status symbols | Replaced with `✓` / `✗` |
| 7 | `ace models` with no query dumped arbitrary results | Now shows usage examples instead |
| 8 | No way to inspect current config | Added `ace config` command |
| 10 | `ace validate` / `ace models` only loaded `.env` from CWD | Added `_load_project_dotenv()` — finds `.env` relative to `ace.toml` via `find_config()` |
| 11 | Unused imports `asdict`, `field` in `config.py` | Removed |
| 12 | Dead function `_litellm_available()` in `registry.py` | Removed |
| 13 | Unused `PROVIDER_MODEL_EXAMPLES` import in `setup.py` | Removed |
| 14 | Duplicate `search_models` import in `setup.py` | Consolidated to single top-level import |

### Open

| # | Issue | Impact |
|---|-------|--------|
| 6 | No `--non-interactive` mode for CI/Docker | Blocks CI automation — requires a future `ace setup --model MODEL --skip-validation` flag |
| 9 | Per-role model selection skips validation when keeping default | Low — the default was already validated in Step 1 |
| 15 | No multi-provider key setup in one pass | When reconfiguring (`ace setup` on an existing config), users who want different providers per role (e.g. OpenAI default + Anthropic agent + Bedrock reflector) must go through each role sequentially — keys are only prompted when a model fails validation. A future improvement should let users configure all providers and their keys upfront in a single credentials step, before role assignment begins. |

---

## Design Decisions

### Why hand-rolled TOML serialisation?

Python's `tomllib` (stdlib since 3.11) only reads TOML. Writing requires `tomli-w` (third-party) or a manual serialiser. To avoid adding a dependency for a simple four-section config file, we hand-roll `_to_toml()`. The format is simple enough that this is reliable — the only complex case is `extra_params` (inline table).

### Why our own key mapping instead of LiteLLM's?

LiteLLM's `validate_environment()` sometimes returns incorrect keys (e.g. suggesting `AWS_BEARER_TOKEN_BEDROCK` for bedrock_converse when the standard auth is `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` + `AWS_REGION_NAME`). Our `PROVIDER_KEY_ENV` mapping is simpler, auditable, and covers the common providers. For unknown providers, we fall back to LiteLLM's response or guess `{PROVIDER}_API_KEY`.

### Why validation-first in setup?

Many users already have credentials in their environment (exported vars, AWS profiles, `.env` from another project). Prompting for keys before trying would waste their time. By attempting the connection first, the happy path is: type model name → instant success → done.

### Why `ace.toml` instead of `pyproject.toml [tool.ace]`?

- Keeps ACE config decoupled from the Python project (ACE might be used in non-Python contexts)
- `find_config()` can walk up the directory tree independently
- Easier to reason about — one file, one purpose
