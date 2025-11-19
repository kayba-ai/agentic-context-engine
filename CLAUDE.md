# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an implementation scaffold for reproducing the Agentic Context Engineering (ACE) method from the paper "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models" (arXiv:2510.04618).

The framework enables AI agents to learn from their execution feedback through three collaborative roles: Generator (produces answers), Reflector (analyzes performance), and Curator (updates the knowledge base called a "playbook").

## Development Commands

### Package Installation
```bash
# Install from PyPI (end users)
pip install ace-framework

# Install with optional dependencies
pip install ace-framework[all]           # All optional features
pip install ace-framework[langchain]     # LangChain integration (enterprise)
pip install ace-framework[observability] # Opik monitoring and cost tracking
pip install ace-framework[transformers]  # Local model support

# Development installation (contributors) - UV Method (Recommended)
git clone https://github.com/kayba-ai/agentic-context-engine
cd agentic-context-engine
uv sync                                  # Install all dependencies (10-100x faster than pip)

# Development installation (contributors) - Traditional Method
git clone https://github.com/kayba-ai/agentic-context-engine
cd agentic-context-engine
pip install -e .
```

### Development with UV (Recommended for Contributors)
UV is a modern, ultra-fast Python package manager (10-100x faster than pip).

**Quick Start:**
```bash
# Clone and setup (one command!)
git clone https://github.com/kayba-ai/agentic-context-engine
cd agentic-context-engine
uv sync                          # Installs everything automatically

# Run any example or script
uv run python examples/simple_ace_example.py
uv run pytest                    # Run tests
```

**That's it!** UV handles virtual environments, dependencies, and Python versions automatically.

**Advanced UV Commands (optional):**
```bash
# Add/remove dependencies
uv add package-name              # Add new dependency
uv remove package-name           # Remove dependency

# Update dependencies
uv lock --upgrade                # Update all to latest versions

# CI/Production
uv sync --locked                 # Use exact versions from uv.lock
```

**Files managed by UV:**
- `pyproject.toml` - Project configuration (edit this for new deps)
- `uv.lock` - Locked versions (auto-generated, commit this)
- `.python-version` - Python 3.12 (UV installs if needed)

### Running Tests
```bash
# Run all tests
python -m unittest discover -s tests

# Run specific test file
python -m unittest tests.test_adaptation

# Run with verbose output
python -m unittest discover -s tests -v

# Using pytest (with dev dependencies - recommended)
uv run pytest                        # Run all tests
uv run pytest tests/test_adaptation.py  # Run specific test file
uv run pytest tests/test_integration.py  # Integration test suite (10 tests)
uv run pytest -v                     # Verbose output
```

**Test Coverage**:
- Unit tests for individual components (playbook, roles, adapters)
- Integration tests for end-to-end workflows (offline/online adaptation, checkpoints)
- Example scripts serve as functional tests

### Code Quality
```bash
# Format code with Black
uv run black ace/ tests/ examples/

# Type checking with MyPy
uv run mypy ace/

# Run all quality checks (when available in dev dependencies)
uv run black --check ace/ tests/ examples/
uv run mypy ace/

# Pre-commit hooks (auto-runs Black + MyPy on git commit)
uv run pre-commit install           # One-time setup
uv run pre-commit run --all-files   # Manual run
```

### Running Examples
```bash
# Quick start with LiteLLM (requires API key)
python examples/simple_ace_example.py

# Kayba Test demo (seahorse emoji challenge)
python examples/kayba_ace_test.py

# Advanced examples
python examples/quickstart_litellm.py
python examples/langchain_example.py
python examples/playbook_persistence.py

# Compare prompt versions
python examples/compare_v1_v2_prompts.py
python examples/advanced_prompts_v2.py

# Browser automation demos (contributors: install with `uv sync --group demos`)
uv run python examples/browser-use/baseline_domain_checker.py    # Baseline automation
uv run python examples/browser-use/ace_domain_checker.py         # ACE-enhanced automation
uv run python examples/browser-use/baseline_form_filler.py       # Baseline form filling
uv run python examples/browser-use/ace_form_filler.py            # ACE-enhanced form filling
```

### Development Scripts (Research Only)
```bash
# Note: These require local model weights and are not in PyPI package
CUDA_VISIBLE_DEVICES=2,3 python scripts/run_questions.py
CUDA_VISIBLE_DEVICES=2,3 python scripts/run_local_adapter.py
python scripts/run_questions_direct.py

# Benchmarking - Scientific evaluation framework
uv run python scripts/run_benchmark.py simple_qa --limit 50              # ACE with train/test split
uv run python scripts/run_benchmark.py simple_qa --limit 50 --compare    # Baseline vs ACE comparison
uv run python scripts/run_benchmark.py finer_ord --limit 100 --epochs 3  # Multi-epoch training
python scripts/compare_baseline_vs_ace.py                                # Analysis scripts
python scripts/analyze_ace_results.py
python scripts/explain_ace_performance.py
```

## Architecture

### Core Concepts
- **Playbook**: Structured context store containing bullets (strategy entries) with helpful/harmful counters
  - Uses TOON (Token-Oriented Object Notation) format for 16-62% token savings
  - `playbook.as_prompt()` returns TOON format (for LLM consumption)
  - `str(playbook)` returns markdown format (for human debugging)
- **Delta Operations**: Incremental updates to the playbook (ADD, UPDATE, TAG, REMOVE)
- **Three Agentic Roles** sharing the same base LLM:
  - **Generator**: Produces answers using the current playbook
  - **Reflector**: Analyzes errors and classifies bullet contributions
  - **Curator**: Emits delta operations to update the playbook
- **Two Architecture Patterns**:
  - **Full ACE Pipeline**: Generator+Reflector+Curator (for new agents)
  - **Integration Pattern**: Reflector+Curator only (for existing systems like browser-use, LangChain)

### Module Structure

**ace/** - Core library modules:
- `playbook.py`: Bullet and Playbook classes for context storage (TOON format)
- `delta.py`: DeltaOperation and DeltaBatch for incremental updates
- `roles.py`: Generator, Reflector, Curator implementations
- `adaptation.py`: OfflineAdapter and OnlineAdapter orchestration loops
- `llm.py`: LLMClient interface with DummyLLMClient and TransformersLLMClient
- `prompts.py`: Default prompt templates (v1.0 - simple, for tutorials)
- `prompts_v2_1.py`: State-of-the-art prompts with MCP enhancements (v2.1 - **RECOMMENDED**)
- `features.py`: Centralized optional dependency detection
- `llm_providers/`: Production LLM client implementations
  - `litellm_client.py`: LiteLLM integration (100+ model providers)
  - `langchain_client.py`: LangChain integration
- `integrations/`: Wrappers for external agentic frameworks (**key pattern**)
  - `base.py`: Base integration pattern and utilities
  - `browser_use.py`: ACEAgent - browser automation with learning
  - `langchain.py`: ACELangChain - wrap LangChain chains/agents
  - `litellm.py`: ACELiteLLM - simple conversational agent

**ace/observability/** - Production monitoring and observability:
- `opik_integration.py`: Enterprise-grade monitoring with Opik
- `tracers.py`: Automatic tracing decorators for all role interactions
- **Automatic token usage and cost tracking** for all LLM calls
- Real-time cost monitoring via Opik dashboard

**benchmarks/** - Benchmark framework:
- `base.py`: Base benchmark classes and interfaces
- `environments.py`: Task environment implementations
- `manager.py`: Benchmark execution and management
- `processors.py`: Data processing utilities
- `loaders/`: Dataset loaders for various benchmarks

**tests/** - Test suite using unittest framework

**examples/** - Production-ready example scripts:
- Basic usage examples with various LLM providers
- Browser automation demos comparing baseline vs ACE-enhanced approaches
- Advanced prompt engineering examples

**scripts/** - Research and development scripts (not in PyPI package)

### Key Implementation Patterns

1. **Full ACE Pipeline** (for new agents):
   - Sample → Generator (produces answer) → Environment (evaluates) → Reflector (analyzes) → Curator (updates playbook)
   - Offline: Multiple epochs over training samples
   - Online: Sequential processing of test samples
   - Use when: Building new agent from scratch, Q&A tasks, classification

2. **Integration Pattern** (for existing agents):
   - External agent executes task → Reflector analyzes → Curator updates playbook
   - No ACE Generator - external framework handles execution
   - Three steps: INJECT context (optional) → EXECUTE with external agent → LEARN from results
   - Use when: Wrapping browser-use, LangChain, CrewAI, or custom agents
   - See `ace/integrations/base.py` for detailed explanation

3. **LLM Integration**:
   - Implement `LLMClient` subclass for your model API
   - LiteLLMClient supports 100+ providers (OpenAI, Anthropic, Google, etc.)
   - LangChainClient provides LangChain integration
   - TransformersLLMClient for local model deployment
   - All roles share the same LLM instance

4. **Task Environment**:
   - Extend `TaskEnvironment` abstract class
   - Implement `evaluate()` to provide execution feedback
   - Return `EnvironmentResult` with feedback and optional ground truth

5. **Observability Integration**:
   - Automatic tracing with Opik when installed
   - Token usage and cost tracking for all LLM calls
   - Real-time monitoring of Generator, Reflector, and Curator interactions
   - View traces at https://www.comet.com/opik or local Opik instance

### New Features & Advanced Usage

#### Configurable Retry Prompts (Added in v0.4.1)
All three ACE roles now support customizable retry prompts for JSON parsing failures:

```python
from ace import Generator, Reflector, Curator, LiteLLMClient

llm = LiteLLMClient(model="gpt-4")

# Use English defaults (recommended)
generator = Generator(llm)
reflector = Reflector(llm)
curator = Curator(llm)

# Or customize for specific models/languages
generator = Generator(
    llm,
    retry_prompt="\n\n[日本語] 有効なJSONオブジェクトのみを返してください。"
)
```

**Benefits**:
- Reduces JSON parse failures by 7-12%
- Supports multilingual models
- Consistent behavior across all roles

#### Checkpoint Saving During Training
OfflineAdapter now supports automatic checkpoint saving:

```python
from ace import OfflineAdapter

adapter = OfflineAdapter(playbook, generator, reflector, curator)

# Save playbook every 10 successful samples
results = adapter.run(
    samples,
    environment,
    epochs=3,
    checkpoint_interval=10,  # Save every 10 samples
    checkpoint_dir="./checkpoints"  # Where to save
)
```

**Output**:
- `checkpoint_10.json`, `checkpoint_20.json`, etc. (numbered)
- `latest.json` (always overwritten with most recent)

**Use Cases**:
- Resume training after interruption
- Compare playbook evolution over time
- Early stopping based on validation metrics

#### Prompt Version Guidance
The framework includes two prompt versions (see `docs/PROMPTS.md`):

1. **v1.0** (`prompts.py`): Simple, minimal - use for tutorials and learning
2. **v2.1** (`prompts_v2_1.py`): **RECOMMENDED** for production (+17% success rate)

```python
from ace.prompts_v2_1 import PromptManager

prompt_mgr = PromptManager()
generator = Generator(llm, prompt_template=prompt_mgr.get_generator_prompt())
reflector = Reflector(llm, prompt_template=prompt_mgr.get_reflector_prompt())
curator = Curator(llm, prompt_template=prompt_mgr.get_curator_prompt())
```

#### Feature Detection
Check which optional dependencies are available:

```python
from ace.features import has_opik, has_litellm, get_available_features

if has_opik():
    print("Opik observability available")

# Or check all features at once
features = get_available_features()
# {'opik': True, 'litellm': True, 'langchain': False, ...}
```

## Python Requirements
- Python 3.11+ (developed with 3.12)
- Dependencies managed via UV (see pyproject.toml/uv.lock)
- Core (~100MB): LiteLLM (required for Reflector/Curator), Pydantic, Python-dotenv, tenacity
- Optional dependencies available for:
  - `observability`: Opik integration for production monitoring and **automatic token/cost tracking**
  - `langchain`: LangChain integration (enterprise - langchain-openai + langchain-litellm)
  - `transformers`: Local model support with transformers, torch, accelerate
  - `all`: All optional dependencies combined (~500MB total)
- Development dependencies (NOT distributed to PyPI):
  - Managed via `[dependency-groups]` (PEP 735)
  - Auto-installed for contributors via `uv sync`
  - Includes: pytest, black, mypy, pre-commit, git-changelog

## Automatic Token Usage & Cost Tracking

ACE framework now includes **automatic token usage and cost tracking** via Opik integration:

### Features
- ✅ **Zero-configuration tracking**: Automatic when using `observability` dependencies
- ✅ **Real-time cost monitoring**: View costs in Opik dashboard
- ✅ **Multi-provider support**: Works with OpenAI, Anthropic, Google, Cohere, etc.
- ✅ **Graceful degradation**: Works without Opik installed

### Setup
```bash
# Install with observability features
pip install ace-framework[observability]

# Or for development
uv sync  # Already includes Opik in optional dependencies
```

### Usage
```python
from ace.llm_providers.litellm_client import LiteLLMClient

# Token tracking is automatically enabled
client = LiteLLMClient(model="gpt-4")
response = client.complete("Hello world!")

# Costs and token usage automatically logged to Opik
# View at: https://www.comet.com/opik
```

### Cost Analytics
- **Per-call tracking**: Individual LLM call costs
- **Role attribution**: Costs by Generator/Reflector/Curator
- **Adaptation metrics**: Cost efficiency over time
- **Budget monitoring**: Optional cost limits and alerts

## Environment Setup
Set your LLM API key for examples and demos:
```bash
export OPENAI_API_KEY="your-api-key"
# Or use other providers: ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.
```