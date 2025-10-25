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
pip install ace-framework[demos]         # Demo applications (browser automation)
pip install ace-framework[langchain]     # LangChain integration
pip install ace-framework[transformers]  # Local model support
pip install ace-framework[dev]           # Development tools

# Development installation (contributors) - UV Method (Recommended)
git clone https://github.com/kayba-ai/agentic-context-engine
cd agentic-context-engine
uv sync                                  # Install all dependencies (10-100x faster than pip)

# Development installation (contributors) - Traditional Method
git clone https://github.com/kayba-ai/agentic-context-engine
cd agentic-context-engine
pip install -e .
```

### Dependency Management (UV - Modern Approach)
This project uses UV for ultra-fast dependency management with automatic locking.

```bash
# Install dependencies (10-100x faster than pip)
uv sync

# Development workflow (contributors)
uv add package-name              # Add new dependency
uv remove package-name           # Remove dependency
uv sync                          # Install/update all dependencies
uv sync --locked                 # Use exact versions from uv.lock (for CI)

# Run scripts (auto-activates virtual environment)
uv run python examples/simple_ace_example.py
uv run pytest                    # Run tests
uv run python -m ace.demos       # Run demos

# Update dependencies
uv lock --upgrade                # Update all to latest compatible versions
uv lock --upgrade-package requests  # Update specific package

# Python version management
uv python pin 3.12              # Pin Python version for project
uv python install 3.11          # Install Python 3.11
```

**Files:**
- `pyproject.toml` - Project metadata and dependencies (PEP 621 standard)
- `uv.lock` - Locked dependencies with exact versions (auto-generated, like package-lock.json)
- `.python-version` - Pinned Python version for the project

### Running Tests
```bash
# Run all tests
python -m unittest discover -s tests

# Run specific test file
python -m unittest tests.test_adaptation

# Run with verbose output
python -m unittest discover -s tests -v

# Using pytest (with dev dependencies)
uv run pytest                        # Run all tests
uv run pytest tests/test_adaptation.py  # Run specific test file
uv run pytest -v                     # Verbose output
```

### Code Quality
```bash
# Format code with Black
uv run black ace/ tests/ examples/

# Type checking with MyPy
uv run mypy ace/

# Run all quality checks (when available in dev dependencies)
uv run black --check ace/ tests/ examples/
uv run mypy ace/
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

# Browser automation demos (requires browser-use dependencies)
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

# Benchmarking
python scripts/run_benchmark.py
python scripts/compare_baseline_vs_ace.py
python scripts/analyze_ace_results.py
python scripts/explain_ace_performance.py
```

## Architecture

### Core Concepts
- **Playbook**: Structured context store containing bullets (strategy entries) with helpful/harmful counters
- **Delta Operations**: Incremental updates to the playbook (ADD, UPDATE, TAG, REMOVE)
- **Three Agentic Roles** sharing the same base LLM:
  - **Generator**: Produces answers using the current playbook
  - **Reflector**: Analyzes errors and classifies bullet contributions
  - **Curator**: Emits delta operations to update the playbook

### Module Structure

**ace/** - Core library modules:
- `playbook.py`: Bullet and Playbook classes for context storage
- `delta.py`: DeltaOperation and DeltaBatch for incremental updates
- `roles.py`: Generator, Reflector, Curator implementations
- `adaptation.py`: OfflineAdapter and OnlineAdapter orchestration loops
- `llm.py`: LLMClient interface with DummyLLMClient and TransformersLLMClient
- `prompts.py`: Default prompt templates for each role
- `prompts_v2.py`: Enhanced prompt templates with improved performance
- `llm_providers/`: Production LLM client implementations
  - `litellm_client.py`: LiteLLM integration (100+ model providers)
  - `langchain_client.py`: LangChain integration

**ace/explainability/** - Explainability framework:
- `evolution_tracker.py`: Track playbook evolution over time
- `attribution_analyzer.py`: Analyze which bullets contribute to performance
- `interaction_tracer.py`: Trace role interactions and dependencies
- `visualizer.py`: Visualization tools for explainability analysis

**ace/observability/** - Production monitoring and observability:
- Integration with Opik for enterprise-grade monitoring and tracking

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

1. **Adaptation Flow**:
   - Sample → Generator (produces answer) → Environment (evaluates) → Reflector (analyzes) → Curator (updates playbook)
   - Offline: Multiple epochs over training samples
   - Online: Sequential processing of test samples

2. **LLM Integration**:
   - Implement `LLMClient` subclass for your model API
   - LiteLLMClient supports 100+ providers (OpenAI, Anthropic, Google, etc.)
   - LangChainClient provides LangChain integration
   - TransformersLLMClient for local model deployment
   - All roles share the same LLM instance

3. **Task Environment**:
   - Extend `TaskEnvironment` abstract class
   - Implement `evaluate()` to provide execution feedback
   - Return `EnvironmentResult` with feedback and optional ground truth

4. **Explainability Integration**:
   - Use `EvolutionTracker` to monitor playbook changes over time
   - Use `AttributionAnalyzer` to identify high-impact bullets
   - Use `InteractionTracer` to understand role dependencies
   - Use `ExplainabilityVisualizer` for visual analysis

## Python Requirements
- Python 3.11+ (developed with 3.12)
- Dependencies managed via UV (see pyproject.toml/uv.lock)
- Core: Pydantic, Python-dotenv, LiteLLM, tenacity
- Optional dependencies available for:
  - `demos`: Browser automation with browser-use, rich UI, datasets
  - `observability`: Opik integration for production monitoring
  - `langchain`: LangChain integration
  - `transformers`: Local model support with transformers, torch
  - `dev`: Development tools (pytest, black, mypy)
  - `all`: All optional dependencies combined

## Environment Setup
Set your LLM API key for examples and demos:
```bash
export OPENAI_API_KEY="your-api-key"
# Or use other providers: ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.
```