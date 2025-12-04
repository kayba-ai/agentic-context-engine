# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an implementation scaffold for reproducing the Agentic Context Engineering (ACE) method from the paper "Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models" (arXiv:2510.04618).

The framework enables AI agents to learn from their execution feedback through three collaborative roles: Agent (produces answers), Reflector (analyzes performance), and SkillManager (updates the knowledge base called a "skillbook").

## Development Commands

### Package Installation
```bash
# Install from PyPI (end users)
pip install ace-framework  # Includes Instructor for robust JSON parsing

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
uv run python examples/litellm/simple_ace_example.py
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
- Unit tests for individual components (skillbook, roles, adapters)
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
# LiteLLM examples (ACELiteLLM)
python examples/litellm/simple_ace_example.py        # Quick start
python examples/litellm/async_learning_example.py   # Async learning demo
python examples/litellm/deduplication_example.py    # Skill deduplication
python examples/litellm/skillbook_persistence.py    # Save/load skillbooks
python examples/litellm/seahorse_emoji_ace.py       # Kayba test demo

# LangChain examples (ACELangChain)
python examples/langchain/simple_chain_example.py   # Basic chain wrapping
python examples/langchain/agent_with_tools_example.py  # Agent with tools
python examples/langchain/async_learning_example.py # Async learning

# Local models (LM Studio, Ollama)
python examples/local-models/lm_studio_example.py
python examples/local-models/ollama_starter_template.py

# Prompt comparison
python examples/prompts/compare_v1_v2_prompts.py
python examples/prompts/advanced_prompts_v2.py

# Browser automation demos (contributors: install with `uv sync --group demos`)
uv run python examples/browser-use/domain-checker/ace_domain_checker.py
uv run python examples/browser-use/form-filler/ace_form_filler.py
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
- **Skillbook**: Structured context store containing skills (strategy entries) with helpful/harmful counters
  - Uses TOON (Token-Oriented Object Notation) format for 16-62% token savings
  - `skillbook.as_prompt()` returns TOON format (for LLM consumption)
  - `str(skillbook)` returns markdown format (for human debugging)
- **Update Operations**: Incremental updates to the skillbook (ADD, UPDATE, TAG, REMOVE)
- **Three Agentic Roles** sharing the same base LLM:
  - **Agent**: Produces answers using the current skillbook
  - **Reflector**: Analyzes errors and classifies skill contributions
  - **SkillManager**: Emits update operations to update the skillbook
- **Two Architecture Patterns**:
  - **Full ACE Pipeline**: Agent+Reflector+SkillManager (for new agents)
  - **Integration Pattern**: Reflector+SkillManager only (for existing systems like browser-use, LangChain)

### Insight Levels (Reflector Scope)

The ACE framework operates at three insight levels based on what scope the Reflector analyzes:

| Level | Reflector Scope | Description | Implementation |
|-------|-----------------|-------------|----------------|
| **Micro** | Single interaction + environment | Request → response → ground truth/feedback | OfflineACE, OnlineACE with TaskEnvironment |
| **Meso** | Full agent run | Multiple steps with reasoning trace (no external feedback) | `_learn_with_trace()` (e.g., AgentExecutor intermediate_steps) |
| **Macro** | Cross-run analysis | Comparing patterns across multiple runs | Future enhancement |

- **Micro-level insights**: Generated from the full ACE loop with environment feedback and ground truth. Learns from correctness evaluation.
- **Meso-level insights**: Generated from agent reasoning traces (thoughts, tool calls, observations) without external ground truth. Learns from execution patterns.
- **Macro-level insights**: Generated by comparing patterns across multiple runs. Most comprehensive but not yet implemented.

### Module Structure

**ace/** - Core library modules:
- `skillbook.py`: Skill and Skillbook classes for context storage (TOON format)
- `updates.py`: UpdateOperation and UpdateBatch for incremental updates
- `roles.py`: Agent, Reflector, SkillManager implementations
- `adaptation.py`: OfflineACE and OnlineACE orchestration loops
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
- `deduplication/`: Skill deduplication (similarity detection, consolidation)

**ace/observability/** - Production monitoring and observability:
- `opik_integration.py`: Enterprise-grade monitoring with Opik
- `tracers.py`: Automatic tracing decorators for all role interactions
- **Automatic token usage and cost tracking** for all LLM calls
- Real-time cost monitoring via Opik dashboard
- Real-time monitoring of Agent, Reflector, and SkillManager interactions

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
   - Sample → Agent (produces answer) → Environment (evaluates) → Reflector (analyzes) → SkillManager (updates skillbook)
   - Offline: Multiple epochs over training samples
   - Online: Sequential processing of test samples
   - Use when: Building new agent from scratch, Q&A tasks, classification

2. **Integration Pattern** (for existing agents):
   - External agent executes task → Reflector analyzes → SkillManager updates skillbook
   - No ACE Agent - external framework handles execution
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
   - Real-time monitoring of Agent, Reflector, and SkillManager interactions
   - View traces at https://www.comet.com/opik or local Opik instance

### New Features & Advanced Usage

#### Checkpoint Saving During Training
OfflineACE now supports automatic checkpoint saving:

```python
from ace import OfflineACE

adapter = OfflineACE(skillbook, agent, reflector, skill_manager)

# Save skillbook every 10 successful samples
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
- Compare skillbook evolution over time
- Early stopping based on validation metrics

#### Async Learning Mode
Enable parallel learning where the Agent returns immediately while Reflector and SkillManager process in the background:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ASYNC LEARNING PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Sample 1 ──► Agent ──► Env ──► Reflector ─┐                               │
│  Sample 2 ──► Agent ──► Env ──► Reflector ─┼──► [Queue] ──► SkillManager ──► Skillbook
│  Sample 3 ──► Agent ──► Env ──► Reflector ─┘              (serialized)     │
│             (parallel)       (parallel)                                     │
│                                                                             │
│  ✓ Agent returns immediately (fast response)                               │
│  ✓ Multiple Reflectors run concurrently (parallel LLM calls)               │
│  ✓ Single SkillManager processes queue sequentially (safe skillbook updates)│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Basic Usage**:
```python
from ace import OfflineACE

adapter = OfflineACE(
    skillbook=skillbook,
    agent=agent,
    reflector=reflector,
    skill_manager=skill_manager,
    async_learning=True,           # Enable async mode
    max_reflector_workers=3,       # Parallel Reflector threads
)

# Returns fast - only Agent + evaluation time
results = adapter.run(samples, environment, epochs=3)
# Learning completes in background, then run() returns
```

**Fire-and-Forget Mode** (get results immediately, learning continues):
```python
# Don't wait for learning to complete
results = adapter.run(samples, environment, wait_for_learning=False)

# Use results immediately while learning continues in background
for r in results:
    print(r.agent_output.final_answer)

# Check learning progress anytime
print(adapter.learning_stats)
# {'tasks_submitted': 30, 'reflections_completed': 25, 'curations_completed': 20, ...}

# Wait when needed (e.g., before saving skillbook)
adapter.wait_for_learning(timeout=60.0)
skillbook.save_to_file("learned.json")

# Cleanup when done
adapter.stop_async_learning()
```

**Control Methods**:
- `adapter.learning_stats` - Dict with queue sizes, completion counts, running status
- `adapter.wait_for_learning(timeout=None)` - Block until queue is drained
- `adapter.stop_async_learning(wait=True)` - Shutdown the pipeline

**Why This Architecture**:
- **Reflector is safe to parallelize**: Reads skillbook, produces independent analysis
- **SkillManager MUST be serialized**: Writes to skillbook, handles deduplication
- **3x faster learning**: Reflector LLM calls run concurrently
- **Eventual consistency**: Agent uses whatever skillbook state is available

#### Prompt Version Guidance
The framework includes two prompt versions (see `docs/PROMPTS.md`):

1. **v1.0** (`prompts.py`): Simple, minimal - use for tutorials and learning
2. **v2.1** (`prompts_v2_1.py`): **RECOMMENDED** for production (+17% success rate)

```python
from ace.prompts_v2_1 import PromptManager

prompt_mgr = PromptManager()
agent = Agent(llm, prompt_template=prompt_mgr.get_agent_prompt())
reflector = Reflector(llm, prompt_template=prompt_mgr.get_reflector_prompt())
skill_manager = SkillManager(llm, prompt_template=prompt_mgr.get_skill_manager_prompt())
```

#### Feature Detection
Check which optional dependencies are available:

```python
from ace.features import has_opik, has_litellm, has_instructor, get_available_features

if has_opik():
    print("Opik observability available")

if has_instructor():
    print("Instructor structured output support available")

# Or check all features at once
features = get_available_features()
# {'opik': True, 'litellm': True, 'instructor': True, 'langchain': False, ...}
```

#### Instructor Integration for Robust Structured Outputs (Default in v0.5.0)
ACE includes Instructor by default for more reliable JSON parsing:

```python
from ace.llm_providers.litellm_client import LiteLLMClient
from ace.llm_providers.instructor_client import wrap_with_instructor

# Enable automatic Pydantic validation for better reliability
llm = wrap_with_instructor(LiteLLMClient(model="ollama/gemma3:1b"))
agent = Agent(llm)  # Auto-validates AgentOutput

# Or use LiteLLMClient directly for standard behavior
llm = LiteLLMClient(model="gpt-4")
agent = Agent(llm)  # Manual JSON parsing (existing behavior)
```

**Benefits**: Field validation, type coercion, intelligent retry (~15% fewer parsing errors)
**Recommended for**: Small models (Ollama, Gemma, Phi) with JSON formatting issues

## Python Requirements
- Python 3.11+ (developed with 3.12)
- Dependencies managed via UV (see pyproject.toml/uv.lock)
- Core (~105MB): LiteLLM, Pydantic, Instructor, Python-dotenv, tenacity
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
- **Role attribution**: Costs by Agent/Reflector/SkillManager
- **Adaptation metrics**: Cost efficiency over time
- **Budget monitoring**: Optional cost limits and alerts

## Environment Setup
Set your LLM API key for examples and demos:
```bash
export OPENAI_API_KEY="your-api-key"
# Or use other providers: ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.
```