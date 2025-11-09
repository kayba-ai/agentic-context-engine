# Browser-Use + ACE Integration Examples

This folder demonstrates how to integrate the **ACE (Agentic Context Engineering)** framework with **[browser-use](https://github.com/browser-use/browser-use)** for self-improving browser automation agents.

## 🎯 What is This?

ACE enables browser automation agents to **learn from their execution feedback** and improve over time.

**How it works:**
```
Sample → [Generator] → Strategy → [Browser-Use] → Result
            ↑                                        ↓
        Playbook ← [Curator] ← [Reflector] ← Feedback
        (learns)
```

Instead of static prompts, ACE agents:

1. **Generate** strategies for browser tasks
2. **Execute** them using browser-use
3. **Reflect** on what worked/failed
4. **Curate** lessons into a persistent playbook
5. **Improve** on subsequent tasks

## 📁 Folder Structure

```
examples/browser-use/
├── README.md              # Getting started guide (you are here!)
├── TEMPLATE.py            # Clean template for your own use cases
├── shared.py              # Generic utilities (domain-agnostic)
├── debug.py               # Debug/inspection utilities
├── domain-checker/        # Domain availability examples
└── form-filler/           # Form filling examples
```

Each example folder contains:
- ACE version (WITH learning)
- Baseline version (WITHOUT learning for comparison)
- Example-specific utilities (*_utils.py)

## 🚀 Quick Start

### 1. Installation

```bash
# Install ACE framework with browser-use demo dependencies
pip install ace-framework[demos]

# Or for development (using UV - recommended)
cd agentic-context-engine
uv sync --extra demos      # Install with demos dependencies
# Or: uv sync --all-extras  # Install all optional dependencies
```

### 2. Set API Key

```bash
# Set your LLM API key (ACE uses LiteLLM, supports 100+ providers)
export OPENAI_API_KEY="your-api-key"
# Or: ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.
```

### 3. Run an Example

```bash
# Domain checker WITH ACE (learns after each domain)
uv run python examples/browser-use/domain-checker/ace_domain_checker.py

# Form filler WITH ACE
uv run python examples/browser-use/form-filler/ace_form_filler.py
```

## 📊 Expected Results

**Baseline (no learning):**
- Same performance on every task
- Static strategies
- No improvement over time

**ACE (with learning):**
- Performance improves across tasks
- Learns efficient patterns
- Adapts strategies based on feedback
- Builds reusable playbook

## 🛠️ Create Your Own Use Case

### Option 1: Start from Template

Copy `TEMPLATE.py` and customize for your task:

```python
# 1. Define your evaluation environment
class MyTaskEnvironment(TaskEnvironment):
    def evaluate(self, sample, generator_output):
        # Your task-specific evaluation logic
        pass

# 2. Create ACE components
llm = LiteLLMClient(model="gpt-4o")
adapter = OnlineAdapter(
    playbook=Playbook(),
    generator=Generator(llm),
    reflector=Reflector(llm),
    curator=Curator(llm)
)

# 3. Run and learn!
results = adapter.run(samples, environment)
```

### Option 2: Adapt an Example

Browse `domain-checker/` or `form-filler/` examples and modify them for your needs.

## 📖 Documentation

- **Main ACE Framework:** See `/README.md` and `/docs/` in root
- **Domain Checker Examples:** See `domain-checker/README.md`
- **Form Filler Examples:** See `form-filler/README.md`
- **Browser-Use Library:** https://github.com/browser-use/browser-use

## 🔬 Key Concepts

### ACE Components

1. **Generator**: Plans browser automation strategies
2. **Reflector**: Analyzes execution feedback (errors, successes, efficiency)
3. **Curator**: Updates playbook with learned lessons
4. **Playbook**: Persistent knowledge base (bullets with helpful/harmful scores)

### Adaptation Modes

- **OnlineAdapter**: Learn after each task (used in these examples)
- **OfflineAdapter**: Train on batch of examples first, then deploy

### Environment Integration

Your `TaskEnvironment` bridges ACE with browser-use:
- Receives strategy from Generator
- Executes browser automation
- Returns feedback to Reflector

## 💡 Tips

1. **Start Simple**: Begin with baseline demo, then compare with ACE version
2. **Headless Mode**: Set `headless=True` for faster execution (no GUI)
3. **Debug Mode**: Use `debug.print_history_details()` to inspect browser actions
4. **Cost Tracking**: Enable Opik observability to monitor token usage
5. **Prompt Versions**: Use v2.1 prompts for best performance - they include MCP-inspired enhancements for better reasoning and error handling

## 📝 Common Utilities

### `shared.py` - Common Utilities

Contains helper functions and constants used across browser automation examples:

```python
from shared import (
    # Timeout and retry handling
    calculate_timeout_steps,   # Convert timeout duration to estimated step count
    MAX_RETRIES,              # Default retry attempts (3)
    DEFAULT_TIMEOUT_SECONDS,   # Default browser timeout (180s)

    # Output formatting and storage
    format_result_output,      # Pretty-print browser results
    save_results_to_file,      # Save results to JSON

    # Browser configuration
    get_browser_config,        # Standard browser settings
)
```

### `debug.py` - Debug Utilities

```python
from debug import print_history_details

# Print comprehensive browser execution details
history = await agent.run()
print_history_details(history)
# Shows: actions, results, URLs, errors, thoughts, timing, etc.
```

### Example-Specific Utilities

- `domain-checker/domain_utils.py` - Domain checking utilities
- `form-filler/form_utils.py` - Form data and utilities

## 🤝 Contributing

Have a cool browser automation use case? Add a new example folder!

1. Create `your-use-case/` folder
2. Add `ace_*.py` and `baseline_*.py` files
3. Create local `README.md` and `*_utils.py`
4. Keep `shared.py` generic (no use-case-specific code)

## 🐛 Troubleshooting

**Import errors after restructuring?**
- Files in subfolders use `sys.path.insert()` to import from parent
- Check that `shared.py` and `debug.py` are in `browser-use/` root

**Browser not starting?**
- Browser-use automatically downloads Chromium via Playwright on first run
- If issues persist, check browser-use installation: `pip install browser-use`

**LLM API errors?**
- Verify API key is set: `echo $OPENAI_API_KEY`
- Check LiteLLM supported models: https://docs.litellm.ai/docs/

Happy automating! 🤖✨
