# Browser-Use + ACE Integration Examples

This folder demonstrates how to integrate the **ACE (Agentic Context Engineering)** framework with **[browser-use](https://github.com/browser-use/browser-use)** for self-improving browser automation agents.

## 🎯 What is This?

ACE enables browser automation agents to **learn from their execution feedback** and improve over time.

**How it works:**
```
Task → [browser-use Agent] → Execution Result
         ↑                           ↓
     Skillbook ← [SkillManager] ← [Reflector] ← Feedback
     (context)     (updates)       (analyzes)
```

Instead of static prompts, ACEAgent:

1. **Injects** skillbook context into browser-use Agent
2. **Executes** browser tasks using browser-use
3. **Reflects** on execution feedback (what worked/failed)
4. **Updates** lessons into the skillbook
5. **Improves** on subsequent tasks

## 📁 Folder Structure

```
examples/browser-use/
├── README.md                           # Getting started guide (you are here!)
├── TEMPLATE.py                         # Clean template for your own use cases
├── simple_ace_agent.py                 # Simple ACEAgent example
├── online-shopping/                    # Online shopping automation demos
│   ├── ace-online-shopping.py          # ACE version (WITH learning)
│   └── baseline-online-shopping.py     # Baseline version (WITHOUT learning)
├── domain-checker/                     # Domain availability examples
│   ├── ace_domain_checker.py           # ACE version (WITH learning)
│   ├── baseline_domain_checker.py      # Baseline version (WITHOUT learning)
│   └── domain_utils.py                 # Domain checking utilities
└── form-filler/                        # Form filling examples
    ├── ace_browser_use.py              # ACE version (WITH learning)
    ├── baseline_browser_use.py         # Baseline version (WITHOUT learning)
    └── form_utils.py                   # Form data and utilities
```

Each example folder contains:
- ACE version (WITH learning)
- Baseline version (WITHOUT learning for comparison)
- Example-specific utilities (*_utils.py)
- Results images and saved skillbooks

## 🚀 Quick Start

### 1. Installation

```bash
# For end users (recommended)
pip install ace-framework[browser-use]

# For contributors running browser demos (UV)
cd agentic-context-engine
uv sync --group demos      # Installs browser-use, playwright, rich, etc.
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
uv run python examples/browser-use/form-filler/ace_browser_use.py
```

## 🎬 Live Demos

### 🛒 Online Shopping Demo

A grocery shopping automation comparison where both agents find the basket price for 5 essential items at Migros online store.
The ACE agent learns optimal shopping strategies while the baseline agent repeats the same mistakes.

![Online Shopping Demo Results](online-shopping/results-online-shopping-brwoser-use.png)

**Task**: Shop for 5 essential items (milk, eggs, bananas, butter, bread) and find the cheapest options while adding them to the basket.

**ACE Performance**:
- **29.8% fewer steps** on average (57.2 vs 81.5)
- **49.0% reduction** in browser-use tokens (595k vs 1,166k)
- **42.6% total cost reduction** even when including ACE learning overhead

**Key Results**:
- **ACE Agent**: Learns efficient product search patterns and UI navigation strategies
- **Baseline Agent**: Struggles with inconsistent website interactions and search failures
- **Learning Advantage**: ACE adapts to website quirks and develops reliable shopping workflows

Try it yourself:
```bash
# Run baseline version (no learning)
uv run python examples/browser-use/online-shopping/baseline-online-shopping.py

# Run ACE-enhanced version (learns and improves)
uv run python examples/browser-use/online-shopping/ace-online-shopping.py
```

---

### 🌐 Domain Availability Checker Demo

A real-world comparison where both Browser Use agents check 10 domains for availability using browser automation. Same prompt, same Browser Use setup—but the ACE agent autonomously generates strategies from execution feedback.

![Browser Use Demo Results](domain-checker/Browseruse_domain_demo_results.png)

**How ACE + Browser-Use Works:**
- **ACE learns strategies**: "Click search box, then type domain name"
- **Browser-Use executes**: Actually controls the browser (clicking, typing, etc.)
- **ACE improves**: Learns from failures like "search box was hidden, scroll first"

**Performance Comparison:**

| Metric | Baseline | ACE |
|--------|---------|-----|
| Success rate | 30% | 100% |
| Avg steps per domain | 38.8 | 6.9 |
| Token cost | 1776k | 605k (incl. ACE) |

**Result**: ACE starts similar to baseline but learns optimal patterns, achieving consistent 3-step completion.

Try it yourself:
```bash
# Run baseline version (no learning)
uv run python examples/browser-use/domain-checker/baseline_domain_checker.py

# Run ACE-enhanced version (learns and improves)
uv run python examples/browser-use/domain-checker/ace_domain_checker.py
```

## 📊 Results

**Baseline (no learning):**
- Same performance on every task
- Static strategies
- No improvement over time

**ACE (with learning):**
- Performance improves across tasks
- Learns efficient patterns
- Adapts strategies based on feedback
- Builds reusable skillbook

## 🛠️ Create Your Own Use Case

### Option 1: Start from Template

Copy `TEMPLATE.py` and customize for your task:

```python
from ace import ACEAgent
from browser_use import ChatBrowserUse

# 1. Create ACE agent - handles everything automatically!
agent = ACEAgent(
    llm=ChatBrowserUse(),                    # Browser automation LLM
    ace_model="claude-haiku-4-5-20251001",   # ACE learning LLM
    skillbook_path=str(skillbook_path) if skillbook_path.exists() else None,
    max_steps=25,                            # Browser automation steps
    calculate_cost=True                      # Track usage
)

# 2. Run and learn!
result = await agent.run(task=your_task_description)
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

**ACEAgent**: Integration wrapper that adds ACE learning to browser-use:
1. **browser-use Agent**: Executes browser automation tasks
2. **Reflector**: Analyzes execution feedback (errors, successes, efficiency)
3. **SkillManager**: Updates skillbook with learned lessons
4. **Skillbook**: Persistent knowledge base injected as context

ACEAgent uses the Integration Pattern - browser-use handles execution, ACE handles learning!

### Learning Modes

**ACEAgent** automatically learns from each task execution:
- Analyzes what worked and what failed
- Updates strategies in the skillbook
- Applies learned lessons to future tasks

### Browser Integration

ACEAgent integrates seamlessly with browser-use:
- Uses browser-use for actual automation (clicking, typing, etc.)
- Learns from browser execution feedback
- Builds strategies specific to web automation tasks

## 💡 Tips

1. **Start Simple**: Begin with baseline demo, then compare with ACE version
2. **Headless Mode**: Set `headless=True` for faster execution (no GUI)
3. **Cost Tracking**: Enable Opik observability to monitor token usage
4. **Prompt Versions**: Use v2.1 prompts for best performance - they include MCP-inspired enhancements for better reasoning and error handling

## 📝 Common Utilities

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
- Check that dependencies are installed correctly

**Browser not starting?**
- Browser-use automatically downloads Chromium via Playwright on first run
- If issues persist, install demos group: `uv sync --group demos` (contributors)

**LLM API errors?**
- Verify API key is set: `echo $OPENAI_API_KEY`
- Check LiteLLM supported models: https://docs.litellm.ai/docs/

Happy automating! 🤖✨
