# ACELiteLLM Examples

Simple examples showing how to use ACELiteLLM for quick learning agents.

## What is ACELiteLLM?

ACELiteLLM is the simplest way to add learning to any LLM:

```python
from ace.integrations import ACELiteLLM
from ace import Sample, SimpleEnvironment

# Create agent
agent = ACELiteLLM(model="gpt-4o-mini")

# Ask questions
answer = agent.ask("What is Python?")

# Learn from examples
samples = [Sample(question="What is Python?", ground_truth="A programming language")]
agent.learn(samples, SimpleEnvironment())

# Save knowledge
agent.save_skillbook("learned.json")
```

## Examples

### 1. litellm_ace_example.py
**The absolute basics** - ask, learn, save.

```bash
export OPENAI_API_KEY="your-key"
python litellm_ace_example.py
```

Shows:
- Basic ask/learn/save pattern
- Before/after learning comparison
- 20 lines of code total


## Quick Setup

1. **Install:**
   ```bash
   pip install ace-framework
   ```

2. **Set API key:**
   ```bash
   export OPENAI_API_KEY="your-key"
   ```

3. **Run:**
   ```bash
   python litellm_ace_example.py
   ```

## Supported Models

Works with 100+ models via LiteLLM:

```python
# OpenAI
ACELiteLLM(model="gpt-4o-mini")

# Anthropic
ACELiteLLM(model="claude-3-haiku-20240307")

# Google
ACELiteLLM(model="gemini/gemini-1.5-flash")

# Local (Ollama)
ACELiteLLM(model="ollama/llama2")
```

## When to Use ACELiteLLM

✅ **Good for:**
- Quick prototypes
- Personal assistants
- Small projects

❌ **Use something else for:**
- Browser automation → Use `ACEAgent`
- LangChain projects → Use `ACELangChain`
- Complex workflows → Use full ACE components

## Next Steps

- See `../starter-templates/` for more complete examples
- Check `../browser-use/` for automation examples
- Read `docs/INTEGRATION_GUIDE.md` for advanced usage