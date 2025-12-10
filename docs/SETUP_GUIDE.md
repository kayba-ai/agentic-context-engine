# ⚙️ ACE Framework Setup Guide

Quick setup and configuration guide for ACE Framework.

## Requirements

- **Python 3.11 or higher**
- API key for your LLM provider (OpenAI, Anthropic, Google, etc.)

Check Python version:
```bash
python --version  # Should show 3.11+
```

---

## Installation

### For Users

```bash
# Basic installation
pip install ace-framework

# With optional features
pip install ace-framework[observability]  # Opik monitoring + cost tracking
pip install ace-framework[browser-use]    # Browser automation
pip install ace-framework[langchain]      # LangChain integration
pip install ace-framework[all]            # All features
```

### For Contributors

```bash
git clone https://github.com/kayba-ai/agentic-context-engine
cd agentic-context-engine
uv sync  # Installs everything automatically (10-100x faster than pip)
```

---

## API Key Setup

### Option 1: Environment Variable (Recommended)

```bash
# Set in your shell
export OPENAI_API_KEY="sk-..."

# Or create .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

Load in Python:
```python
from dotenv import load_dotenv
load_dotenv()  # Loads from .env file
```

### Option 2: Direct in Code

```python
from ace import LiteLLMClient

client = LiteLLMClient(
    model="gpt-4o-mini",
    api_key="your-key-here"  # Not recommended for production
)
```

---

## Provider Examples

### OpenAI

1. Get API key: [platform.openai.com](https://platform.openai.com)
2. Set key: `export OPENAI_API_KEY="sk-..."`
3. Use it:
```python
from ace import LiteLLMClient
client = LiteLLMClient(model="gpt-4o-mini")
```

### Anthropic Claude

1. Get API key: [console.anthropic.com](https://console.anthropic.com)
2. Set key: `export ANTHROPIC_API_KEY="sk-ant-..."`
3. Use it:
```python
client = LiteLLMClient(model="claude-3-5-sonnet-20241022")
```

### Google Gemini

1. Get API key: [makersuite.google.com](https://makersuite.google.com)
2. Set key: `export GOOGLE_API_KEY="AIza..."`
3. Use it:
```python
client = LiteLLMClient(model="gemini-pro")
```

### Local Models (Ollama)

1. Install Ollama: [ollama.ai](https://ollama.ai)
2. Pull model: `ollama pull llama2`
3. Use it:
```python
client = LiteLLMClient(model="ollama/llama2")
```

**Supported Providers:** 100+ via LiteLLM (AWS Bedrock, Azure, Cohere, Hugging Face, etc.)

---

## Advanced Configuration

### Custom LLM Parameters

```python
from ace import LiteLLMClient

client = LiteLLMClient(
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=2048,
    timeout=60  # seconds
)
```

### Production Monitoring (Opik)

```bash
pip install ace-framework[observability]
```

Opik automatically tracks:
- Token usage per LLM call
- Cost per operation
- Agent/Reflector/SkillManager performance
- Skillbook evolution over time

View dashboard: [comet.com/opik](https://www.comet.com/opik)

### Skillbook Storage

```python
from ace import Skillbook

# Save skillbook
skillbook.save_to_file("my_skillbook.json")

# Load skillbook
skillbook = Skillbook.load_from_file("my_skillbook.json")

# For production: Use database storage
# PostgreSQL, SQLite, or vector stores supported
```

### Checkpoint Saving

```python
from ace import OfflineACE

adapter = OfflineACE(skillbook, agent, reflector, skill_manager)

# Save skillbook every 10 samples during training
results = adapter.run(
    samples,
    environment,
    checkpoint_interval=10,
    checkpoint_dir="./checkpoints"
)
```

---

## Troubleshooting

### Import Errors

```bash
# Upgrade to latest version
pip install --upgrade ace-framework

# Check installation
pip show ace-framework
```

### API Key Not Working

```bash
# Verify key is set
echo $OPENAI_API_KEY

# Test different model
from ace import LiteLLMClient
client = LiteLLMClient(model="gpt-3.5-turbo")  # Cheaper for testing
```

### Rate Limits

```python
from ace import LiteLLMClient

# Add delays between calls
import time
time.sleep(1)  # 1 second between calls

# Or use a cheaper/faster model
client = LiteLLMClient(model="gpt-3.5-turbo")
```

### JSON Parse Failures

```python
# Increase max_tokens for SkillManager/Reflector
from ace import SkillManager, Reflector

llm = LiteLLMClient(model="gpt-4o-mini", max_tokens=2048)  # Higher limit
skill_manager = SkillManager(llm)
reflector = Reflector(llm)
```

---

## Need More Help?

- **GitHub Issues:** [github.com/kayba-ai/agentic-context-engine/issues](https://github.com/kayba-ai/agentic-context-engine/issues)
- **Discord Community:** [discord.gg/mqCqH7sTyK](https://discord.gg/mqCqH7sTyK)
- **Documentation:** [Complete Guide](COMPLETE_GUIDE_TO_ACE.md), [Quick Start](QUICK_START.md), [Integration Guide](INTEGRATION_GUIDE.md)

---

**Next Steps:** Check out the [Quick Start Guide](QUICK_START.md) to build your first self-learning agent!
