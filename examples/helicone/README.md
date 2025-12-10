# Helicone ACE Training Examples

This folder contains **example files** for training ACE agents using Helicone API observability data.

These are standalone Python scripts you can **copy and customize** for your own use case.

## Overview

Helicone (https://helicone.ai) provides observability for LLM applications, capturing:
- Complete conversation traces with tool calls
- Token usage and cost metrics
- Prompt caching effectiveness
- Latency and performance data

These tools parse Helicone logs and convert them into training samples for ACE.

## Files

### `helicone_loader.py`
Loads and processes Helicone JSON export files into structured trace objects.

**Key Classes:**
- `HeliconeTrace`: Complete conversation trace with metrics
- `ConversationTurn`: Individual message in the conversation
- `ToolCall`: Tool invocation with input/output
- `HeliconeLoader`: Parser for Helicone JSON files

**Usage:**
```python
from helicone_loader import HeliconeLoader

loader = HeliconeLoader('path/to/helicone.json')
trace = loader.load()

# Analyze the trace
print(f"Total tokens: {trace.total_tokens}")
print(f"Tools used: {trace.get_tool_stats()}")
print(f"Cache effectiveness: {trace.get_cache_effectiveness():.1f}%")
```

### `tool_selection_environment.py`
ACE TaskEnvironment for evaluating tool selection decisions.

**Key Classes:**
- `ToolSelectionEnvironment`: Evaluates whether agents select appropriate tools
- `ToolSelectionSample`: Represents a decision point in the conversation

**Usage:**
```python
from helicone_loader import HeliconeLoader
from tool_selection_environment import ToolSelectionEnvironment

loader = HeliconeLoader('path/to/helicone.json')
trace = loader.load()

env = ToolSelectionEnvironment(trace)
print(f"Extracted {env.get_num_samples()} training samples")

# Format as question for ACE Generator
question = env.format_sample_as_question(0)
```

### `offline_training_replay.py`
Complete example showing ACE offline training with Helicone replay data.

**What it does:**
- Loads Helicone traces and extracts question-response pairs
- Uses `ReplayAgent` to replay historical responses (no new LLM calls for generation)
- Trains Reflector and SkillManager to learn from historical interactions
- **Automatic Opik observability tracking** for all operations
- Saves learned skillbook for future use

**Features:**
- 💰 **Cost-effective**: No LLM calls for generation (only Reflector/SkillManager)
- 📊 **Full observability**: Opik tracks which responses are replayed, coverage metrics
- 🔍 **Debug-friendly**: See exactly which questions are in your mapping vs. using defaults
- 🎯 **Production learning**: Train from real production data

**Usage:**
```bash
# 1. Place your Helicone data
cp your-helicone-export.json ../../.private/helicone/oneline.json

# 2. (Optional) Install observability
pip install ace-framework[observability]

# 3. Run the example
python offline_training_replay.py
```

**Opik Integration:**
When Opik is installed, you'll see:
- Replay operations tagged as `replay-generator` in traces
- Metadata showing question coverage (found vs. defaults)
- Comparison with regular Generator operations
- View at: https://www.comet.com/opik

This demonstrates **replay-based learning** where ACE learns from past interactions without generating new responses.

## Data Setup

**IMPORTANT:** Data files are NOT stored in this folder. They are kept in `.private/helicone/` to avoid committing sensitive data.

### Step 1: Export Helicone Data
1. Log into your Helicone dashboard
2. Navigate to your request logs
3. Export requests as JSON (single file or JSONL format)

### Step 2: Place Data Files
Place your exported files in the `.private/helicone/` directory:

```
agentic-context-engine/
├── .private/
│   └── helicone/
│       ├── oneline.json          # Your Helicone data here
│       └── other-traces.jsonl    # Additional data files
├── examples/
│   └── helicone/
│       ├── README.md             # This file
│       ├── helicone_loader.py
│       └── tool_selection_environment.py
```

### Step 3: Run Examples

```bash
# Test the loader
cd examples/helicone
python helicone_loader.py

# Test the environment
python tool_selection_environment.py
```

## Running ACE Training

See the main training script: `examples/helicone_data_ace_training.py`

```bash
cd examples
python helicone_data_ace_training.py
```

This script:
1. Loads Helicone traces using `helicone_loader.py`
2. Creates training samples
3. Runs ACE offline adaptation
4. Saves learned skillbook

## What ACE Learns

From Helicone data, ACE can learn:

- **Tool Selection Patterns**: Which tools work best in different contexts
- **Conversation Flow**: Optimal sequences of actions
- **Error Recovery**: How to handle tool failures
- **Efficiency**: Token-optimal approaches based on actual usage
- **User Preferences**: Patterns from successful interactions

## Customization

### Custom Evaluation Logic

Extend `ToolSelectionEnvironment` to add your own evaluation criteria:

```python
class CustomHeliconeEnvironment(ToolSelectionEnvironment):
    def evaluate(self, answer: str) -> EnvironmentResult:
        # Your custom evaluation logic
        pass
```

### Different Data Formats

If your Helicone export format differs, modify the parsing logic in `HeliconeLoader._parse_trace()`.

## Data Privacy

**Remember:**
- Helicone data may contain sensitive information (API keys, user data, etc.)
- Keep all data files in `.private/` which should be gitignored
- Never commit actual data files to the repository
- Review data before sharing or using for training

## Next Steps

1. Export your Helicone data
2. Place it in `.private/helicone/`
3. Run the loader to verify parsing works
4. Run the full ACE training pipeline
5. Analyze the learned skillbook

## Questions?

See the main project documentation in the repository root.
