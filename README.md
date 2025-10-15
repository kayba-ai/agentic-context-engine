# Agentic Context Engine (ACE) 🚀

**Build self-improving AI agents that learn from experience**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-arXiv:2510.04618-red.svg)](https://arxiv.org/abs/2510.04618)

🧠 **ACE** is a framework for building AI agents that get smarter over time by learning from their mistakes and successes.

💡 Based on the paper "Agentic Context Engineering" from Stanford/SambaNova - ACE helps your LLM agents build a "playbook" of strategies that improves with each task.

🔌 **Works with any LLM** - OpenAI, Anthropic Claude, Google Gemini, and 100+ more providers out of the box!

## Quick Start

**Minimum Python 3.9 required**

### Install ACE:
```bash
pip install -r requirements.txt
```

### Set up your API key:
```bash
# Copy the example environment file
cp .env.example .env

# Add your OpenAI key (or Anthropic, Google, etc.)
echo "OPENAI_API_KEY=your-key-here" >> .env
```

### Run your first agent:
```python
from ace import LiteLLMClient, OfflineAdapter, Generator, Reflector, Curator
from ace import Playbook, Sample, TaskEnvironment, EnvironmentResult
from dotenv import load_dotenv

load_dotenv()

# Create your agent with any LLM
client = LiteLLMClient(model="gpt-3.5-turbo")  # or claude-3, gemini-pro, etc.

# Set up ACE components
adapter = OfflineAdapter(
    playbook=Playbook(),
    generator=Generator(client),
    reflector=Reflector(client),
    curator=Curator(client)
)

# Define a simple task
class SimpleEnv(TaskEnvironment):
    def evaluate(self, sample, output):
        correct = sample.ground_truth.lower() in output.final_answer.lower()
        return EnvironmentResult(
            feedback="Correct!" if correct else "Try again",
            ground_truth=sample.ground_truth
        )

# Train your agent
samples = [
    Sample(question="What is 2+2?", ground_truth="4"),
    Sample(question="Capital of France?", ground_truth="Paris"),
]

results = adapter.run(samples, SimpleEnv(), epochs=1)
print(f"Agent learned {len(adapter.playbook.bullets())} strategies!")
```

## How It Works

ACE uses three AI "roles" that work together to help your agent improve:

1. **🎯 Generator** - Tries to solve tasks using the current playbook
2. **🔍 Reflector** - Analyzes what went wrong (or right)
3. **📝 Curator** - Updates the playbook with new strategies

Think of it like a sports team reviewing game footage to get better!

## Examples

### Simple Q&A Agent
```python
python examples/simple_ace_example.py
```

### Advanced Examples with Different LLMs
```python
python examples/quickstart_litellm.py
```

Check out the `examples/` folder for more!

## Supported LLM Providers

ACE works with **100+ LLM providers** through LiteLLM:

- **OpenAI** - GPT-4, GPT-3.5-turbo
- **Anthropic** - Claude 3 (Opus, Sonnet, Haiku)
- **Google** - Gemini Pro, PaLM
- **Cohere** - Command models
- **Local Models** - Ollama, Transformers
- **And many more!**

Just change the model name:
```python
# OpenAI
client = LiteLLMClient(model="gpt-4")

# Anthropic Claude
client = LiteLLMClient(model="claude-3-sonnet-20240229")

# Google Gemini
client = LiteLLMClient(model="gemini-pro")

# With fallbacks for reliability
client = LiteLLMClient(
    model="gpt-4",
    fallbacks=["claude-3-haiku", "gpt-3.5-turbo"]
)
```

## Key Features

- ✅ **Self-Improving** - Agents learn from experience and build knowledge
- ✅ **Provider Agnostic** - Switch LLMs with one line of code
- ✅ **Production Ready** - Automatic retries, fallbacks, and error handling
- ✅ **Cost Efficient** - Track costs and use cheaper models as fallbacks
- ✅ **Async Support** - Built for high-performance applications
- ✅ **Fully Typed** - Great IDE support and type safety

## Advanced Usage

### Online Learning (Learn While Running)
```python
from ace import OnlineAdapter

# Agent improves while processing real tasks
adapter = OnlineAdapter(
    playbook=existing_playbook,  # Can start with existing knowledge
    generator=Generator(client),
    reflector=Reflector(client),
    curator=Curator(client)
)

# Process tasks one by one, learning from each
for task in real_world_tasks:
    result = adapter.process(task, environment)
    # Agent automatically updates its strategies
```

### Custom Task Environments
```python
class CodeTestingEnv(TaskEnvironment):
    def evaluate(self, sample, output):
        # Run the generated code
        test_passed = run_tests(output.final_answer)

        return EnvironmentResult(
            feedback=f"Tests {'passed' if test_passed else 'failed'}",
            ground_truth=sample.ground_truth,
            metrics={"pass_rate": 1.0 if test_passed else 0.0}
        )
```

### Streaming Responses
```python
# Get responses token by token
for chunk in client.complete_with_stream("Write a story"):
    print(chunk, end="", flush=True)
```

### Async Operations
```python
import asyncio

async def main():
    response = await client.acomplete("Solve this problem...")
    print(response.text)

asyncio.run(main())
```

## Architecture

ACE implements the Agentic Context Engineering method from the research paper:

- **Playbook**: A structured memory that stores successful strategies
- **Bullets**: Individual strategies with helpful/harmful counters
- **Delta Operations**: Incremental updates that preserve knowledge
- **Three Roles**: Generator, Reflector, and Curator working together

The framework prevents "context collapse" - a common problem where agents forget important information over time.

## Repository Structure

```
ace/
├── ace/                    # Core library
│   ├── playbook.py        # Strategy storage
│   ├── roles.py           # Generator, Reflector, Curator
│   ├── adaptation.py      # Training loops
│   └── llm_providers/     # LLM integrations
├── examples/              # Ready-to-run examples
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## Contributing

We welcome contributions! Feel free to:
- 🐛 Report bugs
- 💡 Suggest features
- 🔧 Submit PRs
- 📚 Improve documentation

## Citation

If you use ACE in your research or project, please cite the original papers:

### ACE Paper (Primary Reference)
```bibtex
@article{zhang2024ace,
  title={Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models},
  author={Zhang, Qizheng and Hu, Changran and Upasani, Shubhangi and Ma, Boyuan and Hong, Fenglu and
          Kamanuru, Vamsidhar and Rainton, Jay and Wu, Chen and Ji, Mengmeng and Li, Hanchen and
          Thakker, Urmish and Zou, James and Olukotun, Kunle},
  journal={arXiv preprint arXiv:2510.04618},
  year={2024}
}
```

### Dynamic Cheatsheet (Foundation Work)
ACE builds upon the adaptive memory concepts from Dynamic Cheatsheet:

```bibtex
@article{suzgun2025dynamiccheatsheet,
  title={Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory},
  author={Suzgun, Mirac and Yuksekgonul, Mert and Bianchi, Federico and Jurafsky, Dan and Zou, James},
  year={2025},
  eprint={2504.07952},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2504.07952}
}
```

### This Implementation
If you use this specific implementation, you can also reference:

```
This repository: https://github.com/yourusername/agentic-context-engine
Based on the open reproduction at: https://github.com/sci-m-wang/ACE-open
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Note**: This is an independent implementation based on the ACE paper (arXiv:2510.04618) and builds upon concepts from Dynamic Cheatsheet. For the original reproduction scaffold, see [sci-m-wang/ACE-open](https://github.com/sci-m-wang/ACE-open).

Made with ❤️ by [Kayba](https://kayba.ai) and the open-source community