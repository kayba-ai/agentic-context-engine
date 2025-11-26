# Ollama Real-World Use Case Agents

Practical, production-ready agents using local Ollama models with ACE learning capabilities.

## Prerequisites

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Pull a capable model (recommend 7B+ for structured output)
ollama pull llama3.1:8b      # Best for complex tasks
ollama pull mistral:7b       # Good balance
ollama pull qwen2.5:7b       # Excellent for code

# 3. Verify Ollama is running
ollama list
```

## Available Agents

### Development & DevOps Agents

#### 1. 🔍 Code Review Agent (`code_review_agent.py`)
Reviews code for bugs, security issues, and best practices. Learns from feedback.

**Use Cases:**
- Pre-commit code quality checks
- Security vulnerability scanning
- Style and best practice enforcement
- Learning team-specific coding patterns

```bash
uv run python examples/ollama/code_review_agent.py
```

#### 2. 🧪 Test Case Generator (`test_generator_agent.py`)
Generates comprehensive unit tests from code. Learns testing patterns and edge cases.

**Use Cases:**
- Generate tests for legacy code
- Suggest missing test scenarios
- Learn project-specific test patterns
- Improve test quality over time

```bash
uv run python examples/ollama/test_generator_agent.py
```

#### 3. 🗄️ SQL Query Generator (`sql_query_agent.py`)
Generates SQL queries from natural language. Learns from execution results.

**Use Cases:**
- Natural language to SQL translation
- Query optimization suggestions
- Database schema understanding
- Learning business-specific query patterns

```bash
uv run python examples/ollama/sql_query_agent.py
```

#### 4. 📝 Git Commit Message Generator (`commit_message_agent.py`)
Generates conventional commit messages from diffs. Learns project conventions.

**Use Cases:**
- Enforce commit standards
- Generate release notes
- Improve commit history quality
- Learn from past commits

```bash
uv run python examples/ollama/commit_message_agent.py
```

#### 5. 🔧 Troubleshooting Assistant (`troubleshooting_agent.py`)
Diagnoses system issues and suggests fixes. Learns from resolution outcomes.

**Use Cases:**
- Log file analysis
- Error message interpretation
- System diagnostics
- Learning environment-specific issues

```bash
uv run python examples/ollama/troubleshooting_agent.py
```

### Operations & Support Agents

#### 6. 📧 Email/Ticket Classifier (`email_classifier_agent.py`)
Categorizes and routes support tickets. Learns routing rules and priorities.

**Use Cases:**
- Customer support automation
- Bug report triage
- Feature request categorization
- Email inbox organization

```bash
uv run python examples/ollama/email_classifier_agent.py
```

#### 7. 🐛 Bug Report Analyzer (`bug_report_agent.py`)
Parses bug reports and extracts structured information. Learns severity classification.

**Use Cases:**
- GitHub issue triage
- JIRA ticket validation
- Duplicate bug detection
- Severity/priority assignment

```bash
uv run python examples/ollama/bug_report_agent.py
```

### Data & Analytics Agents

#### 8. 📊 Data Analysis Agent (`data_analysis_agent.py`)
Analyzes CSV/JSON data and generates insights. Learns what types of analysis are valuable.

**Use Cases:**
- Automated data profiling
- Anomaly detection
- Business metrics analysis
- Learning domain-specific patterns

```bash
uv run python examples/ollama/data_analysis_agent.py
```

### Security Agents

#### 9. 🔐 Security Log Analyzer (`security_log_agent.py`)
Analyzes security logs and detects threats. Learns attack patterns and reduces false positives.

**Use Cases:**
- SIEM log analysis
- Intrusion detection
- Anomaly detection
- Automated incident response

```bash
uv run python examples/ollama/security_log_agent.py
```

### Documentation Agents

#### 10. 📝 Technical Writer Agent (`technical_writer_agent.py`)
Converts technical content between formats (code→docs, API→guide). Learns writing styles.

**Use Cases:**
- Generate README files from code
- Convert API specs to tutorials
- Create release notes from commits
- Learn company documentation style

```bash
uv run python examples/ollama/technical_writer_agent.py
```

## Model Recommendations

| Model | Size | Best For | Speed | Quality |
|-------|------|----------|-------|---------|
| **llama3.1:8b** | 4.7GB | General purpose, code | Fast | Excellent |
| **qwen2.5:7b** | 4.7GB | Code, technical tasks | Fast | Excellent |
| **mistral:7b** | 4.1GB | Reasoning, analysis | Very Fast | Good |
| **codellama:13b** | 7.4GB | Code-specific tasks | Medium | Excellent |
| **phi3:medium** | 7.9GB | Efficient reasoning | Fast | Good |

## Performance Tips

### 1. **Model Selection**
- **Small models (1-3B)**: Fast but struggle with structured JSON output
- **Medium models (7-8B)**: Good balance for most tasks
- **Large models (13B+)**: Best quality but slower

### 2. **Temperature Settings**
- **Code/SQL (0.1-0.3)**: More deterministic, better structure
- **Creative writing (0.5-0.7)**: More variety
- **Data analysis (0.3-0.5)**: Balance of accuracy and insight

### 3. **Context Management**
- Keep playbooks focused (50-100 bullets max)
- Tag bullets by domain for better retrieval
- Save checkpoints during long training sessions

### 4. **Hardware Requirements**
- **Minimum**: 8GB RAM for 7B models
- **Recommended**: 16GB RAM for 13B models
- **GPU**: Optional but significantly faster (CUDA/Metal)

## Integration Patterns

### Pattern 1: Offline Training (Batch Learning)
```python
from ace import ACELiteLLM, Sample, SimpleEnvironment

agent = ACELiteLLM(model="ollama/llama3.1:8b")

# Prepare training samples
samples = [Sample(question=q, ground_truth=a) for q, a in training_data]

# Learn in batches
results = agent.learn(samples, SimpleEnvironment(), epochs=2)

# Save learned knowledge
agent.save_playbook("learned_strategies.json")
```

### Pattern 2: Online Learning (Continuous Improvement)
```python
from ace import ACELiteLLM

agent = ACELiteLLM(
    model="ollama/llama3.1:8b",
    is_learning=True,
    playbook_path="strategies.json"  # Auto-save
)

# Agent learns from each interaction
for task in tasks:
    result = agent.ask(task.question)
    # Automatically learns from feedback
```

### Pattern 3: Human-in-the-Loop
```python
from ace import ACELiteLLM

agent = ACELiteLLM(model="ollama/llama3.1:8b", is_learning=True)

# Interactive learning loop
while True:
    user_input = input("Query: ")
    answer = agent.ask(user_input)
    print(f"Answer: {answer}")

    feedback = input("Was this helpful? (y/n): ")
    # Feedback automatically incorporated through ACE learning
```

## Customization

### Custom Task Environment
```python
from ace import TaskEnvironment, EnvironmentResult

class CustomEvaluator(TaskEnvironment):
    def evaluate(self, sample, answer):
        # Your custom evaluation logic
        success = your_validation_logic(answer)
        feedback = "Good" if success else "Needs improvement"

        return EnvironmentResult(
            success=success,
            feedback=feedback,
            ground_truth=sample.ground_truth
        )
```

### Custom Prompts
```python
from ace.prompts_v2_1 import PromptManager

# Customize role prompts
custom_prompts = PromptManager()
agent = ACELiteLLM(
    model="ollama/llama3.1:8b",
    prompt_manager=custom_prompts
)
```

## Troubleshooting

### Issue: JSON Parsing Errors
**Cause**: Small models struggle with structured output
**Solution**:
- Use 7B+ models (llama3.1:8b, mistral:7b)
- Lower temperature (0.1-0.3)
- Enable retry prompts

### Issue: Slow Performance
**Cause**: Model too large or CPU-only inference
**Solution**:
- Use smaller model (mistral:7b instead of 13b)
- Enable GPU acceleration (Ollama auto-detects)
- Reduce max_tokens parameter

### Issue: Poor Learning Results
**Cause**: Insufficient training data or feedback
**Solution**:
- Provide 10+ diverse training samples
- Use clear, specific ground truth answers
- Run multiple epochs (2-3)

### Issue: Playbook Gets Too Large
**Cause**: Accumulating too many strategies
**Solution**:
- Periodically prune low-scoring bullets
- Use tagging to organize by domain
- Start fresh playbook for new domains

## Best Practices

1. **Start Simple**: Test with small datasets before scaling
2. **Version Playbooks**: Save dated versions (`playbook_2024-11-24.json`)
3. **Monitor Quality**: Track success rates over time
4. **Domain Separation**: Use different playbooks for different tasks
5. **Regular Pruning**: Remove outdated or harmful strategies
6. **Feedback Loops**: Incorporate user/system feedback consistently

## Examples Output

Each example includes:
- ✅ Before/after learning comparison
- 📊 Success metrics and improvements
- 💾 Saved playbook with learned strategies
- 🔄 Reusable for future sessions

## Contributing

Add your own use case agent:
1. Copy existing example as template
2. Implement custom TaskEnvironment for your domain
3. Add training samples relevant to your use case
4. Update this README with your example
5. Submit PR!

## Resources

- [Ollama Models](https://ollama.ai/library)
- [ACE Framework Docs](../../docs/COMPLETE_GUIDE_TO_ACE.md)
- [LiteLLM Ollama Docs](https://docs.litellm.ai/docs/providers/ollama)
