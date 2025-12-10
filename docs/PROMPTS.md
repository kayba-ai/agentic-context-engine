# ACE Framework - Prompt Guide

Comprehensive guide to ACE prompt versions, customization, and best practices.

**Version:** 0.7.0

---

## Table of Contents

- [Prompt Versions](#prompt-versions)
- [Version Comparison](#version-comparison)
- [Template Variables](#template-variables)
- [Using Prompts](#using-prompts)
- [Migration Guide](#migration-guide)
- [Custom Prompts](#custom-prompts)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Prompt Versions

ACE provides three prompt versions with different characteristics:

### v1.0 (Simple) - `ace/prompts.py`

**Status:** Stable
**Use Case:** Tutorials, learning, simple tasks
**Lines:** ~150

```python
from ace.prompts import AGENT_PROMPT, REFLECTOR_PROMPT, SKILL_MANAGER_PROMPT

agent = Agent(llm, prompt_template=AGENT_PROMPT)
```

**Characteristics:**
- Simple, straightforward templates
- Minimal formatting and structure
- Lower token usage
- Good for understanding ACE fundamentals
- Works with weaker models (GPT-3.5)

### v2.0 (Advanced) - `ace/prompts_v2.py`

**Status:** DEPRECATED
**Use Case:** None - superseded by v2.1

```python
# Will emit DeprecationWarning
from ace.prompts_v2 import AGENT_V2_PROMPT  # Don't use
```

**Why Deprecated:** v2.1 includes all v2.0 features plus improvements. There's no reason to use v2.0.

### v2.1 (Recommended) - `ace/prompts_v2_1.py`

**Status:** Recommended for Production
**Use Case:** Production systems, best performance
**Lines:** ~1,500

```python
from ace.prompts_v2_1 import PromptManager

mgr = PromptManager()
agent = Agent(llm, prompt_template=mgr.get_agent_prompt())
reflector = Reflector(llm, prompt_template=mgr.get_reflector_prompt())
skill_manager = SkillManager(llm, prompt_template=mgr.get_skill_manager_prompt())
```

**Characteristics:**
- State-of-the-art prompt engineering
- MCP (Model Context Protocol) techniques
- Identity headers with metadata
- Hierarchical organization
- Meta-cognitive instructions
- Enhanced error handling
- Concrete examples and anti-patterns
- Optimized for Claude 3.5, GPT-4, similar models

---

## Version Comparison

### Quick Reference

| Feature | v1.0 | v2.0 (Deprecated) | v2.1 (Recommended) |
|---------|------|-------------------|-------------------|
| **Status** | Stable | Deprecated | **Recommended** |
| **Performance** | Baseline | +12% | **+17%** |
| **Token Usage** | Low (~850/call) | High (~1,200/call) | High (~1,200/call) |
| **JSON Parse Errors** | 8% | 3% | **1%** |
| **Use Case** | Tutorials | Don't use | **Production** |
| **MCP Support** | No | No | **Yes** |
| **Error Handling** | Basic | Enhanced | **Advanced** |
| **Examples Included** | No | Yes | **Yes** |

### Performance Benchmarks

Based on internal testing (200 samples, Claude 3.5 Sonnet):

| Metric | v1.0 | v2.1 | Improvement |
|--------|------|------|-------------|
| Success Rate | 72% | 89% | **+17%** |
| JSON Parse Errors | 8% | 1% | **-7%** |
| Quality Score | 7.2/10 | 9.1/10 | **+26%** |
| Avg Tokens/Call | 850 | 1,200 | +41% |

**Recommendation:** Use v2.1 for production. The token increase is worth the quality gain.

---

## Template Variables

### Agent Prompts

| Variable | Type | Description | Required |
|----------|------|-------------|----------|
| `{skillbook}` | str | Formatted skillbook (TOON format) | Yes |
| `{question}` | str | The question to answer | Yes |
| `{context}` | str | Additional context | No |
| `{reflection}` | str | Prior reflection for retry | No |

**Expected Output (JSON):**
```json
{
  "reasoning": "Step-by-step thinking process",
  "final_answer": "The actual answer",
  "skill_ids": ["reasoning-00001", "extraction-00003"]
}
```

### Reflector Prompts

| Variable | Type | Description | Required |
|----------|------|-------------|----------|
| `{skillbook}` | str | Cited skills excerpt | Yes |
| `{question}` | str | Original question | Yes |
| `{context}` | str | Additional context | No |
| `{agent_output}` | str | Agent's JSON output | Yes |
| `{feedback}` | str | Environment feedback | Yes |
| `{ground_truth}` | str | Expected answer | No |

**Expected Output (JSON):**
```json
{
  "reasoning": "Overall analysis",
  "error_identification": "What went wrong",
  "root_cause_analysis": "Why it went wrong",
  "correct_approach": "What should have been done",
  "key_insight": "Main lesson learned",
  "extracted_learnings": [...],
  "skill_tags": [
    {"id": "reasoning-00001", "tag": "helpful"},
    {"id": "extraction-00003", "tag": "harmful"}
  ]
}
```

### SkillManager Prompts

| Variable | Type | Description | Required |
|----------|------|-------------|----------|
| `{skillbook}` | str | Current skillbook state | Yes |
| `{reflection}` | str | Reflector's analysis | Yes |
| `{question_context}` | str | Task context | No |
| `{progress}` | str | Training progress | No |
| `{similarity_report}` | str | Dedup report (if enabled) | No |

**Expected Output (JSON):**
```json
{
  "reasoning": "Why these updates",
  "updates": [
    {"operation": "ADD", "section": "reasoning", "content": "New strategy"},
    {"operation": "UPDATE", "skill_id": "reasoning-00001", "content": "Updated"},
    {"operation": "TAG", "skill_id": "extraction-00003", "tag": "helpful", "increment": 1},
    {"operation": "REMOVE", "skill_id": "old-00042"}
  ],
  "consolidation_operations": [...]  // If deduplication enabled
}
```

---

## Using Prompts

### Default (v1.0)

When no prompt is specified, roles use v1.0:

```python
from ace import Agent, Reflector, SkillManager, LiteLLMClient

client = LiteLLMClient(model="gpt-4o-mini")

# Uses v1.0 by default
agent = Agent(client)
reflector = Reflector(client)
skill_manager = SkillManager(client)
```

### Using v2.1 (Recommended)

```python
from ace import Agent, Reflector, SkillManager, LiteLLMClient
from ace.prompts_v2_1 import PromptManager

client = LiteLLMClient(model="gpt-4o-mini")
mgr = PromptManager()

# Use v2.1 prompts
agent = Agent(client, prompt_template=mgr.get_agent_prompt())
reflector = Reflector(client, prompt_template=mgr.get_reflector_prompt())
skill_manager = SkillManager(client, prompt_template=mgr.get_skill_manager_prompt())
```

### With Integrations

Integrations use v1.0 by default. For v2.1:

```python
from ace import ACELiteLLM
from ace.prompts_v2_1 import PromptManager

# Note: High-level integrations manage prompts internally
# For custom prompts, use the full pipeline instead
```

---

## Migration Guide

### v1.0 → v2.1

**Step 1:** Update imports

```python
# Before (v1.0)
from ace.prompts import AGENT_PROMPT, REFLECTOR_PROMPT, SKILL_MANAGER_PROMPT

# After (v2.1)
from ace.prompts_v2_1 import PromptManager
mgr = PromptManager()
```

**Step 2:** Update role initialization

```python
# Before (v1.0)
agent = Agent(llm)  # Uses v1.0 default

# After (v2.1)
agent = Agent(llm, prompt_template=mgr.get_agent_prompt())
reflector = Reflector(llm, prompt_template=mgr.get_reflector_prompt())
skill_manager = SkillManager(llm, prompt_template=mgr.get_skill_manager_prompt())
```

**Step 3:** Test and validate

- Run your test suite
- Monitor JSON parse success rates
- Check output quality
- v2.1 is more reliable, may need fewer retries

### v2.0 → v2.1

Minimal changes:

```python
# Before (emits DeprecationWarning)
from ace.prompts_v2 import AGENT_V2_PROMPT

# After
from ace.prompts_v2_1 import PromptManager
mgr = PromptManager()
agent = Agent(llm, prompt_template=mgr.get_agent_prompt())
```

---

## Custom Prompts

### Creating Custom Prompts

You can create domain-specific prompts:

```python
CUSTOM_AGENT_PROMPT = """
You are a medical diagnosis assistant using ACE strategies.

# Available Strategies
{skillbook}

# Patient Question
{question}

# Medical Context
{context}

# Previous Reflection
{reflection}

## Instructions
- Review available strategies before answering
- Cite relevant strategies using [section-ID] format
- Follow medical best practices

## Output Format
Return ONLY valid JSON with these fields:
- "reasoning": Your diagnostic reasoning
- "final_answer": Your diagnosis and recommendations
- "skill_ids": Strategy IDs you used (e.g., ["med-00001", "diag-00003"])

Respond with JSON only, no other text.
"""

agent = Agent(llm, prompt_template=CUSTOM_AGENT_PROMPT)
```

### Template Requirements

**Required:**
- Include all placeholders: `{skillbook}`, `{question}`, `{context}`, `{reflection}`
- Specify JSON output format clearly
- List required fields: `reasoning`, `final_answer`, `skill_ids`

**Recommended:**
- Include citation instructions (`[section-ID]` format)
- Provide output examples
- Use clear section headers

**Avoid:**
- Hardcoded language-specific instructions
- Ambiguous output requirements
- Complex nested structures

### Testing Custom Prompts

```python
from ace import Agent, Skillbook, LiteLLMClient

# Setup
llm = LiteLLMClient(model="gpt-4o-mini")
skillbook = Skillbook()
skillbook.add_skill(section="test", content="Test strategy")

agent = Agent(llm, prompt_template=CUSTOM_AGENT_PROMPT)

# Test
output = agent.generate(
    question="Test question",
    context="Test context",
    skillbook=skillbook
)

# Validate
assert output.reasoning, "Missing reasoning"
assert output.final_answer, "Missing final_answer"
assert isinstance(output.skill_ids, list), "skill_ids must be list"
print("Custom prompt works!")
```

---

## Best Practices

### Do

- **Use v2.1 for production** - Better performance, fewer errors
- **Provide rich context** - More context = better answers
- **Test prompts thoroughly** - Validate with your specific domain
- **Monitor JSON parse rates** - Track success/failure
- **Include citation guidance** - Helps skill tracking

### Don't

- **Don't use v2.0** - It's deprecated
- **Don't skip testing** - Custom prompts need validation
- **Don't ignore parse errors** - Fix promptly or switch to v2.1
- **Don't hardcode languages** - Keep prompts flexible

### Token Optimization

If token cost is a concern:

```python
# Option 1: Use v1.0 for simple tasks
from ace.prompts import AGENT_PROMPT
agent = Agent(llm, prompt_template=AGENT_PROMPT)

# Option 2: Use a smaller skillbook
# Only include most relevant skills

# Option 3: Truncate context
context = context[:1000] if len(context) > 1000 else context
```

---

## Troubleshooting

### High JSON Parse Failure Rate

**Symptom:** Frequent `RuntimeError: Agent failed to produce valid JSON`

**Solutions:**
1. Upgrade to v2.1 prompts (most effective)
2. Increase `max_retries`: `Agent(llm, max_retries=5)`
3. Use more capable model (GPT-4, Claude 3.5)
4. Check for malformed prompt templates

### Empty Skill IDs

**Symptom:** `skill_ids` always empty `[]`

**Causes:**
- Skillbook is empty
- Agent not citing skills in reasoning
- Citation format not recognized

**Solutions:**
```python
# Check skillbook has skills
print(f"Skills: {len(skillbook.skills())}")

# Check skillbook format
print(skillbook.as_prompt())

# Ensure prompt includes citation instructions
```

### Poor Quality Answers

**Symptom:** Agent produces generic/unhelpful answers

**Solutions:**
1. Add more domain-specific skills
2. Provide richer context
3. Use v2.1 prompts for better reasoning
4. Try a more capable model
5. Check if skills are relevant to task

### Model-Specific Issues

**Claude models:**
```python
# Claude has temperature/top_p conflict
# LiteLLMClient handles this automatically
client = LiteLLMClient(
    model="claude-3-5-sonnet-20241022",
    temperature=0.0  # Will disable top_p automatically
)
```

**Ollama/local models:**
```python
# Local models may need more explicit JSON guidance
# Consider using Instructor wrapper
from ace.llm_providers.instructor_client import wrap_with_instructor

llm = wrap_with_instructor(LiteLLMClient(model="ollama/llama2"))
agent = Agent(llm)  # Auto-validates output
```

---

## Advanced Topics

### Domain-Specific Sections

Organize skills by domain:

```python
skillbook.add_skill(section="diagnosis", content="Check vital signs first")
skillbook.add_skill(section="treatment", content="Consider contraindications")
skillbook.add_skill(section="compliance", content="Verify HIPAA requirements")
```

### Multi-Language Support

v2.1 prompts work with non-English content:

```python
output = agent.generate(
    question="¿Cuál es la capital de Francia?",
    context="Responde en español",
    skillbook=skillbook
)
# Output will be in Spanish
```

### Prompt Versioning

For A/B testing:

```python
from ace.prompts import AGENT_PROMPT as V1_PROMPT
from ace.prompts_v2_1 import PromptManager

mgr = PromptManager()
V21_PROMPT = mgr.get_agent_prompt()

# Compare
agent_v1 = Agent(llm, prompt_template=V1_PROMPT)
agent_v21 = Agent(llm, prompt_template=V21_PROMPT)

# Run same task with both
output_v1 = agent_v1.generate(question, context, skillbook)
output_v21 = agent_v21.generate(question, context, skillbook)
```

---

## See Also

- [Complete Guide](COMPLETE_GUIDE_TO_ACE.md) - ACE concepts
- [API Reference](API_REFERENCE.md) - Complete API
- [Prompt Engineering](PROMPT_ENGINEERING.md) - Advanced techniques
- [Examples](../examples/prompts/) - Prompt examples

---

## References

- [ACE Paper (arXiv)](https://arxiv.org/abs/2510.04618)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [Instructor Library](https://github.com/jxnl/instructor)

---

**Last Updated:** December 2025 | **Version:** 0.7.0
