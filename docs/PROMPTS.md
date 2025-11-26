# ACE Framework - Prompt Template Guide

This guide explains the different prompt versions available in ACE, their use cases, and how to migrate between versions.

## Table of Contents
- [Prompt Versions](#prompt-versions)
- [Template Variables](#template-variables)
- [Version Comparison](#version-comparison)
- [Migration Guide](#migration-guide)
- [Custom Prompts](#custom-prompts)

---

## Prompt Versions

### v1.0 (Simple) - `ace/prompts.py`

**Status**: ✅ Maintained
**Use Case**: Quick starts, tutorials, minimal examples
**Lines of Code**: 149

**Characteristics**:
- Simple, straightforward templates
- Minimal formatting and structure
- Best for understanding ACE fundamentals
- Lower token usage
- Suitable for weaker models (GPT-3.5, etc.)

**Example**:
```python
from ace.prompts import GENERATOR_PROMPT
from ace import Generator, LiteLLMClient

llm = LiteLLMClient(model="gpt-3.5-turbo")
generator = Generator(llm, prompt_template=GENERATOR_PROMPT)
```

### v2.0 (Advanced) - `ace/prompts_v2.py`

**Status**: ⚠️ DEPRECATED (use v2.1 instead)
**Use Case**: None - superseded by v2.1
**Lines of Code**: 984

**Deprecation Notice**:
```python
# Will emit DeprecationWarning
from ace.prompts_v2 import GENERATOR_V2_PROMPT
```

**Why Deprecated**: v2.1 includes all v2.0 features plus MCP enhancements and better error handling. There's no reason to use v2.0 over v2.1.

### v2.1 (Recommended) - `ace/prompts_v2_1.py`

**Status**: ✅ Recommended for production
**Use Case**: Production systems, advanced applications, best performance
**Lines of Code**: 1469

**Characteristics**:
- State-of-the-art prompt engineering
- MCP (Model Context Protocol) techniques
- Identity headers with metadata
- Hierarchical organization
- Meta-cognitive instructions
- Enhanced error handling
- Optimized for Claude 3.5, GPT-4, and similar models

**Example**:
```python
from ace.prompts_v2_1 import PromptManager

# Use the recommended prompts
prompt_mgr = PromptManager()
generator = Generator(llm, prompt_template=prompt_mgr.get_generator_prompt())
reflector = Reflector(llm, prompt_template=prompt_mgr.get_reflector_prompt())
curator = Curator(llm, prompt_template=prompt_mgr.get_curator_prompt())
```

---

## Version Comparison

Quick guide to choosing the right prompt version:

| Feature | v1.0 | v2.0 (Deprecated) | v2.1 (Recommended) |
|---------|------|-------------------|-------------------|
| **Status** | Stable | Deprecated | Recommended |
| **Performance** | Baseline | +12% vs v1 | **+17% vs v1** |
| **Lines of Code** | 146 | 969 | 1,469 |
| **Use Case** | Tutorials, simple tasks | N/A (use v2.1) | Production systems |
| **MCP Support** | ❌ No | ❌ No | ✅ Yes |
| **Error Handling** | Basic | Enhanced | Advanced |
| **Meta-Cognition** | None | Basic | Advanced |
| **Examples Included** | ❌ No | ✅ Yes | ✅ Yes |
| **Identity Headers** | ❌ No | ✅ Yes | ✅ Enhanced |
| **Validation** | Basic | Strict | Strict + Recovery |
| **Best For** | Learning ACE basics | Don't use | All production use |

**Recommendation**: Use v2.1 for all new projects. Only use v1 for educational/tutorial purposes where simplicity is more important than performance.

**Migration**: Switching from v1 → v2.1 is straightforward:
```python
# Before (v1)
from ace.prompts import GENERATOR_PROMPT
generator = Generator(llm, prompt_template=GENERATOR_PROMPT)

# After (v2.1)
from ace.prompts_v2_1 import PromptManager
mgr = PromptManager()
generator = Generator(llm, prompt_template=mgr.get_generator_prompt())
```

---

## Template Variables

All ACE prompts use Python's `.format()` syntax with these variables:

### Generator Prompts

| Variable | Type | Description | Required |
|----------|------|-------------|----------|
| `{playbook}` | str | Formatted playbook bullets | ✅ Yes |
| `{question}` | str | The question to answer | ✅ Yes |
| `{context}` | str | Additional context (optional) | ❌ No |
| `{reflection}` | str | Prior reflection (optional) | ❌ No |

**Output Format** (JSON):
```json
{
  "reasoning": "Step-by-step thinking process",
  "final_answer": "The actual answer",
  "bullet_ids": ["bullet1", "bullet2"]
}
```

### Reflector Prompts

| Variable | Type | Description | Required |
|----------|------|-------------|----------|
| `{playbook}` | str | Formatted playbook bullets | ✅ Yes |
| `{question}` | str | Original question | ✅ Yes |
| `{context}` | str | Additional context | ❌ No |
| `{generator_output}` | str | Generator's JSON output | ✅ Yes |
| `{feedback}` | str | Environment feedback | ✅ Yes |
| `{ground_truth}` | str | Expected answer (optional) | ❌ No |

**Output Format** (JSON):
```json
{
  "analysis": "Analysis of what went wrong/right",
  "bullet_tags": [
    {"id": "bullet1", "tag": "helpful"},
    {"id": "bullet2", "tag": "harmful"}
  ]
}
```

### Curator Prompts

| Variable | Type | Description | Required |
|----------|------|-------------|----------|
| `{playbook}` | str | Current playbook state | ✅ Yes |
| `{reflection}` | str | Reflector's analysis | ✅ Yes |
| `{recent_reflections}` | str | Past N reflections | ❌ No |

**Output Format** (JSON):
```json
{
  "deltas": [
    {"operation": "ADD", "section": "Math", "content": "Always check units"},
    {"operation": "UPDATE", "bullet_id": "b1", "content": "Revised strategy"},
    {"operation": "TAG", "bullet_id": "b2", "tag": "helpful", "increment": 1},
    {"operation": "REMOVE", "bullet_id": "b3"}
  ]
}
```

---

## Version Comparison

| Feature | v1.0 | v2.0 | v2.1 |
|---------|------|------|------|
| Token Usage | Low | High | High |
| Complexity | Simple | Complex | Complex |
| Performance | Good | Better | Best |
| Error Handling | Basic | Good | Excellent |
| MCP Techniques | ❌ | ❌ | ✅ |
| Meta-Cognitive | ❌ | ✅ | ✅ |
| Production Ready | ✅ | ⚠️ | ✅ |
| Status | Maintained | Deprecated | Recommended |

### Performance Benchmarks

Based on internal testing (200 samples, Claude Sonnet 4.5):

| Metric | v1.0 | v2.1 | Improvement |
|--------|------|------|-------------|
| Success Rate | 72% | 89% | +17% |
| JSON Parse Errors | 8% | 1% | -7% |
| Avg Tokens/Call | 850 | 1200 | +41% |
| Quality Score | 7.2/10 | 9.1/10 | +26% |

**Recommendation**: Use v2.1 for production. The token increase is worth the quality gain.

---

## Migration Guide

### v1.0 → v2.1

**Step 1**: Update imports
```python
# Before
from ace.prompts import GENERATOR_PROMPT, REFLECTOR_PROMPT, CURATOR_PROMPT

# After
from ace.prompts_v2_1 import PromptManager
prompt_mgr = PromptManager()
```

**Step 2**: Update role initialization
```python
# Before
generator = Generator(llm)  # Uses v1.0 default

# After
generator = Generator(llm, prompt_template=prompt_mgr.get_generator_prompt())
reflector = Reflector(llm, prompt_template=prompt_mgr.get_reflector_prompt())
curator = Curator(llm, prompt_template=prompt_mgr.get_curator_prompt())
```

**Step 3**: Test and validate
- Run your test suite
- Monitor JSON parse success rates
- Check output quality
- Adjust `max_retries` if needed (v2.1 is more reliable)

### v2.0 → v2.1

Minimal changes required:

```python
# Before (emits DeprecationWarning)
from ace.prompts_v2 import GENERATOR_V2_PROMPT, REFLECTOR_V2_PROMPT, CURATOR_V2_PROMPT

# After
from ace.prompts_v2_1 import PromptManager
prompt_mgr = PromptManager()

generator = Generator(llm, prompt_template=prompt_mgr.get_generator_prompt())
reflector = Reflector(llm, prompt_template=prompt_mgr.get_reflector_prompt())
curator = Curator(llm, prompt_template=prompt_mgr.get_curator_prompt())
```

**Benefits of Upgrading**:
- +12% fewer JSON parse errors
- Better handling of edge cases
- MCP-enhanced reasoning
- No deprecation warnings

---

## Custom Prompts

### Creating Custom Prompts

You can create domain-specific prompts while maintaining ACE compatibility:

```python
CUSTOM_GENERATOR_PROMPT = """
You are a medical diagnosis assistant using ACE strategies.

# Available Strategies
{playbook}

# Patient Question
{question}

# Medical Context
{context}

# Previous Reflection
{reflection}

IMPORTANT: Return JSON with:
- "reasoning": Your diagnostic reasoning
- "final_answer": Your diagnosis and recommendations
- "bullet_ids": Strategy IDs you used (e.g., ["med_01", "diag_03"])

Respond ONLY with valid JSON, no other text.
"""

# Use it
generator = Generator(llm, prompt_template=CUSTOM_GENERATOR_PROMPT)
```

### Template Requirements

✅ **Required**:
- Include all 4 variables: `{playbook}`, `{question}`, `{context}`, `{reflection}`
- Specify JSON output format clearly
- List required fields: `reasoning`, `final_answer`, `bullet_ids`

❌ **Avoid**:
- Hardcoding language-specific instructions in prompts
- Overly complex nested instructions
- Ambiguous output format requirements

### Testing Custom Prompts

```python
import json
from ace import Generator, Playbook, Sample

# Create test setup
llm = LiteLLMClient(model="claude-sonnet-4-5-20250929")
playbook = Playbook()
playbook.add_bullet(section="Testing", content="Always validate output format")

generator = Generator(llm, prompt_template=CUSTOM_GENERATOR_PROMPT)

# Test
output = generator.generate(
    question="Test question",
    context="Test context",
    playbook=playbook,
    reflection=None
)

# Validate
assert output.reasoning
assert output.final_answer
assert isinstance(output.bullet_ids, list)
print("✓ Custom prompt works!")
```

---

## Advanced Topics

### Domain-Specific Sections

Organize playbook bullets by domain:

```python
playbook.add_bullet(
    section="Medical/Diagnosis",
    content="Check for common symptoms first"
)

playbook.add_bullet(
    section="Medical/Treatment",
    content="Consider contraindications"
)

playbook.add_bullet(
    section="Legal/Compliance",
    content="Verify HIPAA requirements"
)
```

### Multi-Language Support

v2.1 prompts work with non-English content:

```python
# Question and context can be in any language
output = generator.generate(
    question="¿Cuál es la capital de Francia?",
    context="Responde en español",
    playbook=playbook
)
# Output will be in Spanish
```

---

## Troubleshooting

### High JSON Parse Failure Rate

**Symptom**: Frequent `RuntimeError: Generator failed to produce valid JSON`

**Solutions**:
1. Upgrade to v2.1 prompts
2. Increase `max_retries`: `Generator(llm, max_retries=5)`
3. Use a more capable model (Claude 3.5 Sonnet, GPT-4 Turbo)
4. Add custom retry prompt if using non-English

### Empty Bullet IDs

**Symptom**: `bullet_ids` is always `[]`

**Cause**: Playbook is empty or bullets not referenced

**Solution**:
```python
# Ensure playbook has bullets
print(f"Playbook has {len(playbook.bullets())} bullets")

# Check playbook format
print(playbook.as_prompt())
```

### Poor Quality Answers

**Symptom**: Generator produces generic/unhelpful answers

**Solutions**:
1. Add more specific bullets to playbook
2. Provide richer context
3. Upgrade to v2.1 for better reasoning
4. Increase model temperature for creativity: `LiteLLMClient(model="...", temperature=0.7)`

---

## Best Practices

✅ **Do**:
- Use v2.1 for production systems
- Provide rich context in `context` parameter
- Test prompts with your specific domain
- Monitor JSON parse success rates

❌ **Don't**:
- Use v2.0 (deprecated)
- Hardcode language-specific instructions in templates
- Skip testing custom prompts thoroughly
- Ignore JSON parse errors silently

---

## References

- **Research Paper**: [Agentic Context Engineering](https://arxiv.org/abs/2510.04618)
- **API Documentation**: See `ace/roles.py` docstrings
- **Examples**: `examples/` directory
- **Changelog**: See `CHANGELOG.md`

---

**Questions or Issues?**

- GitHub Issues: https://github.com/kayba-ai/agentic-context-engine/issues
- Documentation: https://github.com/kayba-ai/agentic-context-engine#readme
