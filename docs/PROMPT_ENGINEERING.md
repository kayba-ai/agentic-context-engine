# ACE Prompt Engineering Guide

## State-of-the-Art Prompt Design Principles

This guide documents best practices for creating and optimizing ACE prompts, based on analysis of 80+ production prompts from leading AI systems (GPT-5, Claude 3.5, and others).

## Table of Contents

1. [Core Principles](#core-principles)
2. [Prompt Structure](#prompt-structure)
3. [Writing Effective Instructions](#writing-effective-instructions)
4. [Domain-Specific Optimizations](#domain-specific-optimizations)
5. [Common Anti-Patterns](#common-anti-patterns)
6. [Testing and Iteration](#testing-and-iteration)
7. [Migration from v1 to v2](#migration-from-v1-to-v2)

## Core Principles

### 1. Identity Headers Are Essential

**Always start prompts with clear identity and metadata:**

```python
# GOOD - Clear identity with version and capabilities
"""
# Identity and Metadata
You are ACE Agent v2.0, an expert problem-solving agent.
Prompt Version: 2.0.0
Current Date: {current_date}
Mode: Strategic Problem Solving
Confidence Threshold: 0.7
"""

# BAD - Vague introduction
"""
You are an assistant that helps solve problems.
"""
```

### 2. Hierarchical Organization

**Structure prompts with clear sections and subsections:**

```python
# GOOD - Hierarchical structure
"""
## Core Responsibilities
1. Analyze questions using skillbook
2. Apply relevant strategies

### Strategy Selection Protocol
- ONLY use bullets with confidence > 0.7
- NEVER apply conflicting strategies

### Output Requirements
Return JSON with exact schema...
"""

# BAD - Flat structure
"""
Analyze questions and apply strategies. Use bullets with high confidence.
Return JSON with your answer.
"""
```

### 3. Concrete Examples Over Abstract Principles

**Show exactly what you want with good/bad examples:**

```python
# GOOD - Concrete examples
"""
### Good Example:
{
  "reasoning": "1. Breaking down 15 × 24: This is multiplication. 2. Using decomposition: 15 × (20 + 4)...",
  "skill_ids": ["bullet_023"],
  "final_answer": "360"
}

### Bad Example (DO NOT DO THIS):
{
  "reasoning": "Using the skillbook strategies, the answer is clear.",
  "final_answer": "360"
}
"""

# BAD - Abstract guidance
"""
Provide clear reasoning and cite relevant bullets.
"""
```

### 4. Emphatic Capitalization for Critical Requirements

**Use MUST/NEVER/ALWAYS only for frequently violated rules:**

```python
# GOOD - Emphatic for critical requirements
"""
**MUST** include step-by-step reasoning
**NEVER** skip intermediate calculations
**ALWAYS** cite specific bullet IDs
"""

# BAD - Overuse dilutes effectiveness
"""
ALWAYS be helpful. NEVER make mistakes. MUST think carefully.
"""
```

## Prompt Structure

### Standard v2 Template

```python
TEMPLATE = """
# Identity and Metadata
[Role, version, capabilities, date]

## Core Mission
[One-sentence primary objective]

## Input Context
[Structured presentation of inputs]

## Processing Protocol
[Numbered steps or decision tree]

## Critical Requirements
**MUST**: [Absolute requirements]
**NEVER**: [Explicit prohibitions]

## Output Format
[Exact schema with examples]

## Error Recovery
[Fallback procedures]
"""
```

### Decision Trees for Complex Logic

```python
# GOOD - Clear conditional logic
"""
Execute in order - use FIRST that applies:

### 1. SUCCESS_CASE_DETECTED
IF prediction matches ground truth:
   → Tag strategies as helpful
   → Extract reusable patterns

### 2. CALCULATION_ERROR_DETECTED
IF mathematical error in reasoning:
   → Pinpoint error location
   → Identify root cause
"""
```

## Writing Effective Instructions

### 1. Specify Anti-Patterns Explicitly

```python
# GOOD - Explicit anti-patterns
"""
**NEVER** use these phrases:
- "Based on the skillbook"
- "The model was wrong"
- "Should have known better"
- "Obviously incorrect"
"""

# BAD - Vague guidance
"""
Avoid unhelpful phrases.
"""
```

### 2. Include Meta-Cognitive Instructions

```python
# GOOD - Self-assessment thresholds
"""
ONLY use strategies if confidence > 0.7
If uncertain about approach, state "low_confidence" in output
Verify each reasoning step before proceeding
"""
```

### 3. Procedural Workflows

```python
# GOOD - Numbered procedures
"""
## Solution Process
1. Classify problem type (arithmetic/algebra/geometry)
2. Select appropriate method based on classification
3. Apply method with ALL intermediate steps shown
4. Verify answer using alternative approach
5. Format output according to schema
"""
```

### 4. Completeness Requirements

```python
# GOOD - Prevent partial outputs
"""
**MUST** output COMPLETE code even if lengthy
**NEVER** use "..." or "rest remains the same"
**ALWAYS** include ALL import statements
"""
```

## Domain-Specific Optimizations

### Mathematics Prompts

```python
# Key additions for math domain:
"""
## Mathematical Protocols
- ALWAYS show intermediate steps
- VERIFY calculations twice
- Use standard order of operations (PEMDAS)
- State units in final answer
- Round only final result, not intermediate values
"""
```

### Code Generation Prompts

```python
# Key additions for code domain:
"""
## Code Requirements
- Write COMPLETE, runnable code
- Include error handling for edge cases
- Follow language idioms and style guides
- Add type hints where applicable
- Include basic test cases
"""
```

### Reasoning/Analysis Prompts

```python
# Key additions for reasoning:
"""
## Analytical Framework
- Identify assumptions explicitly
- Consider multiple perspectives
- Acknowledge uncertainty ranges
- Distinguish correlation from causation
- Cite evidence for claims
"""
```

## Common Anti-Patterns

### 1. Vague Instructions

```python
# BAD
"Be careful with your analysis"
"Think step by step"
"Consider all aspects"

# GOOD
"Verify arithmetic at each step using reverse operations"
"Follow this 5-step analysis procedure: [specific steps]"
"Address these specific aspects: [enumerated list]"
```

### 2. Over-Reliance on Training

```python
# BAD
"Use your knowledge to solve this"
"Apply appropriate methods"

# GOOD
"Use methods from the skillbook section 'algebra'"
"Apply the quadratic formula: x = (-b ± √(b²-4ac))/2a"
```

### 3. Missing Error Recovery

```python
# BAD
"Return JSON with the answer"

# GOOD
"""
Return JSON with exact schema.
If JSON generation fails:
1. Check all required fields present
2. Escape special characters
3. Validate number formats
Maximum retries: 3
"""
```

### 4. Ambiguous Output Format

```python
# BAD
"Respond with your analysis and conclusion"

# GOOD
"""
Return ONLY valid JSON:
{
  "analysis": "<numbered points>",
  "conclusion": "<one sentence>",
  "confidence": 0.0-1.0
}
"""
```

## Testing and Iteration

### 1. A/B Testing Framework

```python
from ace.prompts_v2_1 import PromptManager

# Test different versions
manager = PromptManager()

# Version A - Standard v2.1
prompt_a = manager.get_generator_prompt(version="2.1")

# Version B - Custom variant
prompt_b = custom_prompt_with_modifications

# Track performance
results_a = run_tests_with_prompt(prompt_a)
results_b = run_tests_with_prompt(prompt_b)

# Compare metrics
compare_accuracy(results_a, results_b)
compare_confidence_calibration(results_a, results_b)
```

### 2. Validation Utilities

```python
from ace.prompts_v2_1 import validate_prompt_output

# Test output compliance
output = llm.generate(prompt)
is_valid, errors = validate_prompt_output(output, role="generator")

if not is_valid:
    print(f"Output validation failed: {errors}")
    # Iterate on prompt to fix common failures
```

### 3. Performance Metrics

Track these metrics to evaluate prompt effectiveness:

- **Accuracy**: Correct answers / total
- **Compliance**: Valid JSON outputs / total
- **Confidence Calibration**: Correlation between confidence and accuracy
- **Retry Rate**: Failed attempts requiring retry
- **Token Efficiency**: Average tokens per response

### 4. Iterative Refinement Process

1. **Baseline**: Start with v2.1 template
2. **Observe**: Identify failure patterns
3. **Hypothesize**: Form specific improvements
4. **Test**: A/B test modifications
5. **Adopt**: Integrate successful changes
6. **Document**: Record what worked and why

## Migration to v2.1

### Quick Start

```python
# Old approach (v1)
from ace.prompts import AGENT_PROMPT
agent = Agent(llm, prompt_template=AGENT_PROMPT)

# Recommended approach (v2.1) - +17% success rate vs v1
from ace.prompts_v2_1 import PromptManager
manager = PromptManager(default_version="2.1")
agent = Agent(llm, prompt_template=manager.get_agent_prompt())
```

**Note:** v2.0 prompts are deprecated. Use v2.1 for best performance.

### Key Improvements

| Feature | v1 | v2.1 |
|---------|----|----|
| Performance | Baseline | +17% success rate |
| Structure | Basic sections | Hierarchical with metadata |
| Examples | None | Good/bad examples included |
| Error Handling | Basic JSON check | Detailed recovery procedures |
| Requirements | General guidance | MUST/NEVER with specifics |
| Output | Loose schema | Strict schema with validation |
| Domains | One-size-fits-all | Specialized variants |
| Anti-patterns | Not specified | Explicitly prohibited |
| Confidence | Not tracked | Built-in confidence scores |
| MCP Support | No | Yes (v2.1 enhancement) |

### Gradual Migration Strategy

1. **Phase 1**: Test v2 prompts with small sample
2. **Phase 2**: A/B test v1 vs v2 on real tasks
3. **Phase 3**: Migrate best-performing roles first
4. **Phase 4**: Customize v2 based on your needs
5. **Phase 5**: Fully migrate to v2 framework

## Best Practices Summary

### DO:
- ✅ Start with identity headers and metadata
- ✅ Use hierarchical organization with clear sections
- ✅ Provide concrete good/bad examples
- ✅ Specify exact output schemas
- ✅ Include error recovery procedures
- ✅ Add domain-specific optimizations
- ✅ List explicit anti-patterns to avoid
- ✅ Use emphatic caps sparingly for critical rules
- ✅ Include meta-cognitive assessment instructions
- ✅ Test and iterate based on failure patterns

### DON'T:
- ❌ Write vague, abstract instructions
- ❌ Rely solely on model training
- ❌ Overuse emphatic capitalization
- ❌ Skip error handling
- ❌ Use ambiguous output formats
- ❌ Ignore domain-specific needs
- ❌ Assume one prompt fits all cases
- ❌ Deploy without validation testing
- ❌ Mix multiple concerns in one section
- ❌ Forget to version your prompts

## Advanced Techniques

### 1. Prompt Chaining

```python
# Break complex tasks into stages
stage1_prompt = manager.get_generator_prompt(domain="analysis")
stage2_prompt = manager.get_generator_prompt(domain="synthesis")

# Chain outputs
analysis = generator_stage1.generate(prompt=stage1_prompt, ...)
synthesis = generator_stage2.generate(
    prompt=stage2_prompt,
    context=analysis.output,
    ...
)
```

### 2. Dynamic Prompt Selection

```python
def select_prompt_by_difficulty(question: str) -> str:
    """Select prompt variant based on problem complexity."""
    difficulty = assess_difficulty(question)

    if difficulty > 0.8:
        return manager.get_generator_prompt(variant="expert")
    elif difficulty > 0.5:
        return manager.get_generator_prompt(variant="standard")
    else:
        return manager.get_generator_prompt(variant="simple")
```

### 3. Prompt Versioning

```python
class VersionedPromptManager:
    """Track prompt performance across versions."""

    def __init__(self):
        self.versions = {}
        self.performance = {}

    def register_version(self, version: str, prompt: str):
        self.versions[version] = prompt
        self.performance[version] = {"uses": 0, "success": 0}

    def get_best_performing(self) -> str:
        """Return prompt with highest success rate."""
        best = max(
            self.performance.items(),
            key=lambda x: x[1]["success"] / max(x[1]["uses"], 1)
        )
        return self.versions[best[0]]
```

## Resources

- [Original ACE Paper](https://arxiv.org/abs/2510.04618)
- [Prompt Engineering Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Example Implementations](../examples/) - See examples directory for prompt engineering patterns

## Contributing

When contributing new prompts:

1. Follow the v2 template structure
2. Include at least 2 good/bad examples
3. Add domain-specific optimizations if applicable
4. Test with validation utilities
5. Document performance improvements
6. Submit with A/B test results if available

---

*This guide is based on analysis of production prompts from GPT-5, Claude 3.5, Grok, and 80+ other systems. It will be updated as new patterns emerge.*