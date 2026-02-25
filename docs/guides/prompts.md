# Prompt Engineering

ACE uses specialized prompt templates for each role. The framework includes multiple prompt versions with different trade-offs.

## Default Prompts

`ace_next` ships with v2.1 prompts built in. All three roles (`Agent`, `Reflector`, `SkillManager`) use them by default — no extra imports needed.

!!! tip "Recommendation"
    The built-in v2.1 prompts work well out of the box. Only provide custom prompts when you need domain-specific instructions.

## Overriding Prompts

Pass a `prompt_template` string to any role constructor:

```python
from ace_next import Agent, Reflector, SkillManager, LiteLLMClient

llm = LiteLLMClient(model="gpt-4o-mini")

agent = Agent(llm, prompt_template="Your custom agent prompt ...")
reflector = Reflector(llm, prompt_template="Your custom reflector prompt ...")
skill_manager = SkillManager(llm, prompt_template="Your custom skill manager prompt ...")
```

## Template Variables

### Agent Prompt

| Variable | Description |
|----------|-------------|
| `{skillbook}` | Current skillbook in TOON format |
| `{question}` | The input question |
| `{context}` | Additional context |
| `{reflection}` | Optional reflection from a previous attempt |

### Reflector Prompt

| Variable | Description |
|----------|-------------|
| `{skillbook}` | Current skillbook in TOON format |
| `{question}` | The original question |
| `{agent_output}` | The agent's response |
| `{ground_truth}` | Expected answer |
| `{feedback}` | Environment feedback |

### SkillManager Prompt

| Variable | Description |
|----------|-------------|
| `{skillbook}` | Current skillbook in TOON format |
| `{reflection}` | Reflector's analysis |
| `{question_context}` | Description of the task domain |
| `{progress}` | Current training progress |

## Custom Prompts

You can provide your own prompt templates. They must include the required template variables:

```python
custom_agent_prompt = """
Skillbook: {skillbook}
Question: {question}
Context: {context}

Generate a JSON response with:
- reasoning: Your step-by-step thought process
- skill_ids: List of skillbook IDs you used
- final_answer: Your answer
"""

agent = Agent(llm, prompt_template=custom_agent_prompt)
```

## Formatting Skillbook for External Agents

Integration runners inject the skillbook into external agent prompts using a wrapper function:

```python
from ace_next import wrap_skillbook_context

context = wrap_skillbook_context(skillbook)
# Returns formatted strategies with usage instructions
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| JSON parse failures | Increase `max_tokens`, use Instructor, or try v2.1 prompts |
| Empty skill_ids | Agent not citing skills — check skillbook has content |
| Poor answer quality | Switch to v2.1 prompts or try a larger model |

## What to Read Next

- [Full Pipeline Guide](full-pipeline.md) — use prompts in a complete pipeline
- [The Skillbook](../concepts/skillbook.md) — what goes into `{skillbook}`
