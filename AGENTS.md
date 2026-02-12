# AGENTS.md

Agent role descriptions and configuration for the ACE framework.

## ACE Roles

The framework uses three collaborative roles that share the same base LLM:

| Role | Responsibility | Key Class |
|------|---------------|-----------|
| **Agent** | Executes tasks using skillbook strategies | `Agent` |
| **Reflector** | Analyzes execution results (what worked, what failed) | `Reflector` |
| **SkillManager** | Updates the skillbook with new strategies | `SkillManager` |

## Integration Agents

Pre-built wrappers for external frameworks:

| Agent | Framework | Use Case |
|-------|-----------|----------|
| `ACELiteLLM` | LiteLLM (100+ providers) | Simple self-improving agent |
| `ACELangChain` | LangChain | Wrap chains/agents with learning |
| `ACEAgent` | browser-use | Browser automation with learning |
| `ACEClaudeCode` | Claude Code CLI | Coding tasks with learning |

## Agent-Specific Notes

- `ace-learn` writes learned guidance to `CLAUDE.md` and `.ace/skillbook.json` at the project root.
- When working with Claude Code, keep `CLAUDE.md` up to date and avoid editing generated skillbook files by hand.
- See `ace/integrations/` for integration source code.
- See `CLAUDE.md` for full development instructions and coding guidelines.
