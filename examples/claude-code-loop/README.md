# ACE TypeScript Translation (Integration Pattern)

Translate ACE from Python to TypeScript using ACE's **Integration Pattern** - the same approach used by browser-use and other external agent integrations.

## Key Innovation: Integration Pattern

This project uses ACE's **Integration Pattern** where Claude Code acts as the agent:

```
Task → [Claude Code Agent] → Execution Result
         ↑                           ↓
     Playbook ← [Curator] ← [Reflector] ← Feedback
     (context)  (updates)   (analyzes)
```

**How it works:**

1. **INJECT**: Playbook strategies added to Claude Code's prompt
2. **EXECUTE**: Claude Code runs with learned strategies as context
3. **LEARN**: Reflector analyzes only Claude Code's output (no bias)

**Cost per task:** ~$0.03 (Reflector + Curator via Claude Sonnet 4.5) + Claude Subscription

## Quick Start

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 2. Initialize workspace (clones ACE source from GitHub)
./reset_workspace.sh

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Run ACE learning loop (interactive mode - confirm each task)
python ace_loop.py

# OR: Run in fully automatic mode
export AUTO_MODE=true
python ace_loop.py
```

**📖 See [QUICK_START.md](QUICK_START.md) for detailed setup and monitoring guide**
**⚙️ See [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for technical details**

## How It Works

```
CONTINUOUS LEARNING LOOP (Integration Pattern):

Task 1: Bootstrap (create TODO.md)
  ↓
INJECT: wrap_playbook_context(playbook) → Claude Code prompt
  ↓
EXECUTE: Claude Code runs with strategies as context
  ↓
LEARN: Reflector analyzes Claude Code's output
  - Sees only: task description + Claude Code's actual output
  - Does NOT see: full playbook with helpful counts
  ↓
Curator updates playbook based on what worked
  ↓
Read TODO.md for next task → Task 2 (with improved playbook)
  ↓
INJECT: Updated playbook context → Claude Code
  ↓
... repeat for all ~25 tasks ...
  ↓
Complete TypeScript translation with 20-30 learned strategies
```

### Interactive vs Automatic Mode

**Interactive (default)**: Confirm each task

```bash
python ace_loop.py
# After each task:
# "▶️  Process this task? (y/n/q):"
```

**Automatic**: No confirmations, runs all tasks

```bash
export AUTO_MODE=true
python ace_loop.py
# Processes all tasks automatically
# Cost: ~$1 API for full translation (+ Claude subscription)
```

## Architecture

### Integration Pattern Components

**1. INJECT - Playbook Context**

- Uses `wrap_playbook_context(playbook)` from ACE's integration module
- Formats strategies WITHOUT introducing bias
- Added to Claude Code's prompt as context

**2. EXECUTE - Claude Code Agent**

- Claude Code receives playbook strategies as context
- Selects and applies relevant strategies autonomously
- Cites strategies in output (e.g., "Applied Strategies: [task_management-00001]")

**3. LEARN - Reflector & Curator**

- **Reflector** receives minimal GeneratorOutput:
  ```python
  GeneratorOutput(
      reasoning="Task: {question}",
      final_answer=claude_code_output,  # Just the output
      bullet_ids=[],  # External agents don't pre-select
      raw={"success": True}
  )
  ```
- **NO playbook with helpful counts** - eliminates bias
- **Curator** updates playbook based on what worked

## Project Structure

```
claude-code-loop/
├── ace_loop.py                     # Main orchestrator
├── claude_code_environment.py      # Environment using Integration Pattern
├── execution_trace_parser.py       # Parses stream-json from Claude Code
├── reset_workspace.sh              # Cleans workspace, clones source from GitHub
├── .env.example                    # Environment template
├── workspace/
│   ├── source/                     # Cloned by reset_workspace.sh (not in repo)
│   ├── specs/
│   │   ├── project.md              # Translation specification
│   │   └── rules.md                # TypeScript coding standards
│   └── target/                     # TypeScript output (created at runtime)
└── playbooks/                      # Learned strategies (created at runtime)
```

## What Gets Built

A complete TypeScript port of ACE:

```
workspace/target/
├── src/
│   ├── playbook.ts                 # From ace/playbook.py
│   ├── delta.ts                    # From ace/delta.py
│   ├── llm.ts                      # From ace/llm.py
│   ├── roles.ts                    # From ace/roles.py
│   ├── adaptation.ts               # From ace/adaptation.py
│   └── prompts-v2-1.ts             # From ace/prompts_v2_1.py
├── tests/
├── package.json
└── tsconfig.json
```

## Cost Estimate

Using Claude Sonnet 4.5:

- **Claude Code:** Uses your Claude Code subscription (no API cost)
- **Reflector:** ~$0.02 per task (API)
- **Curator:** ~$0.01 per task (API)
- **Total:** ~$0.03 per task (API only)

For ~25-30 tasks: **~$0.75-$0.90 API cost** (plus subscription usage)

## Success Criteria

Translation complete when:

- ✅ All TypeScript compiles (`npm run build`)
- ✅ All tests pass (`npm test`)
- ✅ Strict TypeScript mode (no `any`)
- ✅ Playbook has 20-30 translation strategies

## Files

- **workspace/source/**: Python ACE source (cloned by `reset_workspace.sh`)
- **workspace/specs/**: Translation specs for Claude Code
- **workspace/target/**: TypeScript output (created at runtime)
- **workspace/.agent/**: TODO.md and logs (managed by Claude Code)
- **playbooks/**: Learned strategies (created at runtime)

## Next Steps

1. Run `python ace_loop.py` to start (interactive mode)
2. ACE will create TODO.md with translation tasks
3. **After task 1**: Loop reads TODO.md and prompts for task 2
4. Each task uses improved playbook from previous tasks
5. Check `playbooks/ace_typescript.json` to see learned strategies grow
6. When comfortable, switch to AUTO_MODE for hands-free completion

## Comparison: Integration Pattern vs Old Approach

| Old Approach                              | Integration Pattern                      |
| ----------------------------------------- | ---------------------------------------- |
| PassthroughGenerator passes full playbook | No Generator - Claude Code is the agent  |
| Reflector sees all strategies with counts | Reflector sees only Claude Code's output |
| Cognitive bias from helpful counts        | No bias - clean feedback                 |
| Custom implementation                     | Follows ACE best practices               |
| $0.08/task                                | ~$0.03/task API (+ subscription)         |

## Why This Works

**Integration Pattern benefits:**

- **No bias**: Reflector never sees the full playbook with helpful counts that could anchor it toward generic strategies
- **Better learning**: Reflector focuses on actual execution outcomes, not pre-selected strategy lists
- **Cleaner architecture**: Follows the same pattern as ACE's browser-use integration
- **Claude Code autonomy**: Acts as a true agent, selecting strategies based on task context

**Claude Code has sufficient context** to select relevant strategies:

- It knows what it's doing (translating Python file X)
- It sees the current codebase state
- It understands TypeScript patterns
- Strategies are provided as context, not rigid instructions

**Result:** ACE learns from real execution feedback, not biased by seeing which strategies have high helpful counts!

## References

- **ACE Integration Pattern**: `workspace/source/ace/integrations/base.py`
- **Browser-use Example**: `workspace/source/examples/browser-use/`
- **ACE Documentation**: `workspace/source/docs/INTEGRATION_GUIDE.md`
