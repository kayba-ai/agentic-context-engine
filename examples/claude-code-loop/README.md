# Claude Code Loop 🔄

![GitHub stars](https://img.shields.io/github/stars/kayba-ai/agentic-context-engine?style=social)
[![Discord](https://img.shields.io/discord/1429935408145236131?label=Discord&logo=discord&logoColor=white&color=5865F2)](https://discord.gg/mqCqH7sTyK)
[![Twitter Follow](https://img.shields.io/twitter/follow/kaybaai?style=social)](https://twitter.com/kaybaai)
[![PyPI version](https://badge.fury.io/py/ace-framework.svg)](https://badge.fury.io/py/ace-framework)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

**Claude Code that learns from itself**

Run Claude Code in a continuous loop. After each run, ACE (our open-source framework for agents that learn from execution feedback) analyzes what worked and what failed, then injects those learnings into the next iteration. Walk away and come back to finished work.

---

## ✨ Example Result

Used this to translate the ACE Python repo to TypeScript: [kayba-ai/ace-ts](https://github.com/kayba-ai/ace-ts)


| Metric           | Result                               |
| ---------------- | ------------------------------------ |
| ⏱️ Duration      | ~4 hours                             |
| 📝 Commits       | 119                                  |
| 📏 Lines written | ~14k                                 |
| ✅ Outcome       | Zero build errors, all tests passing |
| 💰 API cost      | ~$1.5 (Sonnet for learning)          |

---

## 🚀 Quick Start

Simple setup: clone the repo, add your API key, write a prompt, and run.

### 1. Clone

```bash
git clone https://github.com/kayba-ai/agentic-context-engine.git
cd agentic-context-engine/examples/claude-code-loop
```

### 2. Setup

```bash
# Add your ANTHROPIC_API_KEY to .env.ace
./setup.sh  # Initialize workspace
```

`reset_workspace.sh` copies `workspace_template/` to `workspace/` and initializes it as a git repo. Claude Code runs inside `workspace/` and is constrained to that directory. If you want to work on an existing codebase, put it in `workspace_template/` first.

### 3. Define Your Task

Edit `prompt.md` with your task (see [Prompt Tips](#-prompt-tips) for guidance).

### 4. Run

```bash
uv run python ace_loop.py
```

Claude Code starts working in `workspace/` and learned skills get stored in `skillbook/`.

You can stop anytime with `Ctrl+C` and resume later with `uv run python ace_loop.py` - it picks up where it left off. We recommend leaving it running until stall detection kicks in (no new commits for 4 iterations) or you're happy with the result.

### 5. Reset

Run this when starting a new task or trying a different prompt (workspace and skillbook get archived to logs).

```bash
./setup.sh
```

---

## 💳 What You Need

- **Claude Code:** Claude subscription (Max plan with Opus 4.5 recommended - tokens won't run out). On cheaper plans, if you hit your limit just resume later with `uv run python ace_loop.py`
- **Learning loop:** Anthropic API key (~$0.01-0.05 per iteration with Sonnet 4.5)

---

## 💡 Prompt Tips

**Example prompt that worked well:**

- **Task definition:** "Your job is to [task]" - describe what you want accomplished
- **Commit after edits:** "Make a commit after every single file edit" - enables stall detection (loop stops after 4 iterations with no commits)
- **.agent/ directory:** "Use .agent/ as scratchpad. Store long term plans and todos there" - Claude Code tracks its own progress
- **.env file:** "The .env file contains API keys" - add keys to `workspace_template/.env` if Claude Code needs them to test your task
- **Time allocation:** "Spend 80% on X, 20% on Y" - specify focus split to balance implementation and verification
- **Continuation:** "When done, improve code quality" - keeps the loop productive instead of stopping early

```markdown
Your job is to port ACE framework (Python) to TypeScript and maintain the repository.

Make a commit after every single file edit.

Use .agent/ directory as scratchpad for your work. Store long term plans and todo lists there.

The .env file contains API keys for running examples.

Spend 80% of time on porting, 20% on testing.

When porting is complete, improve code quality and fix any issues.
```

---

## 🔄 How It Works

```
Run → Reflect → Learn → Loop
 │       │         │       │
 │       │         │       └── Restart with learned skills
 │       │         └── SkillManager updates skillbook
 │       └── Reflector analyzes execution trace
 └── Claude Code executes prompt.md
```

Each iteration builds on previous work. Skills compound over time.

---

## 📁 Files

| File                  | What it does                              |
| --------------------- | ----------------------------------------- |
| `.env.ace`            | Your API key (edit this)                  |
| `prompt.md`           | Your task (edit this)                     |
| `ace_loop.py`         | Main loop script                          |
| `workspace_template/` | Your codebase + .env (copied on reset)    |
| `workspace/`          | Where Claude Code works                   |
| `.data/skillbooks/`   | Learned strategies (archived on reset)    |
| `setup.sh`            | Initialize/reset workspace                |

---

## ⚙️ Environment Variables

Set in `.env.ace` file:

| Variable            | Description                                                               |
| ------------------- | ------------------------------------------------------------------------- |
| `ANTHROPIC_API_KEY` | Required: API key for ACE learning model                                  |
| `AUTO_MODE`         | `true` (default) runs fully automatic, `false` prompts between iterations |
| `ACE_MODEL`         | Model for learning (default: claude-sonnet-4-5-20250929)                  |
