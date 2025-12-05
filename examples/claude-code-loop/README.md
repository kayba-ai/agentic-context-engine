# Claude Code Loop

**Claude Code that learns from itself 🔄**

Run Claude Code in a continuous loop. After each run, ACE analyzes what worked and what failed, then injects those learnings into the next iteration. Walk away and come back to finished work.

---

## ✨ Example Result

Used this to translate the ACE Python repo to TypeScript:

| Metric           | Result                               |
| ---------------- | ------------------------------------ |
| ⏱️ Duration      | ~4 hours                             |
| 📝 Commits       | 119                                  |
| 📏 Lines written | ~14k                                 |
| ✅ Outcome       | Zero build errors, all tests passing |
| 💰 API cost      | ~$1.5 (Sonnet for learning)          |

---

## 🚀 Quick Start

### 1. Install

```bash
pip install ace-framework
```

### 2. Setup

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
./reset_workspace.sh  # Initialize workspace
```

### 3. Define Your Task

Edit `prompt.md` with your task (see [Prompt Tips](#-prompt-tips) for guidance).

### 4. Run

```bash
python ace_loop.py
```

Walk away. The loop runs until stall detection kicks in (no new commits for 4 iterations).

### 5. Reset

Run this when starting a new task or trying a different prompt (workspace and skillbook get archived to logs).

```bash
./reset_workspace.sh
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

| File                 | What it does                           |
| -------------------- | -------------------------------------- |
| `prompt.md`          | Your task (edit this)                  |
| `ace_loop.py`        | Main loop script                       |
| `workspace/`         | Where Claude Code works                |
| `.data/skillbooks/`  | Learned strategies (archived on reset) |
| `reset_workspace.sh` | Initialize/reset workspace             |

---

## ⚙️ Environment Variables

Set in `.env` file:

| Variable    | Description                                                               |
| ----------- | ------------------------------------------------------------------------- |
| `AUTO_MODE` | `true` (default) runs fully automatic, `false` prompts between iterations |
| `ACE_MODEL` | Model for learning (default: claude-sonnet-4-5)                           |

---

## 💰 Cost

- **Claude Code:** Your Claude subscription (Opus 4.5 on Max Subscription recommended)
- **Learning loop:** ~$0.01-0.05 per iteration (Sonnet 4.5 recommended)

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
