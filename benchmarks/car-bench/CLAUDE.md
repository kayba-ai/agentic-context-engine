# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This directory (`benchmarks/car-bench/`) contains the CAR-bench integration:

- **`car-bench/`** — CAR-bench benchmark (cloned from GitHub) for evaluating LLM agent reliability
- **`convert_traces.py`** — Extracts `traj` from CAR-bench result JSONs into individual JSON files for ACE
- **`traces/`** — Output directory for extracted trace files

The goal is to use ACE to learn from CAR-bench train-set agent traces and improve agent performance on the test set.

## ACE + CAR-bench Integration Pipeline

### Pipeline Overview

1. Run CAR-bench on **train split** → raw result JSON files in `car-bench/results/`
2. **Extract traces** via `convert_traces.py` (pulls `traj` + `task_id` only) → JSON files in `traces/`
3. Convert JSON traces to TOON via `../../examples/agentic-system-prompting/convert.py`, then run ACE → generates skillbook
4. Inject learned skills into agent system prompt (`car-bench/car_bench/envs/car_voice_assistant/wiki.md`)
5. Run CAR-bench on **test split** with enhanced prompt → compare Pass^k against baseline

### Solution Leakage Prevention (CRITICAL)

The raw result JSON files saved by `run.py` contain ground truth that MUST NOT reach ACE:

**What's in the raw result JSON:**

```
result.traj                              → SAFE: agent's conversation messages
result.reward                            → LEAKY: reveals correctness
result.info.task.actions                 → LEAKY: ground truth action sequence
result.info.reward_info.r_*              → LEAKY: per-dimension reward scores
result.info.reward_info.tool_subset_missing_tools → LEAKY: names required tools
result.info.reward_info.policy_*_errors  → LEAKY: names exact violations
result.info.task.removed_part            → LEAKY: hallucination mechanism
result.info.task.disambiguation_element_* → LEAKY: disambiguation answers
```

**Extraction rules (implemented in `convert_traces.py`):**

- Extract `traj` field and `task_id` — nothing else
- The `traj` is already clean: it only contains what the agent saw during execution (system prompt, user messages, assistant messages, tool calls/results)
- Ground truth (`info.task.actions`, `reward_info`, etc.) is never part of `traj` — it's benchmark metadata added after the run for scoring
- Run: `python convert_traces.py car-bench/results/base_train/model.json -o traces/`

**Result JSON Structure (saved by run.py at runtime)**

Each result file is a JSON array of entries:

```json
{
  "task_index": 0,
  "task_id": "base_0",
  "reward": 0.85,
  "trial": 0,
  "traj": [
    {"role": "system", "content": "...wiki..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "tool_calls": [{"function": {"name": "...", "arguments": "..."}, "id": "...", "type": "function"}]},
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "{...}"}
  ],
  "info": {
    "task": { "actions": [...], "removed_part": [...], ... },
    "reward_info": { "r_actions_final": ..., "tool_subset_missing_tools": [...], ... },
    "total_agent_cost": 0.008,
    "total_llm_induced_latency_ms": 10920.0
  }
}
```

The `traj` field is the safe part (agent's lived experience). Everything in `info` contains evaluation metadata including ground truth.

### Key File Locations

| File                                                                                   | Purpose                                                                        |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `car-bench/car_bench/envs/car_voice_assistant/wiki.md`                                 | Agent system prompt (191 lines, 19 policies) — injection target for ACE skills |
| `car-bench/car_bench/envs/car_voice_assistant/wiki.py`                                 | Loads wiki.md, extracts policy tags (LLM-POL, AUT-POL)                         |
| `car-bench/car_bench/orchestrator.py`                                                  | Drives agent-env loop; passes wiki as system prompt (line ~138)                |
| `car-bench/car_bench/types.py`                                                         | `EnvRunResult`, `Task`, `Action` — result/task data structures                 |
| `car-bench/run.py`                                                                     | Main entry point; saves results incrementally (line ~354)                      |
| `car-bench/results/`                                                                   | Raw result JSON files (contain ground truth — do not feed directly to ACE)     |
| `convert_traces.py`                                                                    | Extracts `traj` + `task_id` from result JSONs into individual files            |
| `traces/`                                                                              | Extracted trace files (clean, no ground truth)                                 |
| `../../examples/agentic-system-prompting/agentic_system_prompting.py`                  | ACE CLI — processes `.md`/`.toon` trace files from a directory                 |
| `../../examples/agentic-system-prompting/convert.py`                                   | JSON → TOON format converter                                                   |

### System Prompt Injection

Modify wiki.md directly: Append ACE skills to the end of `car-bench/car_bench/envs/car_voice_assistant/wiki.md`.

### Train/Test Split

- Even-indexed task IDs = train, odd-indexed = test (across all task types)
- **Base**: 50 train / 50 test
- **Hallucination**: 48 train / 50 test
- **Disambiguation**: 31 train / 25 test
- Splits defined in `car-bench/car_bench/envs/car_voice_assistant/tasks/task_splits.json`

### Cost Reference

Full train set (129 tasks, 1 trial, Haiku agent + Gemini Flash user/evaluator):

- Anthropic (Haiku agent): ~$2.85
- Gemini (user sim + policy eval): negligible
- Concurrency 3, ~15 min

### Cross-Task-Type Leakage Note

Hallucination and disambiguation tasks are derived from base tasks (same scenarios with tools/params removed or ambiguity introduced). A train hallucination task may share a scenario with a test base task. This is acceptable as long as ground truth is stripped — ACE learns from conversation patterns, not answer keys.

---

## CAR-bench (`car-bench/`)

### Project Overview

CAR-bench is a benchmark for evaluating epistemic reliability of multi-turn, tool-using LLM agents. It tests whether agents know when they can act, when they must gather more info, and when they should refuse/defer. The domain is an automotive in-car voice assistant with 58 tools (27 set, 29 get, 2 no-op), 19 policies, and 254 tasks across three types: base (100), hallucination (98), disambiguation (56).
