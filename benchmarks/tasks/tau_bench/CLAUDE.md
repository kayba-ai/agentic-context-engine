# TAU-bench Evaluation Guide

TAU-bench (tau2) evaluates tool-calling agents in customer service domains using multi-turn conversations and database state assertions. Official leaderboard: https://tau-bench.github.io

Use `/benchmark <config> [mode] [extra-args]` for quick runs (e.g. `/benchmark haiku`, `/benchmark fast`).

## Quick Start

```bash
# Baseline run (no ACE)
uv run python scripts/run_tau_benchmark.py --config sonnet --skip-ace

# Compare baseline vs ACE
uv run python scripts/run_tau_benchmark.py --config sonnet --compare

# Quick smoke test (3 tasks, k=1)
uv run python scripts/run_tau_benchmark.py --config fast --skip-ace
```

## Config Profiles

All configs live in `benchmarks/tasks/tau_bench/` and inherit from `default.yaml`.

| Profile | Model | Use Case |
|---------|-------|----------|
| `default` | gpt-4.1-mini | Official leaderboard defaults |
| `sonnet` | claude-sonnet-4-5 | Claude Sonnet evaluation |
| `haiku` | claude-haiku-4-5 | Claude Haiku evaluation |
| `gpt4.1-mini` | gpt-4.1-mini | Explicit GPT-4.1-mini |
| `gpt4.1` | gpt-4.1 | GPT-4.1 (stronger) |
| `fast` | gpt-4.1-mini | Quick iteration (3 tasks, k=1) |

CLI args override config values: `--config sonnet --domain retail --k 2`

## Best Practices

- **User simulator must be `gpt-4.1-2025-04-14`** — wrong model gives ~50% lower scores.
- **max_steps=200** — lower values cause tasks to fail early.
- **Temperature 0**, **seed 300**, **k=4** for reproducibility matching the leaderboard.
- Task splits: train=30, test=20 for airline. Use `test` split to match leaderboard.
- Always run `--skip-ace` first to establish a baseline before testing ACE.
- Use `--compare` for side-by-side baseline vs ACE in one run.
- Use `--save-detailed` for per-task trial-level data (debugging).
- ACE trains on `train` split, evaluates on `test` split (automatic with `--compare`).
- Use `--skillbook path/to/skillbook.json` to evaluate a pre-trained skillbook without re-training.
- `--batch-reflect` defers learning until all training tasks complete.

## Presenting Results

Always include these fields: exact model ID (e.g. `claude-sonnet-4-5-20250929`), user LLM, domain, split + task count, skillbook status, training info (or "none"), and all pass^k metrics. Only include max steps / seed if non-default (200 / 300).

Use a settings table + pass^k table. For comparisons, add Baseline / ACE / Delta columns.

## Output Files

Results saved to `tau_benchmark_results/`:
```
tau_{domain}_{config}_{phase}_{timestamp}_summary.json
tau_{domain}_{config}_{phase}_{timestamp}_detailed.json   (with --save-detailed)
tau_{domain}_{config}_{phase}_{timestamp}_skillbook.json   (if skills learned)
```

The summary JSON contains all configuration and metrics needed to reproduce the result.
