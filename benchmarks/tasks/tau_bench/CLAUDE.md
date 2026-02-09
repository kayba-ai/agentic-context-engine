# TAU-bench Evaluation Guide

TAU-bench (tau2) evaluates tool-calling agents in customer service domains using multi-turn conversations and database state assertions. Official leaderboard: https://tau-bench.github.io

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

### Configuration
- **User simulator must be `gpt-4.1-2025-04-14`** (not gpt-4o-mini). Wrong model gives ~50% lower scores.
- **max_steps must be 200**. Lower values (e.g. 30) cause tasks to fail early.
- **Temperature 0** for reproducibility.
- **k=4** matches the official leaderboard (pass^1 through pass^4).
- **Seed 300** for reproducible runs.

### Running Evaluations
- Always run `--skip-ace` first to establish a baseline before testing ACE.
- Use `--compare` to get side-by-side baseline vs ACE in one run.
- Use `--save-detailed` to capture per-task trial-level data for debugging.
- Task splits: train=30, test=20 for airline. Use `test` split to match leaderboard.

### ACE-Specific
- ACE trains on the `train` split and evaluates on the `test` split (automatic with `--compare`).
- Use `--skillbook path/to/skillbook.json` to evaluate a pre-trained skillbook without re-training.
- `--batch-reflect` defers learning until all training tasks complete, then reflects on all traces together.
- ACE currently shows mixed results on TAU-bench: modest gains for weaker models, possible degradation for stronger models due to instruction dilution with the detailed domain policy.

## Presenting Results

When reporting results, always include the full run configuration header and pass^k table.

### Single Run (Baseline)

```
## Baseline: Sonnet 4.5 — Airline (test split, k=4)

| Setting | Value |
|---------|-------|
| Model | claude-sonnet-4-5-20250929 |
| User LLM | gpt-4.1-2025-04-14 |
| Domain | airline |
| Split | test (20 tasks) |
| Max steps | 200 |
| Seed | 300 |
| Skillbook | none |

| Metric | Score |
|--------|-------|
| pass^1 | 66.25% |
| pass^2 | 59.17% |
| pass^3 | 53.75% |
| pass^4 | 50.00% |
```

### Comparison Run (Baseline vs ACE)

```
## Comparison: GPT-4.1-mini — Airline (test split, k=4)

| Setting | Value |
|---------|-------|
| Model | gpt-4.1-mini-2025-04-14 |
| User LLM | gpt-4.1-2025-04-14 |
| Domain | airline |
| Split | test (20 tasks) |
| Training | 30 train tasks, 1 epoch, recursive reflector |
| Skillbook | learned (94 skills) |

| Metric | Baseline | ACE | Delta |
|--------|----------|-----|-------|
| pass^1 | 53.75% | 58.75% | +5.00% |
| pass^2 | 41.67% | 41.67% | 0.00% |
| pass^3 | 32.50% | 31.25% | -1.25% |
| pass^4 | 25.00% | 25.00% | 0.00% |
```

### Pre-trained Skillbook Evaluation

```
## Skillbook Eval: Sonnet 4.5 + Consolidated Playbook — Airline (test split, k=4)

| Setting | Value |
|---------|-------|
| Model | claude-sonnet-4-5-20250929 |
| User LLM | gpt-4.1-2025-04-14 |
| Domain | airline |
| Split | test (20 tasks) |
| Skillbook | consolidated_airline_skillbook.json (20 skills, 123 helpful votes) |
| Training | none (pre-trained) |

| Metric | Baseline | Enhanced | Delta |
|--------|----------|----------|-------|
| pass^1 | 66.25% | 60.00% | -6.25% |
| pass^2 | 59.17% | 47.50% | -11.67% |
| pass^3 | 53.75% | 40.00% | -13.75% |
| pass^4 | 50.00% | 35.00% | -15.00% |
```

### Key Fields to Always Include

- **Model**: Exact model ID (e.g. `claude-sonnet-4-5-20250929`, not just "Sonnet")
- **User LLM**: The simulator model — results are not comparable across different user LLMs
- **Domain**: airline / retail / telecom
- **Split + task count**: e.g. "test (20 tasks)"
- **Skillbook**: `none`, `learned (N skills)`, or filename with skill count
- **Training**: epochs, reflector mode, batch vs sequential — or "none" for baseline/pre-trained
- **Max steps / seed**: only if non-default (200 / 300)

## Output Files

Results are saved to `tau_benchmark_results/` with naming:
```
tau_{domain}_{model}_{phase}_{timestamp}_summary.json
tau_{domain}_{model}_{phase}_{timestamp}_detailed.json   (with --save-detailed)
tau_{domain}_{model}_{phase}_{timestamp}_skillbook.json   (if skills learned)
```

The summary JSON contains all configuration and metrics needed to reproduce the result.
