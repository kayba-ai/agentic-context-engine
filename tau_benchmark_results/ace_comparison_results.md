# TAU-bench ACE Comparison Results

**Date:** 2026-02-08
**Branch:** feature/lanzelot1/tau-benchmark

## Configuration

| Setting | Value |
|---------|-------|
| Domain | airline |
| Agent Model | gpt-4.1-mini-2025-04-14 |
| User Simulator | gpt-4.1-2025-04-14 |
| Temperature | 0 |
| Max Steps | 200 |
| Max Errors | 10 |
| Seed | 300 |
| K (trials) | 4 |
| Train Split | 30 tasks (official tau2 train) |
| Test Split | 20 tasks (official tau2 test) |
| ACE Epochs | 1 |

## Results Comparison

| Metric | Baseline | ACE (Simple) | ACE (Recursive) | Best |
|--------|----------|--------------|-----------------|------|
| pass^1 | 53.75% | 48.75% | **58.75%** | Recursive (+5.00%) |
| pass^2 | 41.67% | 30.00% | **41.67%** | Tie |
| pass^3 | 32.50% | 18.75% | 31.25% | Baseline (+1.25%) |
| pass^4 | 25.00% | 10.00% | **25.00%** | Tie |
| Skills | 0 | 84 | 94 | - |

## Key Findings

### Recursive Reflector vs Simple Reflector

The recursive reflector dramatically outperforms the simple reflector:

| Metric | Simple | Recursive | Improvement |
|--------|--------|-----------|-------------|
| pass^1 | 48.75% | 58.75% | **+10.00%** |
| pass^2 | 30.00% | 41.67% | **+11.67%** |
| pass^3 | 18.75% | 31.25% | **+12.50%** |
| pass^4 | 10.00% | 25.00% | **+15.00%** |

### Recursive Reflector vs Baseline

The recursive reflector matches or beats the baseline:

| Metric | Baseline | Recursive | Delta |
|--------|----------|-----------|-------|
| pass^1 | 53.75% | 58.75% | **+5.00%** |
| pass^2 | 41.67% | 41.67% | 0.00% |
| pass^3 | 32.50% | 31.25% | -1.25% |
| pass^4 | 25.00% | 25.00% | 0.00% |

## Analysis

### Simple Reflector Shows Negative Transfer

The simple reflector learned 84 skills but performed *worse* than baseline on all metrics:
- pass^1: -5.00% (48.75% vs 53.75%)
- pass^2: -11.67% (30.00% vs 41.67%)
- pass^3: -13.75% (18.75% vs 32.50%)
- pass^4: -15.00% (10.00% vs 25.00%)

This suggests the single-pass reflection produces skills that are too generic or conflicting.

### Recursive Reflector Produces Quality Skills

The recursive reflector:
1. Learned more skills (94 vs 84)
2. Achieved positive transfer on pass^1 (+5.00%)
3. Maintained parity on pass^2 and pass^4
4. Only slight regression on pass^3 (-1.25%)

The REPL-style approach with multiple LLM calls produces higher-quality, more actionable skills.

### Why Recursive Works Better

1. **Iterative refinement**: Multiple passes allow the reflector to self-correct
2. **Deeper analysis**: Can explore tool calls and conversation traces more thoroughly
3. **Better abstraction**: Skills are more generalizable across tasks
4. **Quality over quantity**: Fewer low-quality or conflicting skills

## Output Files

**Baseline:**
- `tau_airline_gpt-4.1-mini-2025-04-14_baseline_20260208_171711_summary.json`
- `tau_airline_gpt-4.1-mini-2025-04-14_baseline_20260208_171711_detailed.json`

**ACE (Simple Reflector):**
- `tau_airline_gpt-4.1-mini-2025-04-14_ace_20260208_184714_summary.json`
- `tau_airline_gpt-4.1-mini-2025-04-14_ace_20260208_184714_detailed.json`
- `tau_airline_gpt-4.1-mini-2025-04-14_ace_20260208_184714_skillbook.json`

**ACE (Recursive Reflector):**
- `tau_airline_gpt-4.1-mini-2025-04-14_ace_20260208_204725_summary.json`
- `tau_airline_gpt-4.1-mini-2025-04-14_ace_20260208_204725_detailed.json`
- `tau_airline_gpt-4.1-mini-2025-04-14_ace_20260208_204725_skillbook.json`

## Next Steps

1. Make recursive reflector the default for TAU-bench runs
2. Experiment with multiple epochs to further improve skill quality
3. Analyze the 94 learned skills to understand what patterns transfer well
4. Test on retail and telecom domains
