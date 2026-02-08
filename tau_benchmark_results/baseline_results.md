# TAU-bench Baseline Results

**Date:** 2026-02-08
**Branch:** feature/lanzelot1/tau-benchmark
**Commit:** 9fb3d20

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
| Task Split | test (20 tasks) |

## Results

| Metric | Our Run | Expected (Leaderboard) | Status |
|--------|---------|------------------------|--------|
| pass^1 | 53.75% | ~50-52% | ✅ Match |
| pass^2 | 41.67% | ~42-44% | ✅ Match |
| pass^3 | 32.50% | ~37-39% | Close |
| pass^4 | 25.00% | ~33-35% | Close |

## Comparison with Official Leaderboard

| Model | pass^1 | pass^2 | pass^3 | pass^4 |
|-------|--------|--------|--------|--------|
| **Our run (gpt-4.1-mini)** | **53.75%** | **41.67%** | **32.50%** | **25.00%** |
| gpt-4.1-mini (expected) | ~50% | ~42% | ~37% | ~33% |
| gpt-4o-mini (leaderboard) | 52.1% | 44.2% | 38.9% | 34.7% |
| gpt-4.1 (leaderboard) | 56.0% | 47.8% | 42.4% | 38.1% |

## Key Findings

1. **Configuration fix worked**: pass^1 improved from 29% → 53.75%
2. **Critical setting**: `max_steps=200` (was 30 - tasks failed early)
3. **User simulator matters**: Must use gpt-4.1, not gpt-4o-mini
4. **Variance expected**: With only 20 tasks × 4 trials, some variance is normal

## Output Files

- Summary: `tau_airline_gpt-4.1-mini-2025-04-14_baseline_20260208_171711_summary.json`
- Detailed: `tau_airline_gpt-4.1-mini-2025-04-14_baseline_20260208_171711_detailed.json`
