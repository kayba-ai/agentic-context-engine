# Results Directory Index

All experimental results for the skillbook variance study. The master narrative and analysis live in [`/research/skillbook-variance-study.md`](../research/skillbook-variance-study.md).

## Directories

| Directory | Description |
|-----------|-------------|
| `variance_experiment_haiku_4.5/` | Primary dataset: 35 runs (7 budgets x 5 reps), Opus compressions, consensus skillbooks |
| `variance_experiment_sonnet_4.6/` | Secondary dataset: 35 runs, Opus compressions, consensus skillbooks |
| `variance_experiment_gpt5mini/` | Partial GPT-5-mini runs (abandoned) |
| `variance_experiment_gpt5nano/` | GPT-5-nano config only (never run) |
| `single_run_comparison/` | Three-way comparison: old ACE vs ace_next no-budget vs budget-4000 |
| `tau_bench_eval/` | TAU-bench pass^4 downstream evaluation: 17 configs x 20 tasks |
| `car_bench_eval/` | CAR-bench downstream evaluation: 13 configs x 125 tasks x 4 trials, 3 task types (base/hallucination/disambiguation) |
| `variance_experiment_car_haiku_4.5/` | CAR-bench generation: 10 runs (2 budgets x 5 reps), Opus compressions, consensus skillbooks |
| `variance_experiment_car_sonnet_4.6/` | CAR-bench generation: 10 runs (2 budgets x 5 reps), Opus compressions, consensus skillbooks |
| `traces_25/` | Symlinks to 25 TAU-bench airline traces used as input |
| `traces_car_bench/` | Extracted CAR-bench training traces (129 tasks) |

## Files

| File | Description |
|------|-------------|
| `skillbook_analysis.ipynb` | Interactive Jupyter notebook for cross-model analysis |
