# Benchmark Scripts

Scripts for running ACE benchmarks and analyzing results.

## Scripts

- `run_benchmark.py` - CLI to run ACE benchmarks with train/test splits
- `analyze_ace_results.py` - Analyze benchmark results
- `explain_ace_performance.py` - Generate explanations for ACE performance patterns

## Usage

```bash
# List available benchmarks
uv run python scripts/run_benchmark.py list

# Run ACE evaluation
uv run python scripts/run_benchmark.py simple_qa --limit 50

# Compare baseline vs ACE
uv run python scripts/run_benchmark.py simple_qa --limit 50 --compare
```

See [benchmarks/README.md](../benchmarks/README.md) for full documentation.
