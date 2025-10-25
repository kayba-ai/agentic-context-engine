# ACE Benchmarks

Evaluation framework for Agentic Context Engineering (ACE) with multiple datasets and automatic metrics.

## Quick Start

```bash
# List available benchmarks
uv run python scripts/run_benchmark.py list

# Run a benchmark with default settings
uv run python scripts/run_benchmark.py finer_ord

# Run with custom sample limit (overrides config)
uv run python scripts/run_benchmark.py simple_qa --limit 50

# Skip ACE adaptation (direct evaluation only)
uv run python scripts/run_benchmark.py hellaswag --skip-adaptation --limit 20
```

## Available Benchmarks

| Benchmark | Description | Domain | Default Limit |
|-----------|-------------|---------|---------------|
| **finer_ord** | Financial Named Entity Recognition | Finance | 100 |
| **simple_qa** | Question Answering (SQuAD) | General | 200 |
| **simple_math** | Math Word Problems (GSM8K) | Mathematics | 100 |
| **mmlu** | Massive Multitask Language Understanding | General Knowledge | 500 |
| **hellaswag** | Commonsense Reasoning | Common Sense | 200 |
| **arc_easy** | AI2 Reasoning Challenge (Easy) | Reasoning | 200 |
| **arc_challenge** | AI2 Reasoning Challenge (Hard) | Reasoning | 200 |

## Command Options

```bash
uv run python scripts/run_benchmark.py <benchmark> [options]
```

**Key Options:**
- `--limit` - Override sample limit (always overrides config)
- `--model` - Model name (default: gpt-4o-mini)
- `--skip-adaptation` - Skip ACE learning (faster baseline)
- `--epochs` - ACE adaptation epochs (default: 1)
- `--save-detailed` - Save per-sample results
- `--quiet` - Suppress progress output

## Examples

```bash
# Quick test with 10 samples
uv run python scripts/run_benchmark.py finer_ord --limit 10 --quiet

# Full ACE evaluation
uv run python scripts/run_benchmark.py simple_qa --epochs 3 --save-detailed

# Test all benchmarks quickly
for benchmark in finer_ord simple_qa hellaswag arc_easy; do
  uv run python scripts/run_benchmark.py $benchmark --limit 5 --skip-adaptation --quiet
done
```

## Output

Results saved to `benchmark_results/` with format:
- **Summary**: `{benchmark}_{model}_{timestamp}_summary.json`
- **Detailed**: `{benchmark}_{model}_{timestamp}_detailed.json` (if `--save-detailed`)

## Adding Custom Benchmarks

Create `benchmarks/tasks/my_benchmark.yaml`:

```yaml
task: my_benchmark
version: "1.0"

data:
  source: huggingface
  dataset_path: my/dataset
  split: test
  limit: 100

metrics:
  - name: exact_match
    weight: 1.0

metadata:
  description: "My custom benchmark"
  domain: "my_domain"
```

## Notes

- The `--limit` parameter always overrides config file limits
- ACE adaptation improves performance through iterative learning
- Use `--skip-adaptation` for faster baseline evaluation
- Opik tracing warnings ("Failed to log adaptation metrics") are harmless