# ACE Benchmarks

Evaluate ACE performance with scientific rigor using our comprehensive benchmark suite.

This evaluation framework tests Agentic Context Engineering (ACE) across multiple datasets with automatic metrics, train/test splits, and overfitting analysis to ensure honest performance measurements.

## Quick Start

```bash
# List available benchmarks
uv run python benchmarks/run_benchmark.py list

# Run ACE evaluation with train/test split (default)
uv run python benchmarks/run_benchmark.py finer_ord --limit 100

# Run baseline only (no ACE learning)
uv run python benchmarks/run_benchmark.py simple_qa --limit 50 --skip-adaptation

# Compare baseline vs ACE side-by-side
uv run python benchmarks/run_benchmark.py hellaswag --limit 50 --compare
```

## Available Benchmarks

| Benchmark | Description | Domain | Default Limit |
|-----------|-------------|---------|---------------|
| **finer_ord** | Financial Named Entity Recognition | Finance | 100 |
| **simple_qa** | Question Answering (SQuAD) | General | 200 |
| **gsm8k** | Math Word Problems (Grade School Math 8K) | Mathematics | 100 |
| **mmlu** | Massive Multitask Language Understanding | General Knowledge | 500 |
| **hellaswag** | Commonsense Reasoning | Common Sense | 200 |
| **arc_easy** | AI2 Reasoning Challenge (Easy) | Reasoning | 200 |
| **arc_challenge** | AI2 Reasoning Challenge (Hard) | Reasoning | 200 |
| **truthfulqa** | TruthfulQA - Factual Accuracy | Factual | 200 |
| **winogrande** | Winogrande - Commonsense Reasoning | Common Sense | 200 |
| **xbrl_math** | XBRL Financial Math Problems | Finance | 100 |
| **appworld** | AppWorld Task Completion | Agentic | 100 |
| **letta_bench** | Letta Agent Benchmark | Agentic | 100 |
| **swe_bench** | Software Engineering Benchmark | Code | 100 |
| **simple_math** | Basic Math Problems | Mathematics | 100 |

## Command Options

```bash
uv run python benchmarks/run_benchmark.py <benchmark> [options]
```

**Key Options:**
- `--limit` - Override sample limit (always overrides config)
- `--model` - Model name (default: gpt-4o-mini)
- `--skip-adaptation` - Skip ACE learning (faster baseline)
- `--compare` - Run both baseline and ACE, then compare results
- `--epochs` - ACE adaptation epochs (default: 1)
- `--split-ratio` - Train/test split ratio (default: 0.8)
- `--online-mode` - Use continuous learning instead of offline
- `--prompt-version` - Use v1 or v2 prompts (default: v1)
- `--save-detailed` - Save per-sample results
- `--quiet` - Suppress progress output

## Examples

```bash
# Quick test with 10 samples
uv run python benchmarks/run_benchmark.py finer_ord --limit 10 --quiet

# Compare baseline vs ACE
uv run python benchmarks/run_benchmark.py simple_qa --limit 50 --compare

# Full ACE evaluation with v2 prompts
uv run python benchmarks/run_benchmark.py simple_qa --epochs 3 --prompt-version v2 --save-detailed

# Online learning mode
uv run python benchmarks/run_benchmark.py hellaswag --limit 100 --online-mode

# Custom train/test split (90/10)
uv run python benchmarks/run_benchmark.py mmlu --limit 100 --split-ratio 0.9

# Test all benchmarks quickly (baseline only)
for benchmark in finer_ord simple_qa hellaswag arc_easy; do
  uv run python benchmarks/run_benchmark.py $benchmark --limit 5 --skip-adaptation --quiet
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

## Evaluation Modes

The benchmark script supports three evaluation modes:

1. **ACE Mode (default)**: Train/test split with learning
   ```bash
   uv run python benchmarks/run_benchmark.py simple_qa --limit 100
   ```

2. **Baseline Mode**: No learning, direct evaluation
   ```bash
   uv run python benchmarks/run_benchmark.py simple_qa --limit 100 --skip-adaptation
   ```

3. **Comparison Mode**: Runs both baseline and ACE, shows improvement
   ```bash
   uv run python benchmarks/run_benchmark.py simple_qa --limit 100 --compare
   ```

## Key Features

- **Overfitting Prevention**: Automatic 80/20 train/test splits ensure true generalization metrics
- **Scientific Rigor**: Comprehensive evaluation modes with honest performance analysis
- **Multiple Domains**: Finance, general knowledge, reasoning, math, and common sense benchmarks
- **Flexible Configuration**: Customizable limits, models, and evaluation parameters
- **Performance Tracking**: Detailed results with per-sample analysis options

## AppWorld Direct Testing

For testing ACE with AppWorld directly (bypassing HAL harness):

```bash
# Run with defaults (2 tasks)
uv run python test_appworld_ace.py

# Run with options
uv run python test_appworld_ace.py --model gpt-4o-mini --limit 5
uv run python test_appworld_ace.py --limit 10 --save-skillbook appworld_skills.json
uv run python test_appworld_ace.py --quiet  # Suppress progress output
```

**CLI Options:**
- `--model MODEL` - LLM model (default: gpt-4o-mini)
- `--limit N` - Number of tasks to run (default: 2)
- `--max-interactions N` - Max steps per task (default: 5)
- `--save-skillbook PATH` - Save learned skillbook to file
- `--quiet` - Suppress progress output

**Prerequisites:**
1. AppWorld server running:
   ```bash
   docker run -d --name appworld-env -p 8000:8000 \
     -v ~/.appworld/data:/appworld/data -w /appworld \
     -e SERVER_TYPE=environment ghcr.io/stonybrooknlp/appworld
   ```
2. AppWorld data downloaded:
   ```bash
   pip install appworld && appworld download data
   ```

## Notes

- **Default 80/20 train/test split** prevents overfitting and shows true generalization
- The `--limit` parameter always overrides config file limits
- ACE adaptation improves performance through iterative learning
- Use `--compare` to see baseline vs ACE improvement side-by-side
- Overfitting warnings help identify when ACE memorizes vs generalizes
- Opik tracing warnings ("Failed to log adaptation metrics") are harmless