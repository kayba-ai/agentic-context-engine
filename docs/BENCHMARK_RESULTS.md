# ACE Benchmark Results

This document contains benchmark results comparing baseline LLM performance against ACE (Agentic Context Engineering) enhanced performance.

## Executive Summary

ACE demonstrates measurable improvements on math reasoning tasks while providing good generalization (no overfitting). The framework achieves this through adaptive skillbook learning that captures and applies problem-solving strategies.

| Benchmark | Baseline | ACE | Improvement | Overfitting Gap |
|-----------|----------|-----|-------------|-----------------|
| GSM8K (Math) | 96.00% | 100.00% | **+4.00%** | 0.00% |

## GSM8K Benchmark Results

### Configuration

| Parameter | Value |
|-----------|-------|
| Model | claude-3-haiku-20240307 |
| Total Samples | 100 |
| Train/Test Split | 70/30 (0.7 ratio) |
| Epochs | 1 |
| Async Learning | Enabled |
| Skill Deduplication | Enabled |
| Prompt Version | v1 |
| Temperature | 0.0 |

### Accuracy Results

| Metric | Baseline | ACE (Test) | ACE (Train) | Improvement |
|--------|----------|------------|-------------|-------------|
| Exact Match | 96.00% | 100.00% | 100.00% | +4.00% |
| Accuracy | 96.00% | 100.00% | 100.00% | +4.00% |
| Within 1% | 96.00% | 100.00% | 100.00% | +4.00% |
| Within 5% | 96.00% | 100.00% | 100.00% | +4.00% |

### Overfitting Analysis

| Metric | Gap (Train - Test) | Status |
|--------|-------------------|--------|
| Exact Match | 0.00% | Good generalization |
| Accuracy | 0.00% | Good generalization |

Zero overfitting gap indicates the strategies learned during training generalize well to unseen test samples.

### Performance Metrics

| Metric | Baseline | ACE | Ratio |
|--------|----------|-----|-------|
| Time | 422.1s | 1130.6s | 2.7x |
| Total Tokens | 462,084 | 3,301,370 | 7.1x |
| Prompt Tokens | 397,106 | 2,941,768 | 7.4x |
| Completion Tokens | 64,978 | 359,602 | 5.5x |
| API Calls | 142 | 468 | 3.3x |
| Cost | $0.14 | $0.80 | 5.6x |

### Cost-Benefit Analysis

- **Accuracy improvement**: +4.00% (96% -> 100%)
- **Additional cost**: $0.66 per 100 samples
- **Cost per percentage point improvement**: $0.165

For production use cases where accuracy is critical, the 5.6x cost increase may be justified by the 4% accuracy improvement and elimination of errors.

## Methodology

### Train/Test Split

The benchmark uses a 70/30 train/test split:
- **Training (70 samples)**: ACE learns problem-solving strategies
- **Testing (30 samples)**: Unseen samples evaluate true generalization

### ACE Learning Process

1. **Agent**: Generates answers using current skillbook
2. **Environment**: Evaluates correctness against ground truth
3. **Reflector**: Analyzes errors and identifies helpful strategies
4. **SkillManager**: Updates skillbook with new skills
5. **Deduplication**: Consolidates similar skills to reduce redundancy

### Metrics Explained

- **Exact Match**: Prediction exactly matches ground truth
- **Accuracy**: Binary correctness (same as exact match for math)
- **Within X%**: Numerical answer within X% of ground truth
- **Overfitting Gap**: Difference between train and test performance

## Reproduction

To reproduce these results:

```bash
# Clone the repository
git clone https://github.com/kayba-ai/agentic-context-engine
cd agentic-context-engine

# Install dependencies
uv sync

# Set API key
export ANTHROPIC_API_KEY="your-api-key"

# Run benchmark
uv run python benchmarks/run_benchmark.py gsm8k \
    --limit 100 \
    --compare \
    --save-detailed \
    --model claude-3-haiku-20240307 \
    --split-ratio 0.7 \
    --async-learning \
    --dedup
```

Results are saved to `benchmark_results/comparison_gsm8k_*.json`.

## Available Benchmarks

The benchmark framework supports multiple datasets:

| Benchmark | Type | Description |
|-----------|------|-------------|
| gsm8k | Math | Grade school math word problems |
| simple_qa | QA | Question answering |
| finer_ord | NER | Financial named entity recognition |
| mmlu | MC | Multi-subject multiple choice |
| hellaswag | MC | Common sense reasoning |
| arc_easy | MC | Science questions (easy) |
| arc_challenge | MC | Science questions (hard) |
| truthfulqa | MC | Truthfulness evaluation |
| simple_math | Math | Simple arithmetic |
| xbrl_math | Math | Financial calculations |

Run `uv run python benchmarks/run_benchmark.py list` to see all available benchmarks.

## Conclusion

ACE demonstrates effective self-improvement capability on math reasoning tasks:
- **4% accuracy improvement** on GSM8K
- **Zero overfitting** indicates learned strategies generalize
- **Trade-off**: 5.6x cost increase for perfect accuracy

The framework is particularly valuable for use cases where:
- Accuracy is more important than cost
- Learning from errors can improve future performance
- Task-specific strategies can be captured and reused
