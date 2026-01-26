# Contributing to ACE Benchmarks

This guide explains how to add new benchmarks to the ACE evaluation framework.

## Quick Start

1. Create a YAML configuration in `tasks/`
2. Add any custom processing logic to `processors.py`
3. Add environment class to `environments.py` (if needed)
4. Run tests to verify

## Adding a New Benchmark

### Step 1: Create Configuration File

Create `tasks/your_benchmark.yaml`:

```yaml
task: your_benchmark
version: "1.0"

data:
  source: huggingface
  dataset_path: "organization/dataset-name"
  split: test
  streaming: true

preprocessing:
  question_template: "{question}"
  ground_truth_field: answer

metrics:
  - name: accuracy
    weight: 1.0
  - name: f1
    weight: 0.5

metadata:
  description: "Brief description of the benchmark"
  task_type: "multiple_choice"  # or: numerical_reasoning, qa, ner
  paper_url: "https://arxiv.org/abs/..."
```

### Step 2: Add Processor (if needed)

If your dataset requires custom preprocessing, add a processor to `processors.py`:

```python
class YourBenchmarkProcessor:
    """Processor for YourBenchmark dataset."""

    def process_samples(
        self, sample_stream: Iterator[Dict[str, Any]]
    ) -> Iterator[Sample]:
        """Process raw samples into ACE format."""
        for data in sample_stream:
            # Validate required fields
            _validate_required(data, ["question", "answer"], "YourBenchmark sample")

            yield Sample(
                question=self._format_question(data),
                ground_truth=data["answer"],
                context=data.get("context", ""),
                metadata={"source_id": data.get("id", "")},
            )

    def _format_question(self, data: Dict[str, Any]) -> str:
        """Format question for presentation."""
        return f"Question: {data['question']}\n\nAnswer:"
```

Register it in `get_processor()`:

```python
def get_processor(benchmark_name: str):
    processors = {
        # ... existing processors ...
        "your_benchmark": YourBenchmarkProcessor(),
    }
    return processors.get(benchmark_name)
```

### Step 3: Add Environment (if needed)

For non-standard evaluation logic, add an environment to `environments.py`:

```python
class YourBenchmarkEnvironment(BenchmarkEnvironment):
    """Environment for YourBenchmark evaluation."""

    def evaluate(self, sample: Sample, agent_output) -> EnvironmentResult:
        prediction = agent_output.final_answer or ""
        ground_truth = sample.ground_truth or ""

        # Custom evaluation logic
        metrics = self._compute_metrics(prediction, ground_truth)

        # Generate feedback
        feedback = self._generate_feedback(metrics)

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=ground_truth,
            metrics=metrics,
        )
```

Register in `manager.py`:

```python
def _get_environment_class(self, config: BenchmarkConfig) -> Type[BenchmarkEnvironment]:
    task_name = config.task.lower()

    if "your_benchmark" in task_name:
        return YourBenchmarkEnvironment
    # ... existing checks ...
```

## Benchmark Types

### Multiple Choice
Use `MultipleChoiceEnvironment` for A/B/C/D style questions.

Set in YAML:
```yaml
metadata:
  task_type: multiple_choice
```

### Numerical Reasoning
Use `MathEnvironment` for numerical answers.

Set in YAML:
```yaml
metadata:
  task_type: numerical_reasoning
```

### Named Entity Recognition
Use `FiNEREnvironment` or create a custom NER environment.

### Code Generation
Use `SWEBenchEnvironment` or create a custom code evaluation environment.

## Testing Your Benchmark

### Unit Tests

Add tests to `tests/test_processors.py`:

```python
class TestYourBenchmarkProcessor:
    def test_process_valid_sample(self):
        processor = YourBenchmarkProcessor()
        data = {"question": "Test?", "answer": "Yes"}
        result = list(processor.process_samples([data]))[0]

        assert result.ground_truth == "Yes"
        assert "Test?" in result.question
```

### Integration Test

Run the benchmark with a small sample:

```bash
uv run python benchmarks/run_benchmark.py your_benchmark --limit 10
```

### Comparison Test

Compare baseline vs ACE performance:

```bash
uv run python benchmarks/run_benchmark.py your_benchmark --limit 50 --compare
```

## Code Style

- Use type hints for all function signatures
- Add docstrings to classes and public methods
- Use the validation helpers `_validate_required()` and `_validate_type()`
- Use constants from `constants.py` instead of magic numbers
- Add `__all__` exports for new public classes/functions

## Validation Checklist

Before submitting:

- [ ] YAML config loads without errors
- [ ] Processor handles edge cases (missing fields, empty values)
- [ ] Environment returns correct metrics
- [ ] Unit tests pass
- [ ] Integration test with `--limit 10` works
- [ ] Code follows project style (run `uv run black benchmarks/`)

## Common Patterns

### Handling Dataset Variations

```python
def process_samples(self, sample_stream):
    for data in sample_stream:
        # Handle different field names
        question = data.get("question") or data.get("prompt") or data.get("input")
        answer = data.get("answer") or data.get("label") or data.get("output")

        if not question:
            continue  # Skip malformed samples

        yield Sample(question=question, ground_truth=str(answer))
```

### Custom Metrics

```python
def _compute_metrics(self, prediction: str, ground_truth: str) -> Dict[str, float]:
    base_metrics = super()._compute_metrics(prediction, ground_truth)

    # Add custom metric
    base_metrics["custom_score"] = self._calculate_custom_score(prediction, ground_truth)

    return base_metrics
```

### Feedback Generation

```python
def _generate_feedback(self, metrics: Dict[str, float]) -> str:
    score = metrics.get("accuracy", 0.0)

    if score >= PerformanceThreshold.EXCELLENT:
        return "Excellent performance!"
    elif score >= PerformanceThreshold.MODERATE:
        return f"Good performance ({score:.0%}). Room for improvement."
    else:
        return f"Needs improvement ({score:.0%}). Review approach."
```

## Getting Help

- Check existing benchmarks in `tasks/` for examples
- Review `environments.py` for evaluation patterns
- Run tests with `uv run pytest benchmarks/tests/ -v`
