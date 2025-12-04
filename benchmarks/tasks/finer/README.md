# FiNER Benchmark

## Overview

FiNER (Financial Named Entity Recognition) is a benchmark for evaluating named entity recognition capabilities on financial text. This benchmark uses the `gtfintechlab/finer-ord` dataset from HuggingFace Hub, which contains 201 manually annotated financial news articles.

## Dataset Details

- **Source**: [gtfintechlab/finer-ord](https://huggingface.co/datasets/gtfintechlab/finer-ord)
- **Task Type**: Named Entity Recognition
- **Domain**: Finance
- **Language**: English
- **License**: CC BY-NC 4.0

### Dataset Statistics
- **Total samples**: 201 articles
- **Train split**: 160 articles
- **Validation split**: 20 articles
- **Test split**: 21 articles

### Entity Types
The dataset covers standard entity types commonly found in financial text:
- PERSON: Individual names
- ORGANIZATION: Company names, institutions
- LOCATION: Geographic locations
- And other entity types as found in the original annotations

## Evaluation Metrics

This benchmark uses the following evaluation metrics:

1. **F1 Score** (macro-averaged): Primary metric for entity recognition performance
2. **Precision** (macro-averaged): Measures accuracy of identified entities
3. **Recall** (macro-averaged): Measures coverage of actual entities
4. **Exact Match**: Measures perfect entity set matching

## Usage

### Using the Benchmark Manager

```python
from benchmarks import BenchmarkTaskManager

# Initialize manager
manager = BenchmarkTaskManager()

# Get FiNER benchmark
finer_env = manager.get_benchmark("finer_ord")

# Load data
data_loader = manager.get_data_loader("huggingface")
samples = list(data_loader.load(
    dataset_path="gtfintechlab/finer-ord",
    split="test",
    streaming=True
))

# Evaluate with ACE
from ace import Agent
agent = Agent(your_llm_client)

for sample_data in samples:
    # Convert to BenchmarkSample
    sample = BenchmarkSample(
        question=f"Identify entities in: {sample_data['gold_token']}",
        ground_truth=str(sample_data['gold_label']),
        metadata=sample_data
    )

    # Generate response
    output = agent.generate(
        question=sample.question,
        context="",
        skillbook=skillbook
    )

    # Evaluate
    result = finer_env.evaluate(sample, output)
    print(f"F1: {result.metrics['f1']:.2%}")
```

### Expected Input Format

The model receives financial text and should identify named entities. The input prompt includes:
- The financial text to analyze
- Instructions to identify and classify entities
- Request for structured output

### Expected Output Format

Models should return entities in a structured format, such as:

```json
[
  {"text": "Goldman Sachs", "label": "ORGANIZATION"},
  {"text": "New York", "label": "LOCATION"},
  {"text": "John Smith", "label": "PERSON"}
]
```

Alternative formats are supported, including free text with clear entity mentions.

## Data Access

The benchmark uses HuggingFace's streaming capabilities for efficient data access:

- **No local storage required**: Data streams directly from HuggingFace Hub
- **Automatic caching**: Downloaded data cached for subsequent runs
- **Configurable cache location**: Set `BENCHMARK_CACHE_DIR` environment variable

## Performance Baseline

Typical performance ranges for financial NER:
- **Excellent**: F1 > 80%
- **Good**: F1 60-80%
- **Needs Improvement**: F1 < 60%

Financial text presents unique challenges due to domain-specific terminology and entity types.