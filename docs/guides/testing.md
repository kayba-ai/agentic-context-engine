# Testing

## Running Tests

=== "pytest (recommended)"

    ```bash
    uv run pytest                         # All tests
    uv run pytest -m unit                 # Unit tests only
    uv run pytest -m integration          # Integration tests only
    uv run pytest tests/test_skillbook.py # Specific file
    uv run pytest -v                      # Verbose output
    ```

=== "unittest"

    ```bash
    python -m unittest discover -s tests
    python -m unittest discover -s tests -v  # Verbose
    ```

## Testing Without API Calls

Use a mock LLM to test pipeline wiring without making real API calls. Any object with `complete()` and `complete_structured()` methods satisfies the `LLMClientLike` protocol:

```python
from unittest.mock import MagicMock
from ace_next import Agent, Reflector, SkillManager

mock_llm = MagicMock()
mock_llm.complete.return_value = '{"reasoning": "test", "final_answer": "4", "skill_ids": []}'

agent = Agent(mock_llm)
reflector = Reflector(mock_llm)
skill_manager = SkillManager(mock_llm)
```

## Unit Testing

### Testing the Skillbook

```python
from ace_next import Skillbook

def test_add_skill():
    skillbook = Skillbook()
    skill = skillbook.add_skill(
        section="Test",
        content="Test strategy",
        metadata={"helpful": 0, "harmful": 0, "neutral": 0},
    )
    assert len(skillbook.skills()) == 1
    assert skill.content == "Test strategy"

def test_save_load(tmp_path):
    skillbook = Skillbook()
    skillbook.add_skill(section="Test", content="Strategy")

    path = str(tmp_path / "test.json")
    skillbook.save_to_file(path)

    loaded = Skillbook.load_from_file(path)
    assert len(loaded.skills()) == 1
```

### Testing the Agent

```python
from unittest.mock import MagicMock
from ace_next import Agent, Skillbook

def test_agent_generate():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '{"reasoning": "2+2=4", "final_answer": "4", "skill_ids": []}'

    agent = Agent(mock_llm)
    output = agent.generate(
        question="What is 2+2?",
        context="",
        skillbook=Skillbook(),
    )
    assert output.final_answer is not None
    assert output.reasoning is not None
```

### Testing Reflector and SkillManager

```python
from unittest.mock import MagicMock
from ace_next import Agent, Reflector, SkillManager, Skillbook

def make_mock_llm():
    mock = MagicMock()
    mock.complete.return_value = '{"reasoning": "test", "final_answer": "4", "skill_ids": []}'
    return mock

def test_reflector():
    mock_llm = make_mock_llm()
    reflector = Reflector(mock_llm)
    agent = Agent(mock_llm)

    output = agent.generate(question="Test", context="", skillbook=Skillbook())
    reflection = reflector.reflect(
        question="Test",
        agent_output=output,
        skillbook=Skillbook(),
        ground_truth="expected",
        feedback="Correct",
    )
    assert reflection.key_insight is not None

def test_skill_manager():
    sm = SkillManager(make_mock_llm())
    # ... similar pattern with reflection input
```

## Integration Testing

### End-to-End Learning Cycle

```python
from unittest.mock import MagicMock
from ace_next import (
    ACE, Agent, Reflector, SkillManager,
    Sample, SimpleEnvironment,
)

def test_full_learning_cycle():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '{"reasoning": "test", "final_answer": "answer", "skill_ids": []}'

    runner = ACE.from_roles(
        agent=Agent(mock_llm),
        reflector=Reflector(mock_llm),
        skill_manager=SkillManager(mock_llm),
        environment=SimpleEnvironment(),
    )

    samples = [Sample(question="Test", context="", ground_truth="answer")]
    results = runner.run(samples, epochs=1)

    assert len(results) == 1
```

### Testing Checkpoints

```python
def test_checkpoints(tmp_path):
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '{"reasoning": "test", "final_answer": "A", "skill_ids": []}'

    runner = ACE.from_roles(
        agent=Agent(mock_llm),
        reflector=Reflector(mock_llm),
        skill_manager=SkillManager(mock_llm),
        environment=SimpleEnvironment(),
        checkpoint_dir=str(tmp_path),
        checkpoint_interval=1,
    )

    samples = [Sample(question="Q", context="", ground_truth="A")]
    runner.run(samples, epochs=1)

    # Check that checkpoint files were created
    checkpoints = list(tmp_path.glob("ace_*.json"))
    assert len(checkpoints) > 0
```

## Common Test Patterns

### Fixtures

```python
import pytest
from unittest.mock import MagicMock
from ace_next import Agent, Reflector, SkillManager, Skillbook

@pytest.fixture
def mock_llm():
    mock = MagicMock()
    mock.complete.return_value = '{"reasoning": "test", "final_answer": "4", "skill_ids": []}'
    return mock

@pytest.fixture
def skillbook():
    return Skillbook()

@pytest.fixture
def agent(mock_llm):
    return Agent(mock_llm)
```

### Mocking LLM Responses

```python
from unittest.mock import MagicMock

def test_with_mock():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = '{"reasoning": "...", "final_answer": "4", "skill_ids": []}'

    agent = Agent(mock_llm)
    # ...
```

## CI Configuration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - run: uv sync
      - run: uv run pytest -v
```

## Code Quality

```bash
uv run black ace/ tests/ examples/     # Format
uv run mypy ace/                       # Type check
uv run pre-commit run --all-files      # All hooks
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | Run `uv sync` to install all dependencies |
| API key errors in tests | Use `MagicMock` for unit tests (see above) |
| Flaky async tests | Increase timeout or use `wait_for_background()` |
| Coverage too low | `--cov-fail-under=25` is the threshold |

## What to Read Next

- [Full Pipeline Guide](full-pipeline.md) — what you're testing
- [Async Learning](async-learning.md) — testing async pipelines
