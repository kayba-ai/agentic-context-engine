# RLM Integration for Recursive Reflector - Implementation Plan

## Executive Summary

**Recommendation: Use `alexzhang13/rlm` as optional dependency with careful adapter layer.**

After deep analysis of both codebases, the integration is feasible but requires addressing 5 key concerns:

1. **Async/Sync mismatch** - RLM is async, ACE uses ThreadPoolExecutor
2. **Structured output** - RLM returns strings, ACE expects Pydantic models
3. **Prompt adaptation** - Need RLM-specific reflection prompts
4. **Thread safety** - RLM maintains state, multiple threads cause race conditions
5. **LLM client compatibility** - Both use LiteLLM but differently

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           ACE Framework                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  User Code                                                              │
│  ─────────                                                              │
│  reflector = Reflector(llm, mode=ReflectorMode.AUTO)                   │
│  result = reflector.reflect(question=..., agent_output=..., ...)       │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Reflector.reflect() - Mode Routing (ace/roles.py)                │  │
│  │                                                                   │  │
│  │  if mode == SIMPLE:     → _reflect_impl()      [1 LLM call]      │  │
│  │  if mode == MULTI_PASS: → MultiPassReflector   [3 LLM calls]     │  │
│  │  if mode == RLM_REPL:   → RecursiveReflector   [3-15 LLM calls]  │  │
│  │  if mode == AUTO:       → estimate tokens, select mode           │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ RecursiveReflector (ace/reflector/recursive.py)                  │  │
│  │                                                                   │  │
│  │  1. Build TraceContext from agent_output                         │  │
│  │  2. Build REPL environment (trace, skillbook, helpers)           │  │
│  │  3. Create fresh RLM instance (thread-safe)                      │  │
│  │  4. Call rlm.completion() with reflection prompt                 │  │
│  │  5. Parse FINAL() JSON → ReflectorOutput                         │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ▼                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ RLM Library (alexzhang13/rlm)                                    │  │
│  │                                                                   │  │
│  │  REPL Loop:                                                      │  │
│  │    1. LLM generates Python code                                  │  │
│  │    2. Execute in sandboxed namespace                             │  │
│  │    3. Return output to LLM                                       │  │
│  │    4. Repeat until FINAL() or max_iterations                     │  │
│  │                                                                   │  │
│  │  Environment provides:                                           │  │
│  │    - trace.get_step(i), trace.find_steps("pattern")             │  │
│  │    - llm_query(prompt), llm_query_batched([...])                │  │
│  │    - FINAL(json_string), FINAL_VAR(variable)                    │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Integration Concerns & Solutions

### 1. Async/Sync Mismatch

**Problem:** ACE's async pipeline uses `ThreadPoolExecutor` for sync tasks. RLM uses `asyncio`.

**Location:** `ace/async_learning.py:417-464` (`_reflector_worker`)

**Solution:** RLM already provides sync wrappers internally. Use sync `completion()` not `acompletion()`:

```python
# RecursiveReflector - runs in ThreadPoolExecutor worker
def reflect(self, ...) -> ReflectorOutput:
    # Use sync method (RLM handles event loop internally)
    result = self.rlm.completion(prompt=prompt)  # NOT acompletion
    return self._parse_output(result.response)
```

### 2. Thread Safety

**Problem:** RLM maintains state (`_llm_calls`, `_iterations`, `_current_depth`). Multiple threads cause race conditions.

**Solution:** Create fresh RLM instance per reflection call:

```python
class RecursiveReflector:
    def __init__(self, model: str, config: RecursiveReflectorConfig):
        self.model = model
        self.config = config
        # DON'T create RLM here

    def reflect(self, ...) -> ReflectorOutput:
        # Create fresh instance per call (thread-safe)
        rlm = RLM(
            backend="litellm",
            backend_kwargs={"model_name": self.model},
            environment="local",
            max_iterations=self.config.max_iterations,
            max_depth=self.config.max_depth,
        )

        result = rlm.completion(prompt=self._build_prompt(...))
        return self._parse_output(result.response)
```

### 3. Structured Output (FINAL → ReflectorOutput)

**Problem:** RLM returns strings via `FINAL()`, ACE expects `ReflectorOutput` Pydantic model.

**Location:** `ace/roles.py:460-487`

**Solution:** Instruct RLM to output JSON, parse it:

```python
def _parse_output(self, response: str) -> ReflectorOutput:
    """Parse FINAL() JSON string into ReflectorOutput."""
    try:
        # RLM returns: FINAL('{"reasoning": "...", ...}')
        # The response IS the extracted final answer (just the inner string)
        data = json.loads(response)
        return ReflectorOutput.model_validate(data)
    except json.JSONDecodeError as e:
        # Fallback: retry or raise
        raise ValueError(f"RLM output is not valid JSON: {e}")
```

### 4. Prompt Adaptation

**Problem:** Current `REFLECTOR_V2_1_PROMPT` is designed for single-pass. Need RLM-specific version.

**Location:** `ace/prompts_v2_1.py:240-456`

**Solution:** Create `REFLECTOR_RLM_PROMPT` that preserves v2.1 diagnostic protocol but adds REPL instructions:

```python
REFLECTOR_RLM_PROMPT = '''
You are ACE Reflector v2.1 with recursive analysis capabilities.

## REPL Environment
You have access to a Python REPL with these variables and functions:

Variables:
- `question`: str - The original task/question
- `reasoning`: str - Agent's step-by-step reasoning
- `final_answer`: str - Agent's final answer
- `ground_truth`: str or None - Expected answer (if available)
- `feedback`: str - Execution feedback
- `skillbook`: str - Relevant strategies from skillbook

Functions:
- `llm_query(prompt)` → str - Spawn sub-analysis
- `FINAL(json_str)` - Output final analysis (MUST be valid JSON)

## Diagnostic Protocol (from v2.1)
Execute in priority order:
1. SUCCESS_CASE - Identify contributing strategies
2. CALCULATION_ERROR - Pinpoint exact error location
3. STRATEGY_MISAPPLICATION - Correct application method
4. WRONG_STRATEGY - Explain strategy-problem mismatch
5. MISSING_STRATEGY - Define capability gap

## Analysis Process
Write Python code to explore the reasoning trace:

```python
# 1. Check if answer is correct
is_correct = final_answer == ground_truth if ground_truth else None

# 2. Search for error patterns
import re
errors = re.findall(r'error|failed|exception', reasoning, re.I)

# 3. Analyze step by step
steps = reasoning.split('\\n')
for i, step in enumerate(steps):
    # Examine each reasoning step
    ...

# 4. Build final response
final_response = {
    "reasoning": "...",
    "error_identification": "...",
    "root_cause_analysis": "...",
    "correct_approach": "...",
    "key_insight": "...",
    "extracted_learnings": [...],
    "skill_tags": [...]
}

# 5. Output (MUST be valid JSON)
import json
FINAL(json.dumps(final_response))
```

## Output Schema (MUST match exactly)
{
  "reasoning": "<systematic analysis>",
  "error_identification": "<specific error or 'none'>",
  "root_cause_analysis": "<underlying cause>",
  "correct_approach": "<how to fix>",
  "key_insight": "<most valuable learning>",
  "extracted_learnings": [
    {"learning": "<atomic insight>", "atomicity_score": 0.95, "evidence": "<detail>"}
  ],
  "skill_tags": [
    {"id": "<skill-id>", "tag": "helpful|harmful|neutral", "justification": "<why>"}
  ]
}
'''
```

### 5. LLM Client Compatibility

**Problem:** ACE uses `LiteLLMClient`, RLM uses `litellm` directly with different config.

**Location:** `ace/llm_providers/litellm_client.py:75-205`

**Solution:** Pass model name and let RLM use its own litellm integration:

```python
class RecursiveReflector:
    def __init__(self, llm: LLMClient, config: RecursiveReflectorConfig):
        # Extract model name from ACE's client
        self.model = llm.model
        self.config = config

        # RLM will use litellm directly with same model
        # Environment variables (OPENAI_API_KEY, etc.) are shared
```

---

## File Structure

```
ace/
├── roles.py                         # MODIFY: Add ReflectorMode, routing
├── features.py                      # MODIFY: Add has_rlm()
│
├── reflector/                       # NEW MODULE
│   ├── __init__.py                 # Exports
│   ├── mode.py                     # ReflectorMode enum, config classes
│   ├── trace_context.py            # TraceStep, TraceContext
│   ├── trace_parser.py             # Parse browser-use, LangChain traces
│   │
│   ├── multi_pass.py               # MultiPassReflector (Phase 2)
│   │
│   └── rlm/                        # RLM integration (Phase 3)
│       ├── __init__.py
│       ├── recursive.py            # RecursiveReflector
│       ├── prompts.py              # REFLECTOR_RLM_PROMPT
│       └── environment.py          # Custom REPL env (optional)
│
└── llm_providers/
    └── litellm_client.py           # NO CHANGE (RLM uses litellm directly)
```

---

## Implementation Phases

### Phase 1: Foundation (2-3 days)

**Goal:** Create mode routing infrastructure

**Files:**
- `ace/reflector/__init__.py`
- `ace/reflector/mode.py`
- `ace/reflector/trace_context.py`

**Changes to `ace/roles.py`:**
```python
from enum import Enum

class ReflectorMode(Enum):
    SIMPLE = "simple"          # Current single-pass
    MULTI_PASS = "multi_pass"  # 3-pass pipeline (future)
    RLM_REPL = "rlm_repl"      # RLM recursive
    AUTO = "auto"              # Auto-select based on trace size

class Reflector:
    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = REFLECTOR_PROMPT,
        *,
        mode: ReflectorMode = ReflectorMode.SIMPLE,  # NEW
        recursive_config: Optional[RecursiveReflectorConfig] = None,  # NEW
        max_retries: int = 3,
    ):
        self.mode = mode
        self.recursive_config = recursive_config or RecursiveReflectorConfig()
        # ... rest unchanged

    def reflect(self, ...) -> ReflectorOutput:
        # Mode routing
        if self.mode == ReflectorMode.RLM_REPL:
            return self._recursive_reflect(...)
        elif self.mode == ReflectorMode.AUTO:
            mode = self._select_mode(agent_output)
            if mode == ReflectorMode.RLM_REPL:
                return self._recursive_reflect(...)
        # Default: existing single-pass
        return self._reflect_impl(...)
```

### Phase 2: Multi-Pass Reflector (3-4 days)

**Goal:** Intermediate mode for medium traces (no RLM dependency)

**Files:**
- `ace/reflector/multi_pass.py`
- `ace/reflector/prompts/` (overview, focused, synthesis)

**Pipeline:**
```
Pass 1: Overview → "Which steps need analysis?" → List[int]
Pass 2: Focused → "Analyze step N" → StepAnalysis[]
Pass 3: Synthesis → "Combine into ReflectorOutput" → ReflectorOutput
```

### Phase 3: RLM Integration (4-5 days)

**Goal:** Full recursive analysis for massive traces

**Files:**
- `ace/reflector/rlm/recursive.py`
- `ace/reflector/rlm/prompts.py`

**Key implementation:**
```python
# ace/reflector/rlm/recursive.py
from typing import Optional
import json

try:
    from rlm import RLM
    HAS_RLM = True
except ImportError:
    HAS_RLM = False

class RecursiveReflector:
    """RLM-powered reflector for massive traces."""

    def __init__(
        self,
        model: str,
        config: RecursiveReflectorConfig,
    ):
        if not HAS_RLM:
            raise ImportError(
                "RLM not installed. Install with: pip install ace-framework[rlm]"
            )
        self.model = model
        self.config = config

    def reflect(
        self,
        *,
        question: str,
        agent_output: AgentOutput,
        skillbook: Skillbook,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs,
    ) -> ReflectorOutput:
        # 1. Create fresh RLM instance (thread-safe)
        rlm = RLM(
            backend="litellm",
            backend_kwargs={"model_name": self.model},
            environment="local",
            max_iterations=self.config.max_iterations,
            max_depth=self.config.max_depth,
            verbose=self.config.verbose,
        )

        # 2. Build prompt with context
        prompt = self._build_prompt(
            question=question,
            reasoning=agent_output.reasoning,
            final_answer=agent_output.final_answer,
            ground_truth=ground_truth,
            feedback=feedback,
            skillbook=skillbook,
        )

        # 3. Run RLM (sync method for ThreadPoolExecutor compatibility)
        result = rlm.completion(prompt=prompt)

        # 4. Parse FINAL() output
        return self._parse_output(result.response)

    def _build_prompt(self, **kwargs) -> str:
        from .prompts import REFLECTOR_RLM_PROMPT
        return REFLECTOR_RLM_PROMPT.format(**kwargs)

    def _parse_output(self, response: str) -> ReflectorOutput:
        """Parse JSON from RLM's FINAL() output."""
        try:
            data = json.loads(response)
            return ReflectorOutput.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            # Build minimal output on parse failure
            return ReflectorOutput(
                reasoning=f"RLM analysis completed but output parsing failed: {e}",
                error_identification="output_parse_error",
                key_insight=response[:500],  # Preserve some output
                raw={"rlm_response": response, "parse_error": str(e)},
            )
```

### Phase 4: Testing & Benchmarks (2-3 days)

**Unit Tests:**
- `tests/test_reflector_mode.py` - Mode routing
- `tests/test_recursive_reflector.py` - RLM integration
- `tests/test_trace_context.py` - Trace parsing

**Integration Tests:**
- Compare single-pass vs RLM on identical traces
- Verify ReflectorOutput schema compliance
- Test async pipeline integration

**Benchmarks:**
- Real pilot traces (your massive dumps)
- Metrics: accuracy, latency, cost, token usage

---

## Dependencies

**pyproject.toml additions:**
```toml
[project.optional-dependencies]
rlm = [
    "rlm>=0.1.0",  # alexzhang13/rlm
]

# Combined install
all = [
    "ace-framework[observability,langchain,rlm]",
]
```

**Feature detection:**
```python
# ace/features.py
def has_rlm() -> bool:
    """Check if RLM is available."""
    return _check_import("rlm")
```

---

## Verification Plan

### Unit Tests

```python
# tests/test_recursive_reflector.py
class TestRecursiveReflector(unittest.TestCase):

    @unittest.skipUnless(has_rlm(), "RLM not installed")
    def test_basic_reflection(self):
        """Test RLM produces valid ReflectorOutput."""
        reflector = RecursiveReflector(model="gpt-4o-mini", config=...)
        output = reflector.reflect(
            question="What is 2+2?",
            agent_output=AgentOutput(reasoning="...", final_answer="4"),
            skillbook=Skillbook(),
            ground_truth="4",
            feedback="Correct!",
        )
        self.assertIsInstance(output, ReflectorOutput)
        self.assertTrue(output.reasoning)

    def test_thread_safety(self):
        """Test concurrent reflections don't interfere."""
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(reflector.reflect, ...)
                for _ in range(3)
            ]
            results = [f.result() for f in futures]
        # All should succeed independently
        self.assertEqual(len(results), 3)
```

### Integration Tests

```python
# tests/test_reflector_integration.py
def test_async_pipeline_with_rlm():
    """Test RLM works in async learning pipeline."""
    adapter = OfflineACE(
        skillbook=skillbook,
        agent=agent,
        reflector=Reflector(llm, mode=ReflectorMode.RLM_REPL),
        skill_manager=skill_manager,
        async_learning=True,  # Uses ThreadPoolExecutor
    )
    results = adapter.run(samples, environment, epochs=1)
    # Verify learning occurred
    assert len(skillbook.skills) > initial_count
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| RLM generates invalid JSON | Fallback ReflectorOutput with raw response |
| RLM timeout (>60s) | Configure `max_iterations=15` for faster completion |
| Thread race conditions | Fresh RLM instance per call |
| RLM not installed | Graceful fallback to SIMPLE mode |
| API errors in REPL | RLM has built-in retry logic |

---

## Summary

| Phase | Deliverable | Effort |
|-------|-------------|--------|
| 1 | Mode routing + TraceContext | 2-3 days |
| 2 | MultiPassReflector | 3-4 days |
| 3 | RecursiveReflector (RLM) | 4-5 days |
| 4 | Tests + Benchmarks | 2-3 days |
| **Total** | | **11-15 days** |

**Next Steps:**
1. Add `rlm` to pyproject.toml optional dependencies
2. Create `ace/reflector/` module structure
3. Implement Phase 1 (mode routing)
4. Test with simple cases before RLM integration
