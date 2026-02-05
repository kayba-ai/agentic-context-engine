"""Unit tests for TraceSandbox code execution."""

import platform
import unittest

import pytest

from ace.reflector.sandbox import TraceSandbox, ExecutionResult, ExecutionTimeoutError
from ace.reflector.trace_context import TraceContext, TraceStep


@pytest.mark.unit
class TestTraceSandbox(unittest.TestCase):
    """Test TraceSandbox code execution and security restrictions."""

    def setUp(self):
        """Set up test fixtures."""
        self.trace = TraceContext(
            steps=[
                TraceStep(
                    index=0,
                    action="search",
                    thought="Looking for data",
                    observation="Found 5 results",
                ),
                TraceStep(
                    index=1,
                    action="analyze",
                    thought="Processing results",
                    observation="Analysis complete",
                ),
            ],
            raw_reasoning="Step 1: Search\nStep 2: Analyze",
        )

    def test_basic_execution(self):
        """Test that basic code execution works."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute("x = 2 + 2\nprint(x)")

        self.assertIn("4", result.stdout)
        self.assertTrue(result.success)
        self.assertIsNone(result.exception)

    def test_trace_access(self):
        """Test that trace is accessible in the sandbox."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute("print(len(trace.steps))")

        self.assertIn("2", result.stdout)
        self.assertTrue(result.success)

    def test_trace_methods(self):
        """Test that trace methods work in sandbox."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute(
            """
step = trace.get_step(0)
print(f"Action: {step.action}")
"""
        )

        self.assertIn("Action: search", result.stdout)
        self.assertTrue(result.success)

    def test_final_captures_value(self):
        """Test that FINAL() captures the value correctly."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute('FINAL({"key": "value", "number": 42})')

        self.assertEqual(result.final_value, {"key": "value", "number": 42})
        self.assertTrue(sandbox.final_called)

    def test_final_stops_execution(self):
        """Test that FINAL() stops execution immediately."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute(
            """
FINAL("first value")
FINAL("second value")  # Should not be reached
"""
        )

        self.assertEqual(result.final_value, "first value")

    def test_blocked_open(self):
        """Test that open() is blocked."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute("open('test.txt')")

        self.assertIn("TypeError", result.stderr)
        self.assertFalse(result.success)

    def test_blocked_import(self):
        """Test that __import__ is blocked."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute("__import__('os')")

        self.assertIn("TypeError", result.stderr)
        self.assertFalse(result.success)

    def test_blocked_eval(self):
        """Test that eval is blocked."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute("eval('1 + 1')")

        self.assertIn("TypeError", result.stderr)
        self.assertFalse(result.success)

    def test_blocked_exec(self):
        """Test that exec is blocked (when called from inside sandbox code)."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute("exec('x = 1')")

        self.assertIn("TypeError", result.stderr)
        self.assertFalse(result.success)

    def test_safe_builtins_available(self):
        """Test that safe builtins are available."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute(
            """
# Test various safe builtins
nums = list(range(5))
print(f"len: {len(nums)}")
print(f"sum: {sum(nums)}")
print(f"max: {max(nums)}")
print(f"min: {min(nums)}")
print(f"sorted: {sorted([3, 1, 2])}")
print(f"str: {str(42)}")
print(f"int: {int('42')}")
print(f"bool: {bool(1)}")
print(f"isinstance: {isinstance(nums, list)}")
"""
        )

        self.assertIn("len: 5", result.stdout)
        self.assertIn("sum: 10", result.stdout)
        self.assertIn("max: 4", result.stdout)
        self.assertIn("min: 0", result.stdout)
        self.assertIn("sorted: [1, 2, 3]", result.stdout)
        self.assertIn("isinstance: True", result.stdout)
        self.assertTrue(result.success)

    def test_json_module_available(self):
        """Test that json module is available in sandbox."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute(
            """
import json  # This is not actually an import - json is pre-injected
data = json.dumps({"key": "value"})
print(data)
"""
        )

        # json is pre-injected, not imported, so this should fail
        # The import statement itself won't work, but json is in namespace
        sandbox2 = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result2 = sandbox2.execute(
            """
data = json.dumps({"key": "value"})
print(data)
"""
        )
        self.assertIn('{"key": "value"}', result2.stdout)

    def test_re_module_available(self):
        """Test that re module is available in sandbox."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute(
            """
matches = re.findall(r'Step \\d+', "Step 1, Step 2, Step 3")
print(matches)
"""
        )

        self.assertIn("Step 1", result.stdout)
        self.assertIn("Step 2", result.stdout)
        self.assertIn("Step 3", result.stdout)
        self.assertTrue(result.success)

    def test_collections_module_available(self):
        """Test that collections module is available in sandbox."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute(
            """
# Test Counter
counter = collections.Counter(['a', 'b', 'a', 'c', 'a'])
print(f"Counter: {counter.most_common(2)}")

# Test defaultdict
dd = collections.defaultdict(list)
dd['key'].append('value')
print(f"defaultdict: {dict(dd)}")

# Test deque
dq = collections.deque([1, 2, 3], maxlen=3)
dq.append(4)
print(f"deque: {list(dq)}")
"""
        )

        self.assertIn("Counter: [('a', 3)", result.stdout)
        self.assertIn("defaultdict: {'key': ['value']}", result.stdout)
        self.assertIn("deque: [2, 3, 4]", result.stdout)
        self.assertTrue(result.success)

    def test_datetime_module_available(self):
        """Test that datetime module is available in sandbox."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute(
            """
# Test datetime (direct construction - sandbox restricts time-based methods)
dt = datetime(2024, 1, 15, 10, 30, 0)
print(f"datetime value: {dt.year}-{dt.month}-{dt.day}")
print(f"datetime is datetime: {isinstance(dt, datetime)}")

# Test timedelta
delta = timedelta(days=1, hours=2)
print(f"timedelta: {delta.total_seconds()} seconds")

# Test date (direct construction)
d = date(2024, 1, 15)
print(f"date value: {d}")
print(f"date is date: {isinstance(d, date)}")

# Test time
t = time(10, 30, 0)
print(f"time is time: {isinstance(t, time)}")

# Test timezone
utc = timezone.utc
print(f"timezone: {utc}")
"""
        )

        self.assertIn("datetime is datetime: True", result.stdout)
        self.assertIn("datetime value: 2024-1-15", result.stdout)
        self.assertIn("timedelta:", result.stdout)
        self.assertIn("date is date: True", result.stdout)
        self.assertIn("time is time: True", result.stdout)
        self.assertIn("timezone:", result.stdout)
        self.assertTrue(result.success)

    def test_llm_query_function(self):
        """Test that llm_query function works when provided."""

        def mock_llm_query(prompt: str) -> str:
            return f"Response to: {prompt[:20]}..."

        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=mock_llm_query)
        result = sandbox.execute(
            """
response = llm_query("What is the meaning of life?")
print(response)
"""
        )

        self.assertIn("Response to: What is the meaning", result.stdout)
        self.assertTrue(result.success)

    def test_llm_query_disabled(self):
        """Test that llm_query returns stub when not provided."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute(
            """
response = llm_query("test prompt")
print(response)
"""
        )

        self.assertIn("llm_query disabled", result.stdout)
        self.assertTrue(result.success)

    def test_inject_variable(self):
        """Test that inject() adds variables to namespace."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        sandbox.inject("my_var", 42)
        sandbox.inject("my_list", [1, 2, 3])

        result = sandbox.execute(
            """
print(f"my_var: {my_var}")
print(f"my_list: {my_list}")
"""
        )

        self.assertIn("my_var: 42", result.stdout)
        self.assertIn("my_list: [1, 2, 3]", result.stdout)

    def test_exception_handling(self):
        """Test that exceptions are captured properly."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute("raise ValueError('test error')")

        self.assertIn("ValueError", result.stderr)
        self.assertIn("test error", result.stderr)
        self.assertFalse(result.success)
        self.assertIsInstance(result.exception, ValueError)

    def test_try_except_in_code(self):
        """Test that try/except works in sandbox code."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute(
            """
try:
    x = 1 / 0
except ZeroDivisionError:
    print("Caught division by zero")
"""
        )

        self.assertIn("Caught division by zero", result.stdout)
        self.assertTrue(result.success)

    def test_reset(self):
        """Test that reset() clears the final value."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)

        # First execution with FINAL
        sandbox.execute('FINAL("first")')
        self.assertEqual(sandbox.final_value, "first")
        self.assertTrue(sandbox.final_called)

        # Reset
        sandbox.reset()
        self.assertIsNone(sandbox.final_value)
        self.assertFalse(sandbox.final_called)

    def test_namespace_persistence(self):
        """Test that variables persist between executions."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)

        sandbox.execute("x = 10")
        result = sandbox.execute("print(x + 5)")

        self.assertIn("15", result.stdout)

    def test_no_trace(self):
        """Test that sandbox works without a trace."""
        sandbox = TraceSandbox(trace=None, llm_query_fn=None)
        result = sandbox.execute(
            """
if trace is None:
    print("No trace available")
else:
    print("Trace exists")
"""
        )

        self.assertIn("No trace available", result.stdout)
        self.assertTrue(result.success)


@pytest.mark.unit
class TestExecutionResult(unittest.TestCase):
    """Test ExecutionResult dataclass."""

    def test_success_no_error(self):
        """Test success property with no error."""
        result = ExecutionResult(stdout="output", stderr="", exception=None)
        self.assertTrue(result.success)

    def test_success_with_error_in_stderr(self):
        """Test success property with error in stderr."""
        result = ExecutionResult(stdout="", stderr="Error: something", exception=None)
        self.assertFalse(result.success)

    def test_success_with_exception(self):
        """Test success property with exception."""
        result = ExecutionResult(stdout="", stderr="", exception=ValueError("test"))
        self.assertFalse(result.success)


@pytest.mark.unit
class TestSandboxTimeout(unittest.TestCase):
    """Test sandbox timeout functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.trace = TraceContext(steps=[], raw_reasoning="test")

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="Timeout not supported on Windows"
    )
    def test_timeout_kills_infinite_loop(self):
        """Test that timeout kills an infinite loop."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)

        # Code with an infinite loop
        code = "while True: pass"
        result = sandbox.execute(code, timeout=1.0)

        self.assertFalse(result.success)
        self.assertIsInstance(result.exception, ExecutionTimeoutError)
        self.assertIn("ExecutionTimeoutError", result.stderr)
        self.assertIn("timeout", result.stderr.lower())

    @pytest.mark.skipif(
        platform.system() == "Windows", reason="Timeout not supported on Windows"
    )
    def test_timeout_kills_slow_computation(self):
        """Test that timeout kills slow computations."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)

        # Code that takes too long via CPU-bound work (import is blocked)
        code = """
x = 0
for i in range(10**10):  # Very long loop
    x += 1
"""
        result = sandbox.execute(code, timeout=1.0)

        self.assertFalse(result.success)
        self.assertIsInstance(result.exception, ExecutionTimeoutError)

    def test_fast_code_completes_within_timeout(self):
        """Test that fast code completes without timeout."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)

        code = "x = 1 + 1\nprint(x)"
        result = sandbox.execute(code, timeout=30.0)

        self.assertTrue(result.success)
        self.assertIn("2", result.stdout)


@pytest.mark.unit
class TestLLMQueryLimit(unittest.TestCase):
    """Test llm_query limit functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.trace = TraceContext(steps=[], raw_reasoning="test")
        self.call_count = 0

    def test_llm_query_limit_enforced(self):
        """Test that llm_query respects the call limit."""
        max_calls = 3

        def counting_llm_query(prompt: str) -> str:
            self.call_count += 1
            if self.call_count > max_calls:
                return f"(Max {max_calls} LLM calls exceeded - analyze with available data)"
            return f"Response {self.call_count}"

        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=counting_llm_query)

        # Call llm_query more times than the limit
        code = """
results = []
for i in range(5):
    results.append(llm_query(f"Query {i}"))
print(results)
"""
        result = sandbox.execute(code)

        self.assertTrue(result.success)
        # First 3 calls should succeed
        self.assertIn("Response 1", result.stdout)
        self.assertIn("Response 2", result.stdout)
        self.assertIn("Response 3", result.stdout)
        # Calls 4 and 5 should return the limit message
        self.assertIn("Max 3 LLM calls exceeded", result.stdout)


@pytest.mark.unit
class TestTraceContext(unittest.TestCase):
    """Test TraceContext utility methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.steps = [
            TraceStep(
                index=0,
                action="search",
                thought="Searching for data",
                observation="Found results",
            ),
            TraceStep(
                index=1,
                action="filter",
                thought="Filtering results",
                observation="Filtered to 10",
            ),
            TraceStep(
                index=2,
                action="analyze",
                thought="Analyzing data",
                observation="Error: failed to parse",
            ),
        ]
        self.trace = TraceContext(
            steps=self.steps, raw_reasoning="search -> filter -> analyze"
        )

    def test_len(self):
        """Test __len__ returns step count."""
        self.assertEqual(len(self.trace), 3)

    def test_iter(self):
        """Test iteration over steps."""
        actions = [step.action for step in self.trace]
        self.assertEqual(actions, ["search", "filter", "analyze"])

    def test_getitem(self):
        """Test indexing access."""
        self.assertEqual(self.trace[0].action, "search")
        self.assertEqual(self.trace[2].action, "analyze")

    def test_get_step_valid(self):
        """Test get_step with valid index."""
        step = self.trace.get_step(1)
        self.assertEqual(step.action, "filter")

    def test_get_step_invalid(self):
        """Test get_step with invalid index."""
        step = self.trace.get_step(100)
        self.assertIsNone(step)

    def test_find_steps(self):
        """Test find_steps with pattern."""
        results = self.trace.find_steps("search")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].action, "search")

    def test_find_steps_case_insensitive(self):
        """Test find_steps is case-insensitive by default."""
        results = self.trace.find_steps("SEARCH")
        self.assertEqual(len(results), 1)

    def test_find_steps_in_observation(self):
        """Test find_steps finds matches in observations."""
        results = self.trace.find_steps("filtered")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].action, "filter")

    def test_get_errors(self):
        """Test get_errors finds error steps."""
        errors = self.trace.get_errors()
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].action, "analyze")

    def test_get_actions(self):
        """Test get_actions filters by action type."""
        results = self.trace.get_actions("search")
        self.assertEqual(len(results), 1)

    def test_summary(self):
        """Test summary generation."""
        summary = self.trace.summary()
        self.assertIn("3 steps", summary)
        self.assertIn("search", summary)
        self.assertIn("analyze", summary)

    def test_search_raw(self):
        """Test search_raw returns step indices where pattern is found."""
        # Pattern "search" should match step 0 (action="search")
        indices = self.trace.search_raw(r"search")
        self.assertEqual(indices, [0])
        self.assertTrue(all(isinstance(i, int) for i in indices))

        # Pattern "filter" should match step 1
        indices = self.trace.search_raw(r"filter")
        self.assertEqual(indices, [1])

        # Pattern matching multiple steps
        indices = self.trace.search_raw(r"data")  # In thoughts of step 0, 2
        self.assertIn(0, indices)  # "Searching for data"
        self.assertIn(2, indices)  # "Analyzing data"

    def test_search_raw_text(self):
        """Test search_raw_text returns matched substrings from raw reasoning."""
        matches = self.trace.search_raw_text(r"\w+")
        self.assertIn("search", matches)

    def test_from_reasoning_string(self):
        """Test creating TraceContext from reasoning string."""
        reasoning = "1. First step\n2. Second step\n3. Third step"
        trace = TraceContext.from_reasoning_string(reasoning)

        self.assertEqual(len(trace), 3)


@pytest.mark.unit
class TestSubAgentLLM(unittest.TestCase):
    """Test SubAgentLLM and ask_llm function."""

    def setUp(self):
        """Set up test fixtures."""
        self.trace = TraceContext(
            steps=[
                TraceStep(
                    index=0,
                    action="search",
                    thought="Looking for data",
                    observation="Found 5 results",
                ),
            ],
            raw_reasoning="Step 1: Search",
        )

    def test_ask_llm_basic(self):
        """Test that ask_llm function works in sandbox."""
        from ace.reflector.subagent import create_ask_llm_function

        # Mock LLM that echoes back the question
        class MockLLM:
            def complete(self, prompt, **kwargs):
                class Response:
                    text = f"Analysis of: {prompt[:50]}..."

                return Response()

        ask_llm = create_ask_llm_function(MockLLM(), max_calls=5)
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        sandbox.inject("ask_llm", ask_llm)

        result = sandbox.execute(
            """
insight = ask_llm("What happened?", "Error: timeout after 30s")
print(f"Got insight: {insight[:30]}")
"""
        )

        self.assertIn("Got insight:", result.stdout)
        self.assertTrue(result.success)

    def test_ask_llm_with_trace_context(self):
        """Test ask_llm with extracted trace data."""
        from ace.reflector.subagent import create_ask_llm_function

        class MockLLM:
            def complete(self, prompt, **kwargs):
                class Response:
                    text = "The search found 5 results which is good."

                return Response()

        ask_llm = create_ask_llm_function(MockLLM(), max_calls=5)
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        sandbox.inject("ask_llm", ask_llm)

        result = sandbox.execute(
            """
# Extract trace data and ask about it
step = trace.get_step(0)
context = f"Action: {step.action}, Observation: {step.observation}"
insight = ask_llm("Is this result good?", context)
print(f"Insight: {insight}")
"""
        )

        self.assertIn("Insight:", result.stdout)
        self.assertIn("5 results", result.stdout)
        self.assertTrue(result.success)

    def test_ask_llm_call_limit(self):
        """Test that ask_llm respects call limit."""
        from ace.reflector.subagent import create_ask_llm_function

        class MockLLM:
            def complete(self, prompt, **kwargs):
                class Response:
                    text = "Response"

                return Response()

        ask_llm = create_ask_llm_function(MockLLM(), max_calls=2)
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        sandbox.inject("ask_llm", ask_llm)

        result = sandbox.execute(
            """
results = []
for i in range(5):
    results.append(ask_llm(f"Question {i}", "context"))
print(results)
"""
        )

        self.assertIn("Max 2 sub-agent calls exceeded", result.stdout)
        self.assertTrue(result.success)

    def test_subagent_llm_call_history(self):
        """Test that SubAgentLLM tracks call history."""
        from ace.reflector.subagent import SubAgentLLM, SubAgentConfig

        class MockLLM:
            def complete(self, prompt, **kwargs):
                class Response:
                    text = "Test response"

                return Response()

        subagent = SubAgentLLM(MockLLM(), config=SubAgentConfig(max_tokens=100))

        # Make some calls
        subagent.ask("Question 1", "Context 1")
        subagent.ask("Question 2", "Context 2")

        self.assertEqual(subagent.call_count, 2)
        self.assertEqual(len(subagent.call_history), 2)
        self.assertEqual(subagent.call_history[0]["question"], "Question 1")
        self.assertEqual(subagent.call_history[1]["question"], "Question 2")

    def test_subagent_reset(self):
        """Test that SubAgentLLM reset clears state."""
        from ace.reflector.subagent import SubAgentLLM

        class MockLLM:
            def complete(self, prompt, **kwargs):
                class Response:
                    text = "Response"

                return Response()

        subagent = SubAgentLLM(MockLLM())
        subagent.ask("Question", "Context")
        self.assertEqual(subagent.call_count, 1)

        subagent.reset()
        self.assertEqual(subagent.call_count, 0)
        self.assertEqual(len(subagent.call_history), 0)

    def test_subagent_separate_llm(self):
        """Test that SubAgentLLM can use a separate LLM."""
        from ace.reflector.subagent import SubAgentLLM

        class MainLLM:
            def complete(self, prompt, **kwargs):
                class Response:
                    text = "Main LLM response"

                return Response()

        class SubLLM:
            def complete(self, prompt, **kwargs):
                class Response:
                    text = "Sub LLM response"

                return Response()

        # With separate subagent_llm, it should use that
        subagent = SubAgentLLM(MainLLM(), subagent_llm=SubLLM())
        result = subagent.ask("Question", "Context")

        self.assertEqual(result, "Sub LLM response")


@pytest.mark.unit
class TestFinalVarFunction(unittest.TestCase):
    """Test FINAL_VAR convenience function."""

    def setUp(self):
        """Set up test fixtures."""
        self.trace = TraceContext(
            steps=[
                TraceStep(index=0, action="test", thought="Testing", observation="OK")
            ],
            raw_reasoning="Test step",
        )

    def test_final_var_basic(self):
        """Test FINAL_VAR with an existing variable."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        sandbox.execute(
            """
result = {"reasoning": "test", "key_insight": "works"}
FINAL_VAR("result")
"""
        )

        self.assertTrue(sandbox.final_called)
        self.assertEqual(sandbox.final_value["reasoning"], "test")
        self.assertEqual(sandbox.final_value["key_insight"], "works")

    def test_final_var_nonexistent(self):
        """Test FINAL_VAR with non-existent variable raises error."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute('FINAL_VAR("nonexistent")')

        self.assertIn("ValueError", result.stderr)
        self.assertIn("not found", result.stderr)
        self.assertFalse(sandbox.final_called)

    def test_final_var_equivalent_to_final(self):
        """Test FINAL_VAR produces same result as FINAL."""
        sandbox1 = TraceSandbox(trace=self.trace, llm_query_fn=None)
        sandbox2 = TraceSandbox(trace=self.trace, llm_query_fn=None)

        sandbox1.execute('FINAL({"value": 42})')
        sandbox2.execute(
            """
data = {"value": 42}
FINAL_VAR("data")
"""
        )

        self.assertEqual(sandbox1.final_value, sandbox2.final_value)


@pytest.mark.unit
class TestShowVarsFunction(unittest.TestCase):
    """Test SHOW_VARS debugging function."""

    def setUp(self):
        """Set up test fixtures."""
        self.trace = TraceContext(
            steps=[
                TraceStep(index=0, action="test", thought="Testing", observation="OK")
            ],
            raw_reasoning="Test step",
        )

    def test_show_vars_basic(self):
        """Test SHOW_VARS prints available variables."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute("SHOW_VARS()")

        self.assertIn("Available variables:", result.stdout)
        self.assertIn("trace", result.stdout)
        self.assertIn("FINAL", result.stdout)
        self.assertIn("FINAL_VAR", result.stdout)
        self.assertIn("SHOW_VARS", result.stdout)

    def test_show_vars_includes_injected(self):
        """Test SHOW_VARS shows injected variables."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        sandbox.inject("custom_var", "test")
        sandbox.inject("another_var", 123)
        result = sandbox.execute("SHOW_VARS()")

        self.assertIn("custom_var", result.stdout)
        self.assertIn("another_var", result.stdout)

    def test_show_vars_excludes_internals(self):
        """Test SHOW_VARS excludes internal variables."""
        sandbox = TraceSandbox(trace=self.trace, llm_query_fn=None)
        result = sandbox.execute("SHOW_VARS()")

        # Should not show builtins module or other internals
        self.assertNotIn("__builtins__", result.stdout)


if __name__ == "__main__":
    unittest.main()
