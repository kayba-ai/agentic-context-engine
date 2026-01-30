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
print(f"type: {type(nums).__name__}")
"""
        )

        self.assertIn("len: 5", result.stdout)
        self.assertIn("sum: 10", result.stdout)
        self.assertIn("max: 4", result.stdout)
        self.assertIn("min: 0", result.stdout)
        self.assertIn("sorted: [1, 2, 3]", result.stdout)
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
        """Test search_raw on raw reasoning."""
        matches = self.trace.search_raw(r"\w+")
        self.assertIn("search", matches)

    def test_from_reasoning_string(self):
        """Test creating TraceContext from reasoning string."""
        reasoning = "1. First step\n2. Second step\n3. Third step"
        trace = TraceContext.from_reasoning_string(reasoning)

        self.assertEqual(len(trace), 3)


if __name__ == "__main__":
    unittest.main()
