"""Stress tests for RR components.

Tests sandbox behavior and the PydanticAI-based RRStep entry points.
"""

import asyncio
import copy
import json
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import UsageLimits

from ace.implementations.rr.config import RecursiveConfig
from ace.core.sandbox import TraceSandbox, create_readonly_sandbox

from ace.core.context import ACEStepContext, SkillbookView
from ace.core.outputs import AgentOutput, ReflectorOutput
from ace.core.skillbook import Skillbook
from ace.steps.rr_step import RRConfig, RRStep
from ace.implementations.rr.tools import RRDeps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(
    question: str = "q",
    answer: str = "4",
    reasoning: str = "r",
    ground_truth: str | None = None,
    feedback: str | None = None,
) -> ACEStepContext:
    """Build an ACEStepContext suitable for RRStep.__call__."""
    trace: dict = {
        "question": question,
        "steps": [
            {"role": "agent", "reasoning": reasoning, "answer": answer, "skill_ids": []}
        ],
    }
    if ground_truth is not None:
        trace["ground_truth"] = ground_truth
    if feedback is not None:
        trace["feedback"] = feedback
    return ACEStepContext(trace=trace, skillbook=SkillbookView(Skillbook()))


_RUN_SYNC = "ace.core.recursive_agent.run_agent_sync"


def _mock_compaction_result(
    *,
    reasoning: str = "done",
    key_insight: str = "insight",
    correct_approach: str = "approach",
    timed_out: bool = False,
) -> tuple[ReflectorOutput, dict]:
    """Create a mock return value for run_agent_sync."""
    output = ReflectorOutput(
        reasoning=reasoning,
        error_identification="none",
        root_cause_analysis="mock root cause",
        correct_approach=correct_approach,
        key_insight=key_insight,
        raw={},
    )
    metadata = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "requests": 3,
        },
        "compactions": 0,
        "depth": 0,
        "iterations": 2,
        "timed_out": timed_out,
    }

    return output, metadata


# =========================================================================
# 1. RRStep lifecycle (PydanticAI-based)
# =========================================================================


@pytest.mark.unit
class TestLoopLifecycle:
    def test_successful_reflection(self):
        """Happy path: PydanticAI agent produces valid ReflectorOutput."""
        rr = RRStep("test-model", config=RRConfig())
        reflection, metadata = _mock_compaction_result(key_insight="insight")

        with patch(_RUN_SYNC, return_value=(reflection, metadata)):
            result_ctx = rr(
                _make_ctx(
                    question="What is 2+2?",
                    ground_truth="4",
                    feedback="Correct!",
                )
            )

        result = result_ctx.reflections[0]
        assert isinstance(result, ReflectorOutput)
        assert result.key_insight == "insight"

    def test_max_requests_timeout(self):
        """Budget exhaustion produces timeout output."""
        rr = RRStep(
            "test-model",
            config=RRConfig(max_requests=3),
        )

        reflection, metadata = _mock_compaction_result(
            reasoning="Analysis reached budget limit.",
            timed_out=True,
        )

        with patch(_RUN_SYNC, return_value=(reflection, metadata)):
            result_ctx = rr(_make_ctx())

        assert len(result_ctx.reflections) == 1
        assert isinstance(result_ctx.reflections[0], ReflectorOutput)
        assert "budget limit" in result_ctx.reflections[0].reasoning.lower()

    def test_budget_field_in_config(self):
        """max_tokens and max_requests config fields exist."""
        rr = RRStep(
            "test-model",
            config=RRConfig(max_tokens=100_000, max_requests=42),
        )
        assert rr.config.max_tokens == 100_000
        assert rr.config.max_requests == 42

    def test_config_build_usage_limits(self):
        """build_usage_limits() produces correct UsageLimits."""
        cfg = RecursiveConfig(
            max_tokens=500_000, max_requests=50, context_window=128_000
        )
        limits = cfg.build_usage_limits()
        assert limits.total_tokens_limit == 500_000
        assert limits.request_limit == 50

    def test_config_build_usage_limits_with_remaining(self):
        """build_usage_limits() uses remaining_tokens when provided."""
        cfg = RecursiveConfig(max_tokens=500_000, max_requests=50)
        limits = cfg.build_usage_limits(remaining_tokens=100_000)
        assert limits.total_tokens_limit == 100_000
        assert limits.request_limit == 50

    def test_rr_trace_metadata_on_success(self):
        """Successful reflection populates rr_trace metadata."""
        rr = RRStep("test-model", config=RRConfig())
        reflection, metadata = _mock_compaction_result()

        with patch(_RUN_SYNC, return_value=(reflection, metadata)):
            result_ctx = rr(_make_ctx())

        result = result_ctx.reflections[0]
        assert "rr_trace" in result.raw
        assert result.raw["rr_trace"]["timed_out"] is False
        assert isinstance(result.raw["rr_trace"]["subagent_calls"], list)

    def test_rr_trace_metadata_on_timeout(self):
        """Budget exhaustion produces rr_trace with timed_out=True."""
        from ace.core.recursive_agent import BudgetExhausted

        rr = RRStep("test-model", config=RRConfig())

        with patch(_RUN_SYNC, side_effect=BudgetExhausted(compaction_count=1)):
            result_ctx = rr(_make_ctx())

        result = result_ctx.reflections[0]
        assert "rr_trace" in result.raw
        assert result.raw["rr_trace"]["timed_out"] is True


# =========================================================================
# 2. Sandbox behavior
# =========================================================================


@pytest.mark.unit
class TestSandboxBehavior:
    def test_sandbox_variables_persist_across_iterations(self):
        """Variables set in one execution persist for the next."""
        sandbox = TraceSandbox(trace=None)
        sandbox.execute("x = 42", timeout=5.0)
        result = sandbox.execute("print(x + 1)", timeout=5.0)
        assert "43" in result.stdout

    def test_sandbox_code_modifies_injected_traces(self):
        """Mutation of injected dict is visible in later executions."""
        sandbox = TraceSandbox(trace=None)
        traces = {"question": "q", "items": [1, 2, 3]}
        sandbox.inject("traces", traces)
        sandbox.execute("traces['items'].append(4)", timeout=5.0)
        result = sandbox.execute("print(len(traces['items']))", timeout=5.0)
        assert "4" in result.stdout

    def test_sandbox_exception_produces_stderr(self):
        """Code that raises captures error in stderr."""
        sandbox = TraceSandbox(trace=None)
        result = sandbox.execute("raise RuntimeError('boom')", timeout=5.0)
        assert not result.success
        assert "RuntimeError" in result.stderr
        assert "boom" in result.stderr

    def test_registered_helpers_persist_and_run(self):
        """Registered helpers should persist across execute_code calls."""
        sandbox = TraceSandbox(trace=None)
        sandbox.inject("traces", {"values": [1, 2, 3]})
        result = sandbox.execute(
            """
register_helper(
    "sum_values",
    "def sum_values():\\n    return sum(traces['values'])\\n",
    "Return the sum of traces['values']",
)
print(run_helper("sum_values"))
            """.strip(),
            timeout=5.0,
        )

        assert result.success
        assert "6" in result.stdout
        assert sandbox.namespace["list_helpers"]()[0]["name"] == "sum_values"

    def test_registered_helpers_are_rehydrated_in_snapshots(self):
        """Sub-agent snapshots should recreate helpers against the child namespace."""
        parent = TraceSandbox(trace=None)
        parent.inject("traces", {"values": [1, 2, 3]})
        parent.execute(
            """
register_helper(
    "sum_values",
    "def sum_values():\\n    return sum(traces['values'])\\n",
    "Return the sum of traces['values']",
)
            """.strip(),
            timeout=5.0,
        )

        child = create_readonly_sandbox(parent)
        child.namespace["traces"]["values"].append(4)
        result = child.execute('print(run_helper("sum_values"))', timeout=5.0)

        assert result.success
        assert "10" in result.stdout
        assert parent.namespace["traces"]["values"] == [1, 2, 3]

    def test_batch_accessors_handle_nested_task_payloads(self):
        """Batch helpers should expose a stable view over nested task items."""
        sandbox = TraceSandbox(trace=None)
        batch_items = [
            {
                "task_id": "task_0",
                "question": "Where is my order?",
                "feedback": "Task FAILED (reward=0.0)",
                "trace": {
                    "messages": [
                        {"role": "assistant", "content": "Hi!"},
                        {"role": "user", "content": "Where is my order?"},
                    ]
                },
            }
        ]
        sandbox.inject("batch_items", batch_items)
        sandbox.inject("item_ids", ["task_0"])

        result = sandbox.execute(
            """
item = get_batch_item(0)
print(get_item_id(0))
print(get_item_question(item))
print(get_item_feedback(item))
messages = get_item_messages(item)
print(len(messages))
print(get_message_text(messages[1]))
print(preview_item(0)["payload_type"])
            """.strip(),
            timeout=5.0,
        )

        assert result.success
        lines = result.stdout.strip().splitlines()
        assert lines[0] == "task_0"
        assert lines[1] == "Where is my order?"
        assert lines[2] == "Task FAILED (reward=0.0)"
        assert lines[3] == "2"
        assert lines[4] == "Where is my order?"
        assert lines[5] == "dict"


# =========================================================================
# 3. Entry points (PydanticAI-based)
# =========================================================================


@pytest.mark.unit
class TestEntryPoints:
    def test_call_produces_reflection(self):
        """__call__() produces a ReflectorOutput on the context."""
        rr = RRStep("test-model", config=RRConfig())
        reflection, metadata = _mock_compaction_result(key_insight="insight")

        traces = {
            "question": "q",
            "steps": [
                {"role": "agent", "reasoning": "r", "answer": "4", "skill_ids": []}
            ],
        }
        ctx = ACEStepContext(trace=traces, skillbook=SkillbookView(Skillbook()))

        with patch(_RUN_SYNC, return_value=(reflection, metadata)):
            result_ctx = rr(ctx)

        assert isinstance(result_ctx.reflections[0], ReflectorOutput)
        assert result_ctx.reflections[0].key_insight == "insight"

    def test_reflect_method_works(self):
        """reflect() works as ReflectorLike entry point."""
        rr = RRStep("test-model", config=RRConfig())
        reflection, metadata = _mock_compaction_result(key_insight="reflected")

        with patch(_RUN_SYNC, return_value=(reflection, metadata)):
            result = rr.reflect(
                question="What is 2+2?",
                agent_output=AgentOutput(reasoning="r", final_answer="4"),
                ground_truth="4",
            )

        assert isinstance(result, ReflectorOutput)
        assert result.key_insight == "reflected"


# =========================================================================
# 4. New architecture tests
# =========================================================================


@pytest.mark.unit
class TestRecurseToolRegistration:
    def test_recurse_tool_registered_at_non_leaf_depth(self):
        """Agent at depth 0 with max_depth=2 should have recurse tool."""
        from ace.core.recursive_agent import AgenticConfig, RecursiveAgent

        ra = RecursiveAgent(
            "test-model",
            output_type=ReflectorOutput,
            system_prompt="test",
            config=AgenticConfig(max_depth=2),
        )
        agent = ra._create_agent(depth=0)
        tool_names = list(agent._function_toolset.tools.keys())
        assert "recurse" in tool_names

    def test_recurse_tool_not_registered_at_max_depth(self):
        """Agent at max_depth should NOT have recurse tool."""
        from ace.core.recursive_agent import AgenticConfig, RecursiveAgent

        ra = RecursiveAgent(
            "test-model",
            output_type=ReflectorOutput,
            system_prompt="test",
            config=AgenticConfig(max_depth=2),
        )
        agent = ra._create_agent(depth=2)
        tool_names = list(agent._function_toolset.tools.keys())
        assert "recurse" not in tool_names

    def test_recurse_tool_not_registered_at_depth_zero_max_zero(self):
        """Agent at depth=0, max_depth=0 should NOT have recurse tool."""
        from ace.core.recursive_agent import AgenticConfig, RecursiveAgent

        ra = RecursiveAgent(
            "test-model",
            output_type=ReflectorOutput,
            system_prompt="test",
            config=AgenticConfig(max_depth=0),
        )
        agent = ra._create_agent(depth=0)
        tool_names = list(agent._function_toolset.tools.keys())
        assert "recurse" not in tool_names


@pytest.mark.unit
class TestRecurseStateIsolation:
    def test_failed_child_does_not_leak_mutations(self):
        from ace.core.recursive_agent import AgenticConfig, register_recurse
        from ace.core.skillbook import UpdateOperation
        from ace.implementations.sm_tools import SMDeps

        class _ToolCapture:
            def __init__(self):
                self.tools = {}

            def tool(self, fn=None, **_):
                if fn is None:

                    def decorator(func):
                        self.tools[func.__name__] = func
                        return func

                    return decorator

                self.tools[fn.__name__] = fn
                return fn

        capture = _ToolCapture()
        register_recurse(capture)

        skillbook = Skillbook()
        parent = SMDeps(
            config=AgenticConfig(),
            sandbox=TraceSandbox(trace=None),
            skillbook=skillbook,
        )

        async def failing_child(*, deps, prompt, depth):
            skill = deps.skillbook.add_skill(
                section="strategy",
                issue="should-not-leak",
                keywords=["child"],
                insight="mutation from failed child",
            )

            deps.operations.append(
                UpdateOperation(
                    type="ADD",
                    section=skill.section,
                    issue=skill.issue,
                    keywords=skill.keywords,
                    insight=skill.insight,
                    skill_id=skill.id,
                )
            )

            raise RuntimeError("simulated child failure")

        parent.run_session_fn = failing_child

        ctx = MagicMock()
        ctx.deps = parent

        result = asyncio.run(
            capture.tools["recurse"](
                ctx,
                "run failing child",
            )
        )

        assert "child session failed" in result
        assert not any(skill.issue == "should-not-leak" for skill in skillbook.skills())
        assert parent.operations == []

    def test_parallel_children_have_isolated_state(self):
        from ace.core.recursive_agent import AgenticConfig, register_recurse
        from ace.core.skillbook import UpdateOperation
        from ace.implementations.sm_tools import SMDeps

        class _ToolCapture:
            def __init__(self):
                self.tools = {}

            def tool(self, fn=None, **_):
                if fn is None:

                    def decorator(func):
                        self.tools[func.__name__] = func
                        return func

                    return decorator

                self.tools[fn.__name__] = fn
                return fn

        capture = _ToolCapture()
        register_recurse(capture)

        parent = SMDeps(
            config=AgenticConfig(),
            sandbox=TraceSandbox(trace=None),
            skillbook=Skillbook(),
        )

        children = {}

        async def run_child(*, deps, prompt, depth):
            children[prompt] = deps

            skill = deps.skillbook.add_skill(
                section="strategy",
                issue=prompt,
                keywords=["child"],
                insight=f"mutation from {prompt}",
            )

            deps.operations.append(
                UpdateOperation(
                    type="ADD",
                    section=skill.section,
                    issue=skill.issue,
                    keywords=skill.keywords,
                    insight=skill.insight,
                    skill_id=skill.id,
                )
            )

            await asyncio.sleep(0)
            return prompt, None

        parent.run_session_fn = run_child

        ctx = MagicMock()
        ctx.deps = parent

        async def run_children():
            return await asyncio.gather(
                capture.tools["recurse"](ctx, "child-a"),
                capture.tools["recurse"](ctx, "child-b"),
            )

        asyncio.run(run_children())

        child_a = children["child-a"]
        child_b = children["child-b"]

        assert child_a.skillbook is not parent.skillbook
        assert child_b.skillbook is not parent.skillbook
        assert child_a.skillbook is not child_b.skillbook

        assert child_a.operations is not parent.operations
        assert child_b.operations is not parent.operations
        assert child_a.operations is not child_b.operations

        parent_issues = {skill.issue for skill in parent.skillbook.skills()}

        assert parent_issues == {"child-a", "child-b"}
        assert len(parent.operations) == 2

    def test_commit_child_remaps_followup_update(self):
        from ace.core.recursive_agent import AgenticConfig
        from ace.core.skillbook import UpdateOperation
        from ace.implementations.sm_tools import SMDeps

        parent = SMDeps(
            config=AgenticConfig(),
            skillbook=Skillbook(),
        )
        child = parent.for_child(sandbox=TraceSandbox(trace=None))

        skill = child.skillbook.add_skill(
            section="strategy",
            issue="before",
            keywords=["child"],
            insight="before insight",
        )
        child.operations.append(
            UpdateOperation(
                type="ADD",
                section=skill.section,
                issue=skill.issue,
                keywords=skill.keywords,
                insight=skill.insight,
                skill_id=skill.id,
            )
        )

        child.skillbook.update_skill(
            skill.id,
            issue="after",
            keywords=["child"],
            insight="after insight",
        )
        child.operations.append(
            UpdateOperation(
                type="UPDATE",
                section=skill.section,
                issue="after",
                keywords=["child"],
                insight="after insight",
                skill_id=skill.id,
            )
        )

        parent.commit_child(child)

        committed = parent.skillbook.skills()

        assert len(committed) == 1
        assert committed[0].issue == "after"
        assert committed[0].insight == "after insight"

        assert len(parent.operations) == 2
        assert parent.operations[0].skill_id == committed[0].id
        assert parent.operations[1].skill_id == committed[0].id

    def test_commit_child_failure_leaves_parent_unchanged(self):
        from ace.core.recursive_agent import AgenticConfig
        from ace.core.skillbook import UpdateOperation
        from ace.implementations.sm_tools import SMDeps

        parent = SMDeps(
            config=AgenticConfig(),
            skillbook=Skillbook(),
        )
        child = parent.for_child(sandbox=TraceSandbox(trace=None))

        child.operations.extend(
            [
                UpdateOperation(
                    type="ADD",
                    section="context",
                    issue="temporary",
                    keywords=["child"],
                    insight="valid insight",
                    skill_id="context-00001",
                ),
                UpdateOperation(
                    type="UPDATE",
                    section="context",
                    issue="invalid update",
                    keywords=["child"],
                    insight="",
                    skill_id="context-00001",
                ),
            ]
        )

        with pytest.raises(ValueError):
            parent.commit_child(child)

        assert parent.skillbook.skills() == []
        assert parent.operations == []

    def test_for_child_preserves_computed_embeddings(self):
        from ace.core.recursive_agent import AgenticConfig
        from ace.implementations.sm_tools import SMDeps

        parent = SMDeps(config=AgenticConfig(), skillbook=Skillbook())
        skill = parent.skillbook.add_skill(
            section="context", issue="issue", keywords=["k"], insight="insight"
        )
        skill.embedding = [0.1, 0.2, 0.3]

        child = parent.for_child(sandbox=TraceSandbox(trace=None))

        child_skill = child.skillbook.get_skill(skill.id)
        assert child_skill is not None
        assert child_skill.embedding == [0.1, 0.2, 0.3]

        child_skill.embedding[0] = 999.0
        assert parent.skillbook.get_skill(skill.id).embedding == [0.1, 0.2, 0.3]

    def test_commit_child_remaps_temp_id_when_it_diverges_from_committed_id(self):
        from ace.core.recursive_agent import AgenticConfig
        from ace.core.skillbook import UpdateOperation
        from ace.implementations.sm_tools import SMDeps

        parent = SMDeps(config=AgenticConfig(), skillbook=Skillbook())

        child_a = parent.for_child(sandbox=TraceSandbox(trace=None))
        skill_a = child_a.skillbook.add_skill(
            section="context", issue="first", keywords=["c"], insight="first insight"
        )
        child_a.operations.append(
            UpdateOperation(
                type="ADD",
                section=skill_a.section,
                issue=skill_a.issue,
                keywords=skill_a.keywords,
                insight=skill_a.insight,
                skill_id=skill_a.id,
            )
        )

        child_b = parent.for_child(sandbox=TraceSandbox(trace=None))
        skill_b = child_b.skillbook.add_skill(
            section="context", issue="second", keywords=["c"], insight="second insight"
        )
        child_b.operations.append(
            UpdateOperation(
                type="ADD",
                section=skill_b.section,
                issue=skill_b.issue,
                keywords=skill_b.keywords,
                insight=skill_b.insight,
                skill_id=skill_b.id,
            )
        )

        assert skill_a.id == skill_b.id

        id_map_a = parent.commit_child(child_a)
        id_map_b = parent.commit_child(child_b)

        committed_id_a = id_map_a[skill_a.id]
        committed_id_b = id_map_b[skill_b.id]

        assert committed_id_b != skill_b.id
        assert committed_id_a != committed_id_b

        committed_b_skill = parent.skillbook.get_skill(committed_id_b)
        assert committed_b_skill is not None
        assert committed_b_skill.issue == "second"

    def test_recurse_rewrites_committed_id_in_returned_text(self):
        from ace.core.recursive_agent import AgenticConfig, register_recurse
        from ace.core.skillbook import UpdateOperation
        from ace.implementations.sm_tools import SMDeps

        class _ToolCapture:
            def __init__(self):
                self.tools = {}

            def tool(self, fn=None, **_):
                if fn is None:

                    def decorator(func):
                        self.tools[func.__name__] = func
                        return func

                    return decorator

                self.tools[fn.__name__] = fn
                return fn

        capture = _ToolCapture()
        register_recurse(capture)

        parent = SMDeps(
            config=AgenticConfig(),
            sandbox=TraceSandbox(trace=None),
            skillbook=Skillbook(),
        )

        async def run_child(*, deps, prompt, depth):
            parent.skillbook.add_skill(
                section="context",
                issue="sibling",
                keywords=["s"],
                insight="sibling insight",
            )

            skill = deps.skillbook.add_skill(
                section="context",
                issue="child issue",
                keywords=["c"],
                insight="child insight",
            )
            deps.operations.append(
                UpdateOperation(
                    type="ADD",
                    section=skill.section,
                    issue=skill.issue,
                    keywords=skill.keywords,
                    insight=skill.insight,
                    skill_id=skill.id,
                )
            )
            return f"Added skill {skill.id} for the sub-task.", deps

        parent.run_session_fn = run_child

        ctx = MagicMock()
        ctx.deps = parent

        result = asyncio.run(capture.tools["recurse"](ctx, "investigate"))

        new_skill = next(
            s for s in parent.skillbook.skills() if s.issue == "child issue"
        )

        assert new_skill.id in result
        assert new_skill.id != "context-00001"
        assert "context-00001" not in result

    def test_recurse_id_remap_does_not_cascade(self):
        from ace.core.recursive_agent import AgenticConfig, register_recurse
        from ace.implementations.sm_tools import SMDeps

        class _ToolCapture:
            def __init__(self):
                self.tools = {}

            def tool(self, fn=None, **_):
                if fn is None:

                    def decorator(func):
                        self.tools[func.__name__] = func
                        return func

                    return decorator

                self.tools[fn.__name__] = fn
                return fn

        capture = _ToolCapture()
        register_recurse(capture)

        parent = SMDeps(
            config=AgenticConfig(),
            sandbox=TraceSandbox(trace=None),
            skillbook=Skillbook(),
        )

        async def run_child(*, deps, prompt, depth):
            return "Added skill context-00001 for the sub-task.", deps

        parent.run_session_fn = run_child
        parent.commit_child = MagicMock(
            return_value={
                "context-00001": "context-00002",
                "context-00002": "context-00003",
            }
        )

        ctx = MagicMock()
        ctx.deps = parent

        result = asyncio.run(capture.tools["recurse"](ctx, "investigate"))

        assert "context-00002" in result
        assert "context-00003" not in result

    def test_concurrent_commit_child_serializes_id_assignment(self):
        from ace.core.recursive_agent import AgenticConfig
        from ace.core.skillbook import UpdateOperation
        from ace.implementations.sm_tools import SMDeps

        parent = SMDeps(config=AgenticConfig(), skillbook=Skillbook())
        n_children = 16
        errors: list[Exception] = []
        results: dict[int, str] = {}
        results_lock = threading.Lock()

        def run_one(i: int) -> None:
            try:
                child = parent.for_child(sandbox=TraceSandbox(trace=None))
                skill = child.skillbook.add_skill(
                    section="context",
                    issue=f"issue-{i}",
                    keywords=["c"],
                    insight=f"insight-{i}",
                )
                child.operations.append(
                    UpdateOperation(
                        type="ADD",
                        section=skill.section,
                        issue=skill.issue,
                        keywords=skill.keywords,
                        insight=skill.insight,
                        skill_id=skill.id,
                    )
                )
                id_map = parent.commit_child(child)
                with results_lock:
                    results[i] = id_map[skill.id]
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=run_one, args=(i,)) for i in range(n_children)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Concurrent commit errors: {errors}"

        committed_ids = list(results.values())
        assert len(committed_ids) == n_children
        assert len(set(committed_ids)) == n_children

        assert len(parent.skillbook.skills()) == n_children
        for i, committed_id in results.items():
            skill = parent.skillbook.get_skill(committed_id)
            assert skill is not None
            assert skill.issue == f"issue-{i}"


@pytest.mark.unit
class TestMicrocompaction:
    def test_microcompact_clears_old_tool_results(self):
        """_microcompact should clear old tool results, keeping recent ones."""
        rr = RRStep("test-model", config=RRConfig())

        # Build a message list with 5 tool results
        messages = []
        for i in range(5):
            messages.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name="execute_code",
                            content=f"result {i}",
                            tool_call_id=f"call_{i}",
                        ),
                    ]
                )
            )

        from ace.core.recursive_agent import microcompact

        compacted = microcompact(messages, keep_recent=2, tool_names=("execute_code",))

        # Should NOT be the same object (changes were made)
        assert compacted is not messages

        # First 3 should be cleared, last 2 kept
        for i in range(3):
            assert "[cleared" in compacted[i].parts[0].content
        assert compacted[3].parts[0].content == "result 3"
        assert compacted[4].parts[0].content == "result 4"

    def test_microcompact_returns_same_when_nothing_to_clear(self):
        """_microcompact returns identity when keep_recent >= total tool results."""
        rr = RRStep("test-model", config=RRConfig())

        messages = [
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="execute_code",
                        content="result 0",
                        tool_call_id="call_0",
                    ),
                ]
            ),
        ]

        from ace.core.recursive_agent import microcompact

        result = microcompact(messages, keep_recent=3, tool_names=("execute_code",))
        assert result is messages  # identity = no change

    def test_microcompact_ignores_non_tool_messages(self):
        """_microcompact should not touch model response messages."""
        rr = RRStep("test-model", config=RRConfig())

        messages = [
            ModelResponse(parts=[TextPart(content="thinking...")]),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="execute_code",
                        content="old result",
                        tool_call_id="call_0",
                    ),
                ]
            ),
            ModelResponse(parts=[TextPart(content="more thinking...")]),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="execute_code",
                        content="new result",
                        tool_call_id="call_1",
                    ),
                ]
            ),
        ]

        from ace.core.recursive_agent import microcompact

        compacted = microcompact(messages, keep_recent=1, tool_names=("execute_code",))
        assert compacted is not messages
        # First tool result cleared, second kept
        assert "[cleared" in compacted[1].parts[0].content
        assert compacted[3].parts[0].content == "new result"
        # Model responses untouched
        assert compacted[0].parts[0].content == "thinking..."
        assert compacted[2].parts[0].content == "more thinking..."


@pytest.mark.unit
class TestBudgetExhausted:
    def test_is_budget_exhausted_tokens(self):
        """is_budget_exhausted detects total token limit."""
        from ace.core.recursive_agent import is_budget_exhausted

        limits = UsageLimits(total_tokens_limit=1000, request_limit=50)
        usage = MagicMock()
        usage.total_tokens = 1000
        usage.requests = 5
        assert is_budget_exhausted(limits, usage) is True

    def test_is_budget_exhausted_requests(self):
        """is_budget_exhausted detects request limit."""
        from ace.core.recursive_agent import is_budget_exhausted

        limits = UsageLimits(total_tokens_limit=100_000, request_limit=10)
        usage = MagicMock()
        usage.total_tokens = 500
        usage.requests = 10
        assert is_budget_exhausted(limits, usage) is True

    def test_is_budget_not_exhausted(self):
        """is_budget_exhausted returns False when under budget."""
        from ace.core.recursive_agent import is_budget_exhausted

        limits = UsageLimits(total_tokens_limit=100_000, request_limit=50)
        usage = MagicMock()
        usage.total_tokens = 500
        usage.requests = 5
        assert is_budget_exhausted(limits, usage) is False
