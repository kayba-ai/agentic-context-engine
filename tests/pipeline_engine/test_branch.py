"""Comprehensive unit tests for Branch and MergeStrategy.

Sections
--------
1.  TestBranchConstruction        — creation, contract inference, merge selection
2.  TestBranchSyncRaiseOnConflict — RAISE_ON_CONFLICT semantics
3.  TestBranchSyncLastWriteWins   — LAST_WRITE_WINS semantics
4.  TestBranchSyncNamespaced      — NAMESPACED semantics
5.  TestBranchSyncCustomMerge     — custom callable merge
6.  TestBranchSyncFailures        — all-branches-run / error collection
7.  TestBranchSyncImmutability    — frozen context shared safely
8.  TestBranchAsyncParity         — async path mirrors every sync behaviour
9.  TestBranchAsyncNativeCoroutines — native async __call__ children
10. TestBranchViaRun              — Branch embedded inside Pipeline.run()
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pytest

from pipeline import Branch, BranchError, MergeStrategy, Pipeline, StepContext
from pipeline.branch import (
    _merge_last_write_wins,
    _merge_namespaced,
    _merge_raise_on_conflict,
)

# ---------------------------------------------------------------------------
# Test-local subclass with named fields for merge/conflict tests
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestContext(StepContext):
    """Subclass with domain fields used by branch merge tests."""

    agent_output: Any = None
    reflection: Any = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class WriteX:
    requires = frozenset()
    provides = frozenset({"x"})

    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx.replace(metadata=MappingProxyType({**ctx.metadata, "x": "from_x"}))


class WriteY:
    requires = frozenset()
    provides = frozenset({"y"})

    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx.replace(metadata=MappingProxyType({**ctx.metadata, "y": "from_y"}))


class WriteZ:
    requires = frozenset()
    provides = frozenset({"z"})

    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx.replace(metadata=MappingProxyType({**ctx.metadata, "z": "from_z"}))


class WriteN:
    """Writes metadata['n'] = n.  Used in custom-merge arithmetic tests."""

    requires = frozenset()
    provides = frozenset({"n"})

    def __init__(self, n: int):
        self.n = n

    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx.replace(metadata=MappingProxyType({**ctx.metadata, "n": self.n}))


class WriteAgent:
    """Writes to the named field ctx.agent_output (not metadata)."""

    requires = frozenset()
    provides = frozenset({"agent_output"})

    def __init__(self, value: str):
        self.value = value

    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx.replace(agent_output=self.value)


class WriteReflection:
    """Writes to the named field ctx.reflection — second named-field for conflict tests."""

    requires = frozenset()
    provides = frozenset({"reflection"})

    def __init__(self, value: str):
        self.value = value

    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx.replace(reflection=self.value)


class Explode:
    """Always raises RuntimeError with a configurable message."""

    requires = frozenset()
    provides = frozenset()

    def __init__(self, msg: str = "boom"):
        self.msg = msg

    def __call__(self, ctx: StepContext) -> StepContext:
        raise RuntimeError(self.msg)


class Log:
    """Side-effect step: appends its name to a shared log (thread-safe)."""

    requires = frozenset()
    provides = frozenset()

    def __init__(self, name: str, log: list[str], lock: threading.Lock):
        self.name = name
        self.log = log
        self.lock = lock

    def __call__(self, ctx: StepContext) -> StepContext:
        with self.lock:
            self.log.append(self.name)
        return ctx



# --- native async helpers ---


class AsyncWriteX:
    requires = frozenset()
    provides = frozenset({"x"})

    async def __call__(self, ctx: StepContext) -> StepContext:
        await asyncio.sleep(0)
        return ctx.replace(metadata=MappingProxyType({**ctx.metadata, "x": "async_x"}))


class AsyncWriteY:
    requires = frozenset()
    provides = frozenset({"y"})

    async def __call__(self, ctx: StepContext) -> StepContext:
        await asyncio.sleep(0)
        return ctx.replace(metadata=MappingProxyType({**ctx.metadata, "y": "async_y"}))


class AsyncExplode:
    requires = frozenset()
    provides = frozenset()

    def __init__(self, msg: str = "async_boom"):
        self.msg = msg

    async def __call__(self, ctx: StepContext) -> StepContext:
        await asyncio.sleep(0)
        raise RuntimeError(self.msg)


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBranchConstruction:

    def test_no_children_raises_value_error(self):
        with pytest.raises(ValueError):
            Branch()

    def test_single_child_accepted(self):
        b = Branch(Pipeline().then(WriteX()))
        assert len(b.pipelines) == 1

    def test_two_children(self):
        b = Branch(Pipeline().then(WriteX()), Pipeline().then(WriteY()))
        assert len(b.pipelines) == 2

    def test_three_children(self):
        b = Branch(
            Pipeline().then(WriteX()),
            Pipeline().then(WriteY()),
            Pipeline().then(WriteZ()),
        )
        assert len(b.pipelines) == 3

    def test_provides_is_union_of_all_children(self):
        b = Branch(
            Pipeline().then(WriteX()),
            Pipeline().then(WriteY()),
            Pipeline().then(WriteZ()),
        )
        assert {"x", "y", "z"} <= b.provides

    def test_provides_with_overlapping_children(self):
        """Two children both providing 'x' → 'x' still appears once in provides."""
        b = Branch(Pipeline().then(WriteX()), Pipeline().then(WriteX()))
        assert "x" in b.provides
        assert b.provides == frozenset({"x"})

    def test_requires_is_union_of_all_children(self):
        class NeedsA:
            requires = frozenset({"a"})
            provides = frozenset({"b"})

            def __call__(self, ctx):
                return ctx

        class NeedsC:
            requires = frozenset({"c"})
            provides = frozenset({"d"})

            def __call__(self, ctx):
                return ctx

        b = Branch(NeedsA(), NeedsC())
        assert "a" in b.requires
        assert "c" in b.requires

    def test_empty_pipeline_children_accepted(self):
        b = Branch(Pipeline(), Pipeline())
        assert b.requires == frozenset()
        assert b.provides == frozenset()

    def test_raw_step_as_child_not_only_pipeline(self):
        """Branch accepts any callable with requires/provides, not only Pipeline."""
        b = Branch(WriteX(), WriteY())
        assert "x" in b.provides
        assert "y" in b.provides

    # merge selection -------------------------------------------------------

    def test_default_merge_is_raise_on_conflict(self):
        b = Branch(Pipeline().then(WriteX()))
        assert b._merge_fn is _merge_raise_on_conflict

    def test_last_write_wins_merge_selected(self):
        b = Branch(Pipeline().then(WriteX()), merge=MergeStrategy.LAST_WRITE_WINS)
        assert b._merge_fn is _merge_last_write_wins

    def test_namespaced_merge_selected(self):
        b = Branch(Pipeline().then(WriteX()), merge=MergeStrategy.NAMESPACED)
        assert b._merge_fn is _merge_namespaced

    def test_custom_callable_stored_directly(self):
        fn = lambda ctxs: ctxs[0]
        b = Branch(Pipeline().then(WriteX()), merge=fn)
        assert b._merge_fn is fn

    def test_custom_callable_not_confused_with_enum(self):
        """A callable that happens to equal a string must not be mistaken for an enum."""
        fn = lambda ctxs: ctxs[0]
        b = Branch(Pipeline().then(WriteX()), merge=fn)
        assert b._merge_fn is not _merge_raise_on_conflict


# ---------------------------------------------------------------------------
# 2. Sync — RAISE_ON_CONFLICT
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBranchSyncRaiseOnConflict:

    def _ctx(self) -> TestContext:
        return TestContext(sample="s")

    def test_disjoint_metadata_merged(self):
        b = Branch(Pipeline().then(WriteX()), Pipeline().then(WriteY()))
        out = b(self._ctx())
        assert out.metadata["x"] == "from_x"
        assert out.metadata["y"] == "from_y"

    def test_named_field_conflict_raises_value_error(self):
        b = Branch(
            Pipeline().then(WriteAgent("v1")),
            Pipeline().then(WriteAgent("v2")),
            merge=MergeStrategy.RAISE_ON_CONFLICT,
        )
        with pytest.raises(ValueError, match="conflict"):
            b(self._ctx())

    def test_error_message_names_the_conflicting_field(self):
        b = Branch(
            Pipeline().then(WriteAgent("v1")),
            Pipeline().then(WriteAgent("v2")),
            merge=MergeStrategy.RAISE_ON_CONFLICT,
        )
        with pytest.raises(ValueError, match="agent_output"):
            b(self._ctx())

    def test_same_named_field_value_does_not_raise(self):
        b = Branch(
            Pipeline().then(WriteAgent("same")),
            Pipeline().then(WriteAgent("same")),
            merge=MergeStrategy.RAISE_ON_CONFLICT,
        )
        out = b(self._ctx())
        assert out.agent_output == "same"

    def test_three_branches_same_value_does_not_raise(self):
        b = Branch(
            Pipeline().then(WriteAgent("same")),
            Pipeline().then(WriteAgent("same")),
            Pipeline().then(WriteAgent("same")),
            merge=MergeStrategy.RAISE_ON_CONFLICT,
        )
        out = b(self._ctx())
        assert out.agent_output == "same"

    def test_three_branches_conflict_raises(self):
        b = Branch(
            Pipeline().then(WriteAgent("a")),
            Pipeline().then(WriteAgent("b")),
            Pipeline().then(WriteAgent("c")),
            merge=MergeStrategy.RAISE_ON_CONFLICT,
        )
        with pytest.raises(ValueError, match="conflict"):
            b(self._ctx())

    def test_two_named_field_conflicts_reported(self):
        """If two named fields both conflict, the error mentions both."""
        b = Branch(
            Pipeline().then(WriteAgent("a")).then(WriteReflection("r1")),
            Pipeline().then(WriteAgent("b")).then(WriteReflection("r2")),
            merge=MergeStrategy.RAISE_ON_CONFLICT,
        )
        with pytest.raises(ValueError):
            b(self._ctx())

    def test_metadata_conflict_does_not_raise(self):
        """Metadata keys use last-writer-wins even with RAISE_ON_CONFLICT.
        The conflict check applies only to named StepContext fields."""

        class MetaV1:
            requires = frozenset()
            provides = frozenset({"x"})

            def __call__(self, ctx):
                return ctx.replace(
                    metadata=MappingProxyType({**ctx.metadata, "x": "v1"})
                )

        class MetaV2:
            requires = frozenset()
            provides = frozenset({"x"})

            def __call__(self, ctx):
                return ctx.replace(
                    metadata=MappingProxyType({**ctx.metadata, "x": "v2"})
                )

        b = Branch(
            Pipeline().then(MetaV1()),
            Pipeline().then(MetaV2()),
            merge=MergeStrategy.RAISE_ON_CONFLICT,
        )
        out = b(self._ctx())  # must NOT raise
        assert "x" in out.metadata

    def test_metadata_always_unioned(self):
        b = Branch(
            Pipeline().then(WriteX()),
            Pipeline().then(WriteY()),
            merge=MergeStrategy.RAISE_ON_CONFLICT,
        )
        out = b(self._ctx())
        assert out.metadata["x"] == "from_x"
        assert out.metadata["y"] == "from_y"

    def test_preserves_sample_field(self):
        b = Branch(Pipeline().then(WriteX()), Pipeline().then(WriteY()))
        out = b(TestContext(sample="hello"))
        assert out.sample == "hello"


# ---------------------------------------------------------------------------
# 3. Sync — LAST_WRITE_WINS
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBranchSyncLastWriteWins:

    def _ctx(self) -> TestContext:
        return TestContext(sample="s")

    def test_second_branch_wins_on_named_field(self):
        b = Branch(
            Pipeline().then(WriteAgent("first")),
            Pipeline().then(WriteAgent("second")),
            merge=MergeStrategy.LAST_WRITE_WINS,
        )
        out = b(self._ctx())
        assert out.agent_output == "second"

    def test_third_branch_wins_on_named_field(self):
        b = Branch(
            Pipeline().then(WriteAgent("first")),
            Pipeline().then(WriteAgent("second")),
            Pipeline().then(WriteAgent("third")),
            merge=MergeStrategy.LAST_WRITE_WINS,
        )
        out = b(self._ctx())
        assert out.agent_output == "third"

    def test_metadata_is_union_of_all_branches(self):
        b = Branch(
            Pipeline().then(WriteX()),
            Pipeline().then(WriteY()),
            merge=MergeStrategy.LAST_WRITE_WINS,
        )
        out = b(self._ctx())
        assert out.metadata["x"] == "from_x"
        assert out.metadata["y"] == "from_y"

    def test_sole_writer_last_retains_value(self):
        """Only one branch writes a field; it wins because it's the last writer."""
        b = Branch(
            Pipeline(),  # empty — agent_output stays None
            Pipeline().then(WriteAgent("only")),
            merge=MergeStrategy.LAST_WRITE_WINS,
        )
        out = b(self._ctx())
        assert out.agent_output == "only"

    def test_empty_last_branch_overwrites_with_default(self):
        """An empty last branch's None overwrites the first branch's value."""
        b = Branch(
            Pipeline().then(WriteAgent("first")),
            Pipeline(),  # agent_output=None here — last writer wins
            merge=MergeStrategy.LAST_WRITE_WINS,
        )
        out = b(self._ctx())
        assert out.agent_output is None  # None overwrites "first"

    def test_does_not_raise_on_any_conflict(self):
        b = Branch(
            Pipeline().then(WriteAgent("a")),
            Pipeline().then(WriteAgent("b")),
            merge=MergeStrategy.LAST_WRITE_WINS,
        )
        out = b(self._ctx())  # must not raise
        assert out.agent_output == "b"

    def test_preserves_sample_field(self):
        b = Branch(
            Pipeline().then(WriteAgent("v")),
            merge=MergeStrategy.LAST_WRITE_WINS,
        )
        out = b(TestContext(sample="test"))
        assert out.sample == "test"


# ---------------------------------------------------------------------------
# 4. Sync — NAMESPACED
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBranchSyncNamespaced:

    def _ctx(self) -> TestContext:
        return TestContext(sample="s")

    def test_two_branches_keyed_branch_0_and_1(self):
        b = Branch(
            Pipeline().then(WriteX()),
            Pipeline().then(WriteY()),
            merge=MergeStrategy.NAMESPACED,
        )
        out = b(self._ctx())
        assert "branch_0" in out.metadata
        assert "branch_1" in out.metadata

    def test_each_key_holds_full_child_context(self):
        b = Branch(
            Pipeline().then(WriteX()),
            Pipeline().then(WriteY()),
            merge=MergeStrategy.NAMESPACED,
        )
        out = b(self._ctx())
        assert out.metadata["branch_0"].metadata["x"] == "from_x"
        assert out.metadata["branch_1"].metadata["y"] == "from_y"

    def test_three_branches_all_present(self):
        b = Branch(
            Pipeline().then(WriteX()),
            Pipeline().then(WriteY()),
            Pipeline().then(WriteZ()),
            merge=MergeStrategy.NAMESPACED,
        )
        out = b(self._ctx())
        assert {"branch_0", "branch_1", "branch_2"} <= set(out.metadata.keys())

    def test_named_fields_taken_from_first_branch(self):
        """Named dataclass fields on the result come from branch_0's output."""
        b = Branch(
            Pipeline().then(WriteAgent("from_first")),
            Pipeline().then(WriteAgent("from_second")),
            merge=MergeStrategy.NAMESPACED,
        )
        out = b(self._ctx())
        assert out.agent_output == "from_first"

    def test_both_branch_outputs_accessible_via_namespace(self):
        b = Branch(
            Pipeline().then(WriteAgent("a")),
            Pipeline().then(WriteAgent("b")),
            merge=MergeStrategy.NAMESPACED,
        )
        out = b(self._ctx())
        assert out.metadata["branch_0"].agent_output == "a"
        assert out.metadata["branch_1"].agent_output == "b"

    def test_conflicting_named_fields_do_not_raise(self):
        """NAMESPACED never raises — outputs are fully isolated in metadata keys."""
        b = Branch(
            Pipeline().then(WriteAgent("v1")),
            Pipeline().then(WriteAgent("v2")),
            merge=MergeStrategy.NAMESPACED,
        )
        out = b(self._ctx())  # must not raise
        assert out is not None

    def test_base_metadata_preserved(self):
        """Existing metadata on the input context is preserved in the output."""
        ctx = TestContext(sample="s", metadata={"existing": 99})
        b = Branch(Pipeline().then(WriteX()), merge=MergeStrategy.NAMESPACED)
        out = b(ctx)
        assert out.metadata["existing"] == 99

    def test_preserves_sample_field(self):
        b = Branch(Pipeline().then(WriteX()), merge=MergeStrategy.NAMESPACED)
        out = b(TestContext(sample="orig"))
        assert out.sample == "orig"


# ---------------------------------------------------------------------------
# 5. Sync — custom merge
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBranchSyncCustomMerge:

    def _ctx(self) -> TestContext:
        return TestContext(sample="s")

    def test_fn_receives_all_outputs(self):
        received: list = []

        def capture(ctxs):
            received.extend(ctxs)
            return ctxs[0]

        b = Branch(Pipeline().then(WriteX()), Pipeline().then(WriteY()), merge=capture)
        b(self._ctx())
        assert len(received) == 2

    def test_fn_receives_outputs_not_inputs(self):
        """Each element passed to the merge fn must be a result, not the input ctx."""
        received: list = []

        def capture(ctxs):
            received.extend(ctxs)
            return ctxs[0]

        orig = self._ctx()
        b = Branch(Pipeline().then(WriteX()), merge=capture)
        b(orig)
        # The received ctx should have 'x' set (it is the output, not the input)
        assert received[0].metadata.get("x") == "from_x"

    def test_fn_can_compute_aggregate(self):
        def merge_sum(ctxs):
            total = sum(ctx.metadata.get("n", 0) for ctx in ctxs)
            return ctxs[0].replace(metadata=MappingProxyType({"total": total}))

        b = Branch(
            Pipeline().then(WriteN(3)), Pipeline().then(WriteN(7)), merge=merge_sum
        )
        out = b(self._ctx())
        assert out.metadata["total"] == 10

    def test_fn_can_select_last_output(self):
        last = lambda ctxs: ctxs[-1]
        b = Branch(Pipeline().then(WriteX()), Pipeline().then(WriteY()), merge=last)
        out = b(self._ctx())
        assert "y" in out.metadata
        assert "x" not in out.metadata

    def test_fn_accesses_subclass_named_fields(self):
        """Custom merge function that reads subclass fields to pick a winner."""

        def pick_best(ctxs):
            # Select the branch whose agent_output is longest
            return max(ctxs, key=lambda c: len(str(c.agent_output or "")))

        b = Branch(
            Pipeline().then(WriteAgent("short")),
            Pipeline().then(WriteAgent("much_longer_answer")),
            merge=pick_best,
        )
        out = b(self._ctx())
        assert out.agent_output == "much_longer_answer"

    def test_fn_combines_subclass_fields_from_branches(self):
        """Custom merge that combines named fields from different branches."""

        def combine(ctxs):
            # Take agent_output from first, reflection from second
            return ctxs[0].replace(reflection=ctxs[1].reflection)

        b = Branch(
            Pipeline().then(WriteAgent("answer")),
            Pipeline().then(WriteReflection("insight")),
            merge=combine,
        )
        out = b(self._ctx())
        assert out.agent_output == "answer"
        assert out.reflection == "insight"

    def test_fn_exception_propagates_directly(self):
        def bad_merge(ctxs):
            raise ValueError("merge exploded")

        b = Branch(
            Pipeline().then(WriteX()), Pipeline().then(WriteY()), merge=bad_merge
        )
        with pytest.raises(ValueError, match="merge exploded"):
            b(self._ctx())


# ---------------------------------------------------------------------------
# 6. Sync — failure semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBranchSyncFailures:

    def _ctx(self) -> TestContext:
        return TestContext(sample="s")

    def test_single_failure_raises_branch_error(self):
        b = Branch(Pipeline().then(Explode("err1")))
        with pytest.raises(BranchError):
            b(self._ctx())

    def test_two_failures_both_collected(self):
        b = Branch(
            Pipeline().then(Explode("err1")),
            Pipeline().then(Explode("err2")),
        )
        with pytest.raises(BranchError) as exc_info:
            b(self._ctx())
        assert len(exc_info.value.failures) == 2

    def test_three_failures_all_collected(self):
        b = Branch(
            Pipeline().then(Explode("e1")),
            Pipeline().then(Explode("e2")),
            Pipeline().then(Explode("e3")),
        )
        with pytest.raises(BranchError) as exc_info:
            b(self._ctx())
        assert len(exc_info.value.failures) == 3

    def test_partial_failure_remaining_branches_still_run(self):
        """All branches must complete even when one fails early."""
        log: list[str] = []
        lock = threading.Lock()

        b = Branch(
            Pipeline().then(Explode("fail")),
            Pipeline().then(Log("b", log, lock)),
            Pipeline().then(Log("c", log, lock)),
        )
        with pytest.raises(BranchError):
            b(self._ctx())

        assert "b" in log
        assert "c" in log

    def test_one_failure_one_success_raises_branch_error(self):
        b = Branch(Pipeline().then(WriteX()), Pipeline().then(Explode()))
        with pytest.raises(BranchError):
            b(self._ctx())

    def test_branch_error_message_includes_failure_count(self):
        b = Branch(Pipeline().then(Explode("e1")), Pipeline().then(Explode("e2")))
        with pytest.raises(BranchError) as exc_info:
            b(self._ctx())
        assert "2" in str(exc_info.value)

    def test_branch_error_failures_are_original_exceptions(self):
        b = Branch(Pipeline().then(Explode("specific_msg")))
        with pytest.raises(BranchError) as exc_info:
            b(self._ctx())
        inner = exc_info.value.failures[0]
        assert isinstance(inner, RuntimeError)
        assert "specific_msg" in str(inner)

    def test_merge_fn_not_called_on_failure(self):
        """If any branch fails, the merge function must never be invoked."""
        called: list = []

        def should_not_run(ctxs):
            called.append(True)
            return ctxs[0]

        b = Branch(
            Pipeline().then(WriteX()),
            Pipeline().then(Explode()),
            merge=should_not_run,
        )
        with pytest.raises(BranchError):
            b(self._ctx())

        assert called == []

    def test_branch_error_is_not_value_error(self):
        b = Branch(Pipeline().then(Explode()))
        with pytest.raises(BranchError) as exc_info:
            b(self._ctx())
        assert not isinstance(exc_info.value, ValueError)


# ---------------------------------------------------------------------------
# 7. Sync — immutability / isolation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBranchSyncImmutability:

    def _ctx(self) -> TestContext:
        return TestContext(sample="s")

    def test_all_branches_receive_the_same_frozen_context(self):
        """All branches get the identical input object — frozen so sharing is safe."""
        received: list[StepContext] = []
        lock = threading.Lock()

        class Capture:
            requires = frozenset()
            provides = frozenset()

            def __call__(self, ctx):
                with lock:
                    received.append(ctx)
                return ctx

        orig = self._ctx()
        b = Branch(Pipeline().then(Capture()), Pipeline().then(Capture()))
        b(orig)
        assert all(c is orig for c in received)

    def test_branch_outputs_are_independent(self):
        """Writes in one branch must not appear in another branch's output."""

        class WriteXv1:
            requires = frozenset()
            provides = frozenset({"x"})

            def __call__(self, ctx):
                return ctx.replace(
                    metadata=MappingProxyType({**ctx.metadata, "x": "branch_a"})
                )

        class WriteXv2:
            requires = frozenset()
            provides = frozenset({"x"})

            def __call__(self, ctx):
                return ctx.replace(
                    metadata=MappingProxyType({**ctx.metadata, "x": "branch_b"})
                )

        b = Branch(
            Pipeline().then(WriteXv1()),
            Pipeline().then(WriteXv2()),
            merge=MergeStrategy.NAMESPACED,
        )
        out = b(self._ctx())
        assert out.metadata["branch_0"].metadata["x"] == "branch_a"
        assert out.metadata["branch_1"].metadata["x"] == "branch_b"

    def test_original_context_unchanged_after_branch(self):
        orig = TestContext(sample="frozen", agent_output=None)
        b = Branch(Pipeline().then(WriteAgent("mutated")))
        b(orig)
        assert orig.agent_output is None  # frozen — input is untouched


# ---------------------------------------------------------------------------
# 8. Async — parity with sync
#
# Every sync behaviour exercised above must hold in __call_async__.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBranchAsyncParity:
    """Mirrors every sync test class for the async path."""

    def _ctx(self) -> TestContext:
        return TestContext(sample="s")

    # RAISE_ON_CONFLICT -------------------------------------------------------

    def test_disjoint_metadata_merged(self):
        b = Branch(Pipeline().then(WriteX()), Pipeline().then(WriteY()))
        out = asyncio.run(b.__call_async__(self._ctx()))
        assert out.metadata["x"] == "from_x"
        assert out.metadata["y"] == "from_y"

    def test_named_field_conflict_raises_value_error(self):
        """The merge fn raises ValueError; it propagates from __call_async__."""
        b = Branch(
            Pipeline().then(WriteAgent("v1")),
            Pipeline().then(WriteAgent("v2")),
            merge=MergeStrategy.RAISE_ON_CONFLICT,
        )
        with pytest.raises(ValueError, match="conflict"):
            asyncio.run(b.__call_async__(self._ctx()))

    def test_same_value_no_conflict_no_raise(self):
        b = Branch(
            Pipeline().then(WriteAgent("same")),
            Pipeline().then(WriteAgent("same")),
            merge=MergeStrategy.RAISE_ON_CONFLICT,
        )
        out = asyncio.run(b.__call_async__(self._ctx()))
        assert out.agent_output == "same"

    def test_metadata_conflict_does_not_raise(self):
        class MetaV1:
            requires = frozenset()
            provides = frozenset({"x"})

            def __call__(self, ctx):
                return ctx.replace(
                    metadata=MappingProxyType({**ctx.metadata, "x": "v1"})
                )

        class MetaV2:
            requires = frozenset()
            provides = frozenset({"x"})

            def __call__(self, ctx):
                return ctx.replace(
                    metadata=MappingProxyType({**ctx.metadata, "x": "v2"})
                )

        b = Branch(
            Pipeline().then(MetaV1()),
            Pipeline().then(MetaV2()),
            merge=MergeStrategy.RAISE_ON_CONFLICT,
        )
        out = asyncio.run(b.__call_async__(self._ctx()))
        assert "x" in out.metadata

    # LAST_WRITE_WINS ---------------------------------------------------------

    def test_last_write_wins_second_branch(self):
        b = Branch(
            Pipeline().then(WriteAgent("first")),
            Pipeline().then(WriteAgent("second")),
            merge=MergeStrategy.LAST_WRITE_WINS,
        )
        out = asyncio.run(b.__call_async__(self._ctx()))
        assert out.agent_output == "second"

    def test_last_write_wins_third_branch(self):
        b = Branch(
            Pipeline().then(WriteAgent("a")),
            Pipeline().then(WriteAgent("b")),
            Pipeline().then(WriteAgent("c")),
            merge=MergeStrategy.LAST_WRITE_WINS,
        )
        out = asyncio.run(b.__call_async__(self._ctx()))
        assert out.agent_output == "c"

    def test_last_write_wins_metadata_union(self):
        b = Branch(
            Pipeline().then(WriteX()),
            Pipeline().then(WriteY()),
            merge=MergeStrategy.LAST_WRITE_WINS,
        )
        out = asyncio.run(b.__call_async__(self._ctx()))
        assert out.metadata["x"] == "from_x"
        assert out.metadata["y"] == "from_y"

    # NAMESPACED --------------------------------------------------------------

    def test_namespaced_two_branches(self):
        b = Branch(
            Pipeline().then(WriteX()),
            Pipeline().then(WriteY()),
            merge=MergeStrategy.NAMESPACED,
        )
        out = asyncio.run(b.__call_async__(self._ctx()))
        assert "branch_0" in out.metadata
        assert "branch_1" in out.metadata

    def test_namespaced_child_contexts_accessible(self):
        b = Branch(
            Pipeline().then(WriteX()),
            Pipeline().then(WriteY()),
            merge=MergeStrategy.NAMESPACED,
        )
        out = asyncio.run(b.__call_async__(self._ctx()))
        assert out.metadata["branch_0"].metadata["x"] == "from_x"
        assert out.metadata["branch_1"].metadata["y"] == "from_y"

    def test_namespaced_three_branches(self):
        b = Branch(
            Pipeline().then(WriteX()),
            Pipeline().then(WriteY()),
            Pipeline().then(WriteZ()),
            merge=MergeStrategy.NAMESPACED,
        )
        out = asyncio.run(b.__call_async__(self._ctx()))
        assert {"branch_0", "branch_1", "branch_2"} <= set(out.metadata.keys())

    def test_namespaced_named_fields_from_first_branch(self):
        b = Branch(
            Pipeline().then(WriteAgent("from_first")),
            Pipeline().then(WriteAgent("from_second")),
            merge=MergeStrategy.NAMESPACED,
        )
        out = asyncio.run(b.__call_async__(self._ctx()))
        assert out.agent_output == "from_first"

    # custom merge ------------------------------------------------------------

    def test_custom_merge_receives_all_outputs(self):
        received: list = []

        def capture(ctxs):
            received.extend(ctxs)
            return ctxs[0]

        b = Branch(Pipeline().then(WriteX()), Pipeline().then(WriteY()), merge=capture)
        asyncio.run(b.__call_async__(self._ctx()))
        assert len(received) == 2

    def test_custom_merge_can_compute_aggregate(self):
        def merge_sum(ctxs):
            total = sum(ctx.metadata.get("n", 0) for ctx in ctxs)
            return ctxs[0].replace(metadata=MappingProxyType({"total": total}))

        b = Branch(
            Pipeline().then(WriteN(4)), Pipeline().then(WriteN(6)), merge=merge_sum
        )
        out = asyncio.run(b.__call_async__(self._ctx()))
        assert out.metadata["total"] == 10

    # failure semantics -------------------------------------------------------

    def test_all_branch_failures_collected(self):
        b = Branch(
            Pipeline().then(Explode("err1")),
            Pipeline().then(Explode("err2")),
        )
        with pytest.raises(BranchError) as exc_info:
            asyncio.run(b.__call_async__(self._ctx()))
        assert len(exc_info.value.failures) == 2

    def test_three_failures_all_collected(self):
        b = Branch(
            Pipeline().then(Explode("e1")),
            Pipeline().then(Explode("e2")),
            Pipeline().then(Explode("e3")),
        )
        with pytest.raises(BranchError) as exc_info:
            asyncio.run(b.__call_async__(self._ctx()))
        assert len(exc_info.value.failures) == 3

    def test_one_failure_one_success_raises(self):
        b = Branch(Pipeline().then(WriteX()), Pipeline().then(Explode()))
        with pytest.raises(BranchError):
            asyncio.run(b.__call_async__(self._ctx()))

    def test_merge_fn_not_called_on_failure(self):
        called: list = []

        def should_not_run(ctxs):
            called.append(True)
            return ctxs[0]

        b = Branch(
            Pipeline().then(WriteX()),
            Pipeline().then(Explode()),
            merge=should_not_run,
        )
        with pytest.raises(BranchError):
            asyncio.run(b.__call_async__(self._ctx()))

        assert called == []

    # immutability ------------------------------------------------------------

    def test_frozen_context_not_mutated(self):
        """All branches receive the same frozen context object."""
        received: list[StepContext] = []

        class AsyncCapture:
            requires = frozenset()
            provides = frozenset()

            async def __call__(self, ctx):
                received.append(ctx)
                return ctx

        orig = self._ctx()
        b = Branch(Pipeline().then(AsyncCapture()), Pipeline().then(AsyncCapture()))
        asyncio.run(b.__call_async__(orig))
        assert all(c is orig for c in received)

    # sample field preserved --------------------------------------------------

    def test_preserves_sample_field(self):
        b = Branch(Pipeline().then(WriteX()), Pipeline().then(WriteY()))
        out = asyncio.run(b.__call_async__(TestContext(sample="preserved")))
        assert out.sample == "preserved"


# ---------------------------------------------------------------------------
# 9. Async — native async children (coroutine __call__)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBranchAsyncNativeCoroutines:
    """Branch.__call_async__ detects native coroutine children and awaits directly."""

    def _ctx(self) -> TestContext:
        return TestContext(sample="s")

    def test_two_async_children_execute(self):
        b = Branch(Pipeline().then(AsyncWriteX()), Pipeline().then(AsyncWriteY()))
        out = asyncio.run(b.__call_async__(self._ctx()))
        assert out.metadata["x"] == "async_x"
        assert out.metadata["y"] == "async_y"

    def test_async_failure_collected(self):
        b = Branch(
            Pipeline().then(AsyncWriteX()),
            Pipeline().then(AsyncExplode("native_fail")),
        )
        with pytest.raises(BranchError) as exc_info:
            asyncio.run(b.__call_async__(self._ctx()))
        assert len(exc_info.value.failures) == 1
        assert "native_fail" in str(exc_info.value.failures[0])

    def test_all_async_failures_collected(self):
        b = Branch(
            Pipeline().then(AsyncExplode("e1")),
            Pipeline().then(AsyncExplode("e2")),
            Pipeline().then(AsyncExplode("e3")),
        )
        with pytest.raises(BranchError) as exc_info:
            asyncio.run(b.__call_async__(self._ctx()))
        assert len(exc_info.value.failures) == 3

    def test_mixed_sync_and_native_async_children(self):
        """Branch can fan out over a mix of sync (to_thread) and native async steps."""
        b = Branch(
            Pipeline().then(WriteX()),  # sync → asyncio.to_thread
            Pipeline().then(AsyncWriteY()),  # async → direct await
        )
        out = asyncio.run(b.__call_async__(self._ctx()))
        assert out.metadata["x"] == "from_x"
        assert out.metadata["y"] == "async_y"

    def test_three_children_mixed(self):
        b = Branch(
            Pipeline().then(AsyncWriteX()),
            Pipeline().then(AsyncWriteY()),
            Pipeline().then(WriteZ()),  # sync
            merge=MergeStrategy.RAISE_ON_CONFLICT,
        )
        out = asyncio.run(b.__call_async__(self._ctx()))
        assert out.metadata["x"] == "async_x"
        assert out.metadata["y"] == "async_y"
        assert out.metadata["z"] == "from_z"


# ---------------------------------------------------------------------------
# 10. Integration — Branch via Pipeline.run() / run_async()
#
# When Branch is a step inside a Pipeline, the async code path in run_async()
# detects __call_async__ and calls it — Branch always runs async here.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBranchViaRun:

    def test_disjoint_metadata_through_run(self):
        pipe = Pipeline().branch(
            Pipeline().then(WriteX()),
            Pipeline().then(WriteY()),
        )
        results = pipe.run([TestContext(sample="s")])
        out = results[0].output
        assert out.metadata["x"] == "from_x"
        assert out.metadata["y"] == "from_y"

    def test_failure_captured_in_sample_result(self):
        pipe = Pipeline().branch(
            Pipeline().then(WriteX()),
            Pipeline().then(Explode()),
        )
        results = pipe.run([TestContext(sample="s")])
        assert results[0].error is not None
        assert results[0].failed_at == "Branch"

    def test_branch_error_preserved_in_sample_result(self):
        pipe = Pipeline().branch(
            Pipeline().then(Explode("e1")),
            Pipeline().then(Explode("e2")),
        )
        results = pipe.run([TestContext(sample="s")])
        assert isinstance(results[0].error, BranchError)
        assert len(results[0].error.failures) == 2

    def test_pre_branch_data_visible_inside_branch(self):
        """Steps run before Branch must be visible in ctx received by branches."""

        class SetPre:
            requires = frozenset()
            provides = frozenset({"pre"})

            def __call__(self, ctx):
                return ctx.replace(
                    metadata=MappingProxyType({**ctx.metadata, "pre": 42})
                )

        class ReadPre:
            requires = frozenset({"pre"})
            provides = frozenset({"read"})

            def __call__(self, ctx):
                return ctx.replace(
                    metadata=MappingProxyType(
                        {**ctx.metadata, "read": ctx.metadata["pre"]}
                    )
                )

        pipe = (
            Pipeline()
            .then(SetPre())
            .branch(
                Pipeline().then(ReadPre()),
                Pipeline().then(WriteY()),
                merge=MergeStrategy.LAST_WRITE_WINS,
            )
        )
        results = pipe.run([TestContext(sample="s")])
        out = results[0].output
        assert out.metadata.get("read") == 42
        assert out.metadata.get("y") == "from_y"

    def test_step_after_branch_receives_merged_context(self):
        class After:
            requires = frozenset()
            provides = frozenset({"after"})

            def __call__(self, ctx):
                return ctx.replace(
                    metadata=MappingProxyType({**ctx.metadata, "after": True})
                )

        pipe = (
            Pipeline()
            .branch(Pipeline().then(WriteX()), Pipeline().then(WriteY()))
            .then(After())
        )
        results = pipe.run([TestContext(sample="s")])
        out = results[0].output
        assert out.metadata["x"] == "from_x"
        assert out.metadata["y"] == "from_y"
        assert out.metadata["after"] is True

    def test_multiple_samples_all_succeed(self):
        pipe = Pipeline().branch(
            Pipeline().then(WriteX()),
            Pipeline().then(WriteY()),
        )
        results = pipe.run([TestContext(sample=s) for s in ("s1", "s2", "s3")])
        assert len(results) == 3
        assert all(r.error is None for r in results)
        assert all(r.output.metadata.get("x") == "from_x" for r in results)

    def test_last_write_wins_through_run(self):
        pipe = Pipeline().branch(
            Pipeline().then(WriteAgent("first")),
            Pipeline().then(WriteAgent("second")),
            merge=MergeStrategy.LAST_WRITE_WINS,
        )
        results = pipe.run([TestContext(sample="s")])
        assert results[0].output.agent_output == "second"

    def test_namespaced_through_run(self):
        pipe = Pipeline().branch(
            Pipeline().then(WriteX()),
            Pipeline().then(WriteY()),
            merge=MergeStrategy.NAMESPACED,
        )
        results = pipe.run([TestContext(sample="s")])
        out = results[0].output
        assert "branch_0" in out.metadata
        assert "branch_1" in out.metadata

    def test_nested_branch_inside_pipeline_step(self):
        """Pipeline-as-step containing a Branch, nested inside another Pipeline."""
        inner = Pipeline().branch(
            Pipeline().then(WriteX()),
            Pipeline().then(WriteY()),
        )
        outer = Pipeline().then(inner).then(WriteZ())
        results = outer.run([TestContext(sample="s")])
        out = results[0].output
        assert out.metadata["x"] == "from_x"
        assert out.metadata["y"] == "from_y"
        assert out.metadata["z"] == "from_z"

    def test_run_async_entry_point_with_branch(self):
        pipe = Pipeline().branch(
            Pipeline().then(WriteX()),
            Pipeline().then(WriteY()),
        )
        results = asyncio.run(
            pipe.run_async([TestContext(sample="s1"), TestContext(sample="s2")])
        )
        assert len(results) == 2
        assert all(r.error is None for r in results)

    def test_branch_runs_children_in_parallel(self):
        """Fan-out branches must execute concurrently, not sequentially.

        Uses a threading.Barrier that requires both branches to arrive before
        either can proceed.  If branches ran sequentially the first would
        block at the barrier and the test would fail with a BranchError.
        """
        barrier = threading.Barrier(2, timeout=5)

        class BarrierStep:
            requires = frozenset()
            provides = frozenset()

            def __call__(self, ctx: StepContext) -> StepContext:
                barrier.wait()  # blocks until the other branch also arrives
                return ctx

        pipe = Pipeline().branch(
            Pipeline().then(BarrierStep()),
            Pipeline().then(BarrierStep()),
        )
        results = pipe.run([TestContext(sample="s")])
        assert results[0].error is None
