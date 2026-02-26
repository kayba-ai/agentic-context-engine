"""Shared fixtures and reusable dummy steps for pipeline engine tests.

No ACE imports — every step here is a generic dummy that only uses the
pipeline primitives (StepContext, StepProtocol).
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import MappingProxyType

import pytest

from pipeline import StepContext

# ---------------------------------------------------------------------------
# Reusable dummy step classes (no ACE knowledge)
# ---------------------------------------------------------------------------


class Noop:
    """Pass-through step — does not change context."""

    requires = frozenset()
    provides = frozenset()

    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx


class SetA:
    """Writes metadata['a'] = 1.  No requirements."""

    requires = frozenset()
    provides = frozenset({"a"})

    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx.replace(metadata=MappingProxyType({**ctx.metadata, "a": 1}))


class SetB:
    """Reads 'a', writes metadata['b'] = metadata['a'] + 1."""

    requires = frozenset({"a"})
    provides = frozenset({"b"})

    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx.replace(
            metadata=MappingProxyType({**ctx.metadata, "b": ctx.metadata["a"] + 1})
        )


class SetC:
    """Reads 'b', writes metadata['c'] = metadata['b'] * 2."""

    requires = frozenset({"b"})
    provides = frozenset({"c"})

    def __call__(self, ctx: StepContext) -> StepContext:
        return ctx.replace(
            metadata=MappingProxyType({**ctx.metadata, "c": ctx.metadata["b"] * 2})
        )


class Boom:
    """Always raises RuntimeError."""

    requires = frozenset()
    provides = frozenset()

    def __call__(self, ctx: StepContext) -> StepContext:
        raise RuntimeError("boom")


class Slow:
    """Sleeps for *delay* seconds then sets metadata['done'] = True."""

    requires = frozenset()
    provides = frozenset({"done"})

    def __init__(self, delay: float = 0.05):
        self.delay = delay

    def __call__(self, ctx: StepContext) -> StepContext:
        time.sleep(self.delay)
        return ctx.replace(metadata=MappingProxyType({**ctx.metadata, "done": True}))


class AsyncStep:
    """Async step — sets metadata['async'] = True."""

    requires = frozenset()
    provides = frozenset({"async_done"})

    async def __call__(self, ctx: StepContext) -> StepContext:
        await asyncio.sleep(0)  # yield to event loop
        return ctx.replace(
            metadata=MappingProxyType({**ctx.metadata, "async_done": True})
        )


class Recorder:
    """Records every ctx it receives via call_log (thread-safe)."""

    requires = frozenset()
    provides = frozenset()

    def __init__(self):
        self.call_log: list[StepContext] = []
        self._lock = threading.Lock()

    def __call__(self, ctx: StepContext) -> StepContext:
        with self._lock:
            self.call_log.append(ctx)
        return ctx


class BoundaryStep:
    """Foreground step that marks the async_boundary handoff."""

    requires = frozenset()
    provides = frozenset({"bg_result"})
    async_boundary = True
    max_workers = 2

    def __call__(self, ctx: StepContext) -> StepContext:
        time.sleep(0.01)  # simulate background work
        return ctx.replace(
            metadata=MappingProxyType({**ctx.metadata, "bg_result": True})
        )


class SlowBoundaryStep:
    """Slow boundary step for timeout testing."""

    requires = frozenset()
    provides = frozenset({"slow_bg"})
    async_boundary = True
    max_workers = 1

    def __call__(self, ctx: StepContext) -> StepContext:
        time.sleep(2.0)  # intentionally slow
        return ctx.replace(metadata=MappingProxyType({**ctx.metadata, "slow_bg": True}))


class SerialStep:
    """Background step that must serialize (max_workers=1).  Appends to a shared log."""

    requires = frozenset()
    provides = frozenset({"serial_done"})
    max_workers = 1
    _log: list[str] = []
    _log_lock = threading.Lock()

    def __call__(self, ctx: StepContext) -> StepContext:
        with self._log_lock:
            SerialStep._log.append(f"start-{ctx.sample}")
            time.sleep(0.02)  # ensure ordering is visible if concurrent
            SerialStep._log.append(f"end-{ctx.sample}")
        return ctx.replace(
            metadata=MappingProxyType({**ctx.metadata, "serial_done": True})
        )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def noop():
    return Noop()


@pytest.fixture
def set_a():
    return SetA()


@pytest.fixture
def set_b():
    return SetB()


@pytest.fixture
def set_c():
    return SetC()


@pytest.fixture
def boom():
    return Boom()


@pytest.fixture
def recorder():
    return Recorder()


@pytest.fixture
def base_ctx():
    """A minimal StepContext with sample='test'."""
    return StepContext(sample="test")
