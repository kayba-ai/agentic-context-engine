"""Branch — parallel fork/join step."""

from __future__ import annotations

import asyncio
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from types import MappingProxyType
from typing import Callable

from .context import StepContext
from .errors import BranchError


class MergeStrategy(Enum):
    """Built-in merge strategies for Branch outputs."""

    RAISE_ON_CONFLICT = "raise_on_conflict"
    LAST_WRITE_WINS = "last_write_wins"
    NAMESPACED = "namespaced"


# ---------------------------------------------------------------------------
# Built-in merge functions
# ---------------------------------------------------------------------------

def _merge_raise_on_conflict(ctxs: list[StepContext]) -> StepContext:
    """Raise if any two branches wrote different values for the same field.

    Metadata is always merged (union across all branches; last writer wins
    within metadata — there is no named-field semantic there).
    """
    if len(ctxs) == 1:
        return ctxs[0]

    conflicts: set[str] = set()
    for f in dataclasses.fields(StepContext):
        if f.name == "metadata":
            continue
        first_val = getattr(ctxs[0], f.name)
        if any(getattr(ctx, f.name) != first_val for ctx in ctxs[1:]):
            conflicts.add(f.name)

    if conflicts:
        raise ValueError(
            f"Branch outputs conflict on fields {conflicts!r}. "
            "Use a different merge strategy or ensure branches write disjoint fields."
        )

    merged_meta: dict = {}
    for ctx in ctxs:
        merged_meta.update(ctx.metadata)

    return dataclasses.replace(ctxs[0], metadata=MappingProxyType(merged_meta))


def _merge_last_write_wins(ctxs: list[StepContext]) -> StepContext:
    """Last branch's value wins for every conflicting field."""
    if len(ctxs) == 1:
        return ctxs[0]

    # Start from first context, overlay with each subsequent one
    result = ctxs[0]
    for ctx in ctxs[1:]:
        changes: dict = {}
        for f in dataclasses.fields(StepContext):
            if f.name == "metadata":
                continue
            val = getattr(ctx, f.name)
            if val != getattr(result, f.name):
                changes[f.name] = val
        if changes:
            result = dataclasses.replace(result, **changes)

    merged_meta: dict = {}
    for ctx in ctxs:
        merged_meta.update(ctx.metadata)

    return dataclasses.replace(result, metadata=MappingProxyType(merged_meta))


def _merge_namespaced(ctxs: list[StepContext]) -> StepContext:
    """Each branch's output is stored at ``ctx.metadata["branch_N"]``.

    Named fields are taken from the first branch; no conflict is possible
    because branch outputs are kept in separate metadata keys.
    """
    base = ctxs[0]
    extra: dict = {f"branch_{i}": ctx for i, ctx in enumerate(ctxs)}
    merged_meta = MappingProxyType({**base.metadata, **extra})
    return dataclasses.replace(base, metadata=merged_meta)


_BUILTIN_MERGES: dict[MergeStrategy, Callable] = {
    MergeStrategy.RAISE_ON_CONFLICT: _merge_raise_on_conflict,
    MergeStrategy.LAST_WRITE_WINS: _merge_last_write_wins,
    MergeStrategy.NAMESPACED: _merge_namespaced,
}


# ---------------------------------------------------------------------------
# Branch
# ---------------------------------------------------------------------------

class Branch:
    """Runs multiple pipelines in parallel, then merges their outputs.

    ``Branch`` satisfies ``StepProtocol`` — it can be used wherever a step
    is expected.  ``requires`` and ``provides`` are inferred from the union
    of the child pipelines' contracts.

    In sync contexts (called directly), fan-out is via
    ``ThreadPoolExecutor``.  In async contexts (awaited), fan-out is via
    ``asyncio.gather``.

    All branches always run to completion before any failure is raised —
    ``BranchError`` carries the full list of failures.
    """

    def __init__(
        self,
        *pipelines: object,
        merge: MergeStrategy | Callable = MergeStrategy.RAISE_ON_CONFLICT,
    ) -> None:
        if not pipelines:
            raise ValueError("Branch requires at least one child pipeline.")

        self.pipelines = list(pipelines)

        if callable(merge) and not isinstance(merge, MergeStrategy):
            self._merge_fn: Callable = merge
        else:
            self._merge_fn = _BUILTIN_MERGES[merge]  # type: ignore[index]

        # Infer requires/provides from the union of child contracts
        all_requires: set[str] = set()
        all_provides: set[str] = set()
        for p in self.pipelines:
            all_requires |= set(getattr(p, "requires", frozenset()))
            all_provides |= set(getattr(p, "provides", frozenset()))

        self.requires: frozenset[str] = frozenset(all_requires)
        self.provides: frozenset[str] = frozenset(all_provides)

    # ------------------------------------------------------------------
    # Sync execution
    # ------------------------------------------------------------------

    def __call__(self, ctx: StepContext) -> StepContext:
        """Sync fan-out via ThreadPoolExecutor.

        All branches receive the same (frozen) context — no copy needed.
        All branches run to completion before any failure is raised.
        """
        with ThreadPoolExecutor(max_workers=len(self.pipelines)) as executor:
            futures = [executor.submit(p, ctx) for p in self.pipelines]
            results: list[StepContext] = []
            failures: list[BaseException] = []
            for f in futures:
                try:
                    results.append(f.result())
                except BaseException as exc:  # noqa: BLE001
                    failures.append(exc)

        if failures:
            raise BranchError(failures)

        return self._merge_fn(results)

    # ------------------------------------------------------------------
    # Async execution
    # ------------------------------------------------------------------

    async def __call_async__(self, ctx: StepContext) -> StepContext:
        """Async fan-out via asyncio.gather.

        ``return_exceptions=True`` guarantees all branches run to completion
        even when one fails; the full failure list is surfaced via
        ``BranchError``.

        Sync child pipelines are wrapped with ``asyncio.to_thread`` so they
        run in a thread pool rather than blocking the event loop.
        """

        async def _run_child(child: object) -> StepContext:
            if asyncio.iscoroutinefunction(getattr(child, "__call__", None)):
                return await child(ctx)  # type: ignore[operator]
            if hasattr(child, "__call_async__"):
                return await child.__call_async__(ctx)  # type: ignore[union-attr]
            # Sync callable — run in thread pool so it doesn't block the loop
            return await asyncio.to_thread(child, ctx)  # type: ignore[arg-type]

        raw = await asyncio.gather(
            *[_run_child(p) for p in self.pipelines],
            return_exceptions=True,
        )
        failures = [r for r in raw if isinstance(r, BaseException)]
        if failures:
            raise BranchError(failures)
        return self._merge_fn([r for r in raw if not isinstance(r, BaseException)])
