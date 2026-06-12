"""ReflectionEnsembleStep — run N independent reflections for one trace."""

from __future__ import annotations

from types import MappingProxyType

from pipeline import Pipeline
from pipeline.protocol import StepProtocol

from ..core.context import ACEStepContext
from ..core.outputs import ReflectorOutput
from ..protocols import ReflectorLike
from .reflect import ReflectStep


class ReflectionEnsembleStep:
    """Run the same trace through a reflector multiple times.

    This is a map-reduce step: one input context is expanded into
    ``ensemble_size`` sub-contexts, each sub-context runs the normal
    reflection step through the pipeline engine, and the resulting
    ``ReflectorOutput`` objects are flattened back onto the original
    context as ``ctx.reflections``.
    """

    requires = frozenset({"trace", "skillbook"})
    provides = frozenset({"reflections"})

    async_boundary = True
    max_workers = 1

    def __init__(
        self,
        reflector: ReflectorLike | StepProtocol[ACEStepContext],
        *,
        ensemble_size: int = 2,
        workers: int | None = None,
    ) -> None:
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1.")
        if workers is not None and workers < 1:
            raise ValueError("workers must be >= 1.")

        self.reflector = reflector
        self.ensemble_size = ensemble_size
        self.workers = workers or min(ensemble_size, ReflectStep.max_workers)
        self._reflect_step = self._coerce_reflect_step(reflector)

    @staticmethod
    def _coerce_reflect_step(
        reflector: ReflectorLike | StepProtocol[ACEStepContext],
    ) -> StepProtocol[ACEStepContext]:
        provides = getattr(reflector, "provides", ())
        if callable(reflector) and "reflections" in provides:
            return reflector  # type: ignore[return-value]
        return ReflectStep(reflector)  # type: ignore[arg-type]

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        subcontexts = [
            ctx.replace(
                metadata=MappingProxyType(
                    {
                        **ctx.metadata,
                        "reflection_ensemble_index": index,
                        "reflection_ensemble_size": self.ensemble_size,
                    }
                )
            )
            for index in range(self.ensemble_size)
        ]

        pipe = Pipeline().then(self._reflect_step)
        results = pipe.run(subcontexts, workers=self.workers)
        pipe.wait_for_background()

        reflections: list[ReflectorOutput] = []
        for index, result in enumerate(results):
            if result.error is not None:
                raise RuntimeError(
                    f"Reflection ensemble member {index} failed."
                ) from result.error
            if result.output is None:
                raise RuntimeError(
                    f"Reflection ensemble member {index} produced no output."
                )
            reflections.extend(result.output.reflections)

        if not reflections:
            raise RuntimeError("Reflection ensemble produced no reflections.")

        return ctx.replace(
            reflections=tuple(reflections),
            metadata=MappingProxyType(
                {
                    **ctx.metadata,
                    "reflection_ensemble_size": self.ensemble_size,
                    "reflection_ensemble_completed": len(reflections),
                }
            ),
        )
