"""``MeteredModel`` — a pydantic-ai ``WrapperModel`` that fires a usage hook.

Wraps any pydantic-ai ``Model`` and invokes ``callback(usage, model_name)``
after every completed ``request`` / ``request_stream`` call. Exceptions raised
inside the callback are caught and logged so metering failures never crash the
pipeline.

Using a ``WrapperModel`` gives metering at the framework's own boundary:
every agent-driven LLM call — orchestrator turns, sub-agent runs, tool-call
follow-ups — is metered from one place, with no per-call-site plumbing.
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Callable
from contextlib import asynccontextmanager

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

logger = logging.getLogger(__name__)

UsageCallback = Callable[[RequestUsage, str], None]


class MeteredModel(WrapperModel):
    """Wraps a ``Model`` and fires ``callback(usage, model_name)`` per request."""

    def __init__(self, wrapped: Model, callback: UsageCallback) -> None:
        super().__init__(wrapped)
        self._callback = callback

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await self.wrapped.request(
            messages, model_settings, model_request_parameters
        )
        self._emit(response.usage)
        return response

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        async with self.wrapped.request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as stream:
            yield stream
            # Streamed usage is only final once iteration completes.
            self._emit(stream.usage())

    def _emit(self, usage: RequestUsage) -> None:
        try:
            self._callback(usage, self.model_name)
        except Exception:
            logger.exception("usage_callback failed")
