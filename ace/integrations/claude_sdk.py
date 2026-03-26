"""Claude SDK integration — execute step, result type, and trace converter.

Uses the ``anthropic`` Python SDK directly for full API access with
built-in observability (token tracking, latency, Logfire spans and
auto-instrumentation).

Usage::

    from ace.integrations import ClaudeSDKExecuteStep, ClaudeSDKToTrace
    from ace.steps import learning_tail

    steps = [
        ClaudeSDKExecuteStep(model="claude-sonnet-4-20250514"),
        ClaudeSDKToTrace(),
        *learning_tail(reflector, skill_manager, skillbook),
    ]
    pipeline = Pipeline(steps)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.context import ACEStepContext
from ..implementations.prompts import wrap_skillbook_for_external_agent

logger = logging.getLogger(__name__)


def _get_logfire() -> Any:
    """Return the ``logfire`` module if configured, else ``None``."""
    try:
        from ace.observability import is_configured

        if is_configured():
            import logfire

            return logfire
    except Exception:
        pass
    return None


@contextmanager
def _logfire_span(name: str, **attributes: Any) -> Iterator[Any]:
    """Open a Logfire span if configured, otherwise yield a no-op object.

    Attributes can be set on the yielded object via ``set_attribute``.
    When Logfire is not active the context manager yields a lightweight
    stub so callers don't need conditional logic.
    """
    lf = _get_logfire()
    if lf is not None:
        with lf.span(name, **attributes) as span:
            yield span
    else:
        yield _NoOpSpan()


class _NoOpSpan:
    """Stub returned when Logfire is not configured."""

    def set_attribute(self, key: str, value: Any) -> None:  # noqa: ARG002
        pass

    def record_exception(self, exc: BaseException) -> None:  # noqa: ARG002
        pass


try:
    import anthropic

    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False
    anthropic = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Input / Output types
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A single tool call from the Claude API response."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Tool call ID (e.g. toolu_01...)")
    name: str = Field(..., description="Tool name")
    input: Dict[str, Any] = Field(
        default_factory=dict, description="Tool input arguments"
    )


class _ClaudeSDKConfig(BaseModel):
    """Validated configuration for :class:`ClaudeSDKExecuteStep`."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(default="claude-sonnet-4-20250514", min_length=1)
    system_prompt: Optional[str] = None
    max_tokens: int = Field(default=4096, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    tools: Optional[List[Dict[str, Any]]] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    inject_skillbook: bool = True


class ClaudeSDKResult(BaseModel):
    """Output from a direct Anthropic SDK call.

    This is the integration-specific result — not yet in ACE trace format.
    Use ``ClaudeSDKToTrace`` to convert to a standardised trace dict.

    Includes validated observability data: token usage, latency, model info.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    task: str = Field(..., description="Original task/question sent to the API")
    success: bool = Field(..., description="Whether the API call succeeded")
    output: str = Field(default="", description="Text output from the model")
    error: Optional[str] = Field(
        default=None, description="Error message if the call failed"
    )
    model: str = Field(default="", description="Model ID used for the request")
    stop_reason: Optional[str] = Field(
        default=None,
        description="Why the model stopped: end_turn, max_tokens, tool_use, etc.",
    )
    # Observability — token usage
    input_tokens: int = Field(default=0, ge=0, description="Prompt tokens consumed")
    output_tokens: int = Field(
        default=0, ge=0, description="Completion tokens generated"
    )
    total_tokens: int = Field(default=0, ge=0, description="Total tokens (in + out)")
    # Observability — latency
    latency_seconds: float = Field(
        default=0.0, ge=0.0, description="Wall-clock time for the API call"
    )
    # Tool use tracking
    tool_calls: List[ToolCall] = Field(
        default_factory=list, description="Tool calls made by the model"
    )
    cited_skill_ids: List[str] = Field(
        default_factory=list,
        description="Skill IDs cited in the output ([section-00001] patterns)",
    )
    # Raw response for full access
    raw_response: Any = Field(
        default=None,
        exclude=True,
        description="Raw Anthropic API response object",
    )

    @model_validator(mode="after")
    def _compute_total(self) -> "ClaudeSDKResult":
        """Auto-compute total_tokens from input + output if left at default."""
        if self.total_tokens == 0 and (self.input_tokens or self.output_tokens):
            self.total_tokens = self.input_tokens + self.output_tokens
        return self


# ---------------------------------------------------------------------------
# Execute step
# ---------------------------------------------------------------------------


class ClaudeSDKExecuteStep:
    """INJECT skillbook context and EXECUTE via the Anthropic Python SDK.

    Reads a task/question from ``ctx.sample``, calls the Claude Messages
    API directly, and writes a ``ClaudeSDKResult`` to ``ctx.trace``.

    Observability is built in:

    - **Logfire spans** with structured attributes (model, tokens,
      latency, stop_reason, tool_count) when Logfire is configured
    - **Logfire auto-instrumentation** of the underlying Anthropic
      client (child spans for each API call)
    - **Token usage** (input/output/total) from the API response
    - **Latency** (wall-clock time for the API call)
    - **Structured logging** of per-call metrics (always active)

    Compose with the learning tail::

        steps = [
            ClaudeSDKExecuteStep(model="claude-sonnet-4-20250514"),
            ClaudeSDKToTrace(),
            *learning_tail(reflector, skill_manager, skillbook),
        ]
    """

    requires = frozenset({"sample", "skillbook"})
    provides = frozenset({"trace"})

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        *,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        tools: Optional[List[Dict[str, Any]]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        inject_skillbook: bool = True,
        client: Any = None,
        **client_kwargs: Any,
    ) -> None:
        config = _ClaudeSDKConfig(
            model=model,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            api_key=api_key,
            base_url=base_url,
            inject_skillbook=inject_skillbook,
        )
        self.model = config.model
        self.system_prompt = config.system_prompt
        self.max_tokens = config.max_tokens
        self.temperature = config.temperature
        self.tools = config.tools
        self.inject_skillbook = config.inject_skillbook

        if client is not None:
            self._client = client
        elif ANTHROPIC_SDK_AVAILABLE:
            ckw: Dict[str, Any] = {**client_kwargs}
            if config.api_key is not None:
                ckw["api_key"] = config.api_key
            if config.base_url is not None:
                ckw["base_url"] = config.base_url
            self._client = anthropic.Anthropic(**ckw)
        else:
            raise ImportError(
                "anthropic SDK not installed. Install with: uv add "
                '"ace-framework[claude-sdk]" or uv add anthropic'
            )

        self._try_instrument()

    # ------------------------------------------------------------------
    # Logfire auto-instrumentation
    # ------------------------------------------------------------------

    def _try_instrument(self) -> None:
        """Auto-instrument the Anthropic client with Logfire if configured.

        Logfire instruments the client eagerly and returns an optional context
        manager for later uninstrumentation. The instrumentation call itself is
        sufficient here.
        """
        try:
            from ace.observability import is_configured

            if is_configured():
                import logfire

                logfire.instrument_anthropic(self._client)
                logger.debug("ClaudeSDKExecuteStep: Logfire instrumentation active")
        except Exception as exc:
            logger.debug("Logfire instrumentation skipped: %s", exc)

    # ------------------------------------------------------------------
    # Step execution
    # ------------------------------------------------------------------

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        task = self._extract_task(ctx.sample)
        system = self._build_system(ctx.skillbook)
        messages: List[Dict[str, Any]] = [{"role": "user", "content": task}]

        with _logfire_span(
            "ClaudeSDKExecuteStep",
            model=self.model,
            task=task[:200],
            has_system=system is not None,
            has_tools=bool(self.tools),
        ) as span:
            result = self._execute(task, system, messages)
            span.set_attribute("success", result.success)
            span.set_attribute("input_tokens", result.input_tokens)
            span.set_attribute("output_tokens", result.output_tokens)
            span.set_attribute("total_tokens", result.total_tokens)
            span.set_attribute("latency_seconds", result.latency_seconds)
            span.set_attribute("stop_reason", result.stop_reason or "")
            span.set_attribute("tool_call_count", len(result.tool_calls))
            span.set_attribute("cited_skill_count", len(result.cited_skill_ids))
            if result.error:
                span.set_attribute("error", result.error)

        return ctx.replace(trace=result)

    # ------------------------------------------------------------------
    # Task extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_task(sample: Any) -> str:
        """Extract task string from sample (string or ACESample)."""
        if isinstance(sample, str):
            return sample
        if hasattr(sample, "question"):
            parts = [sample.question]
            if hasattr(sample, "context") and sample.context:
                parts.append(f"\nContext: {sample.context}")
            return "\n".join(parts)
        return str(sample)

    # ------------------------------------------------------------------
    # System prompt with skillbook injection
    # ------------------------------------------------------------------

    def _build_system(self, skillbook: Any) -> Optional[str]:
        """Build the system prompt, optionally injecting skillbook context."""
        parts: List[str] = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        if self.inject_skillbook and skillbook is not None:
            context = wrap_skillbook_for_external_agent(skillbook)
            if context:
                parts.append(context)
        return "\n\n".join(parts) if parts else None

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    def _execute(
        self,
        task: str,
        system: Optional[str],
        messages: List[Dict[str, Any]],
    ) -> ClaudeSDKResult:
        """Call the Anthropic Messages API and build a result with metrics."""
        api_kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
        }
        if system is not None:
            api_kwargs["system"] = system
        if self.tools:
            api_kwargs["tools"] = self.tools

        start = time.monotonic()
        try:
            response = self._client.messages.create(**api_kwargs)
            latency = time.monotonic() - start

            text_parts: List[str] = []
            tool_calls: List[ToolCall] = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=block.id,
                            name=block.name,
                            input=block.input or {},
                        )
                    )

            output = "\n".join(text_parts)
            cited_ids = self._extract_skill_ids(output)

            result = ClaudeSDKResult(
                task=task,
                success=True,
                output=output,
                model=response.model,
                stop_reason=response.stop_reason,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=(
                    response.usage.input_tokens + response.usage.output_tokens
                ),
                latency_seconds=round(latency, 3),
                tool_calls=tool_calls,
                cited_skill_ids=cited_ids,
                raw_response=response,
            )
            self._log_metrics(result)
            return result

        except Exception as exc:
            latency = time.monotonic() - start
            logger.error(
                "ClaudeSDKExecuteStep failed after %.2fs: %s",
                latency,
                exc,
            )
            lf = _get_logfire()
            if lf is not None:
                lf.error(
                    "ClaudeSDK call failed",
                    error=str(exc),
                    model=self.model,
                    latency_seconds=round(latency, 3),
                )
            return ClaudeSDKResult(
                task=task,
                success=False,
                error=str(exc),
                model=self.model,
                latency_seconds=round(latency, 3),
            )

    # ------------------------------------------------------------------
    # Observability helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _log_metrics(result: ClaudeSDKResult) -> None:
        """Log structured observability metrics via logging and Logfire."""
        logger.info(
            "ClaudeSDK: model=%s tokens=%d/%d/%d latency=%.2fs stop=%s tools=%d",
            result.model,
            result.input_tokens,
            result.output_tokens,
            result.total_tokens,
            result.latency_seconds,
            result.stop_reason,
            len(result.tool_calls),
        )
        lf = _get_logfire()
        if lf is not None:
            lf.info(
                "ClaudeSDK call completed",
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                total_tokens=result.total_tokens,
                latency_seconds=result.latency_seconds,
                stop_reason=result.stop_reason,
                tool_call_count=len(result.tool_calls),
            )

    @staticmethod
    def _extract_skill_ids(text: str) -> List[str]:
        """Extract cited skill IDs from output text."""
        try:
            from ..implementations.helpers import extract_cited_skill_ids

            return extract_cited_skill_ids(text)
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Convert step — ClaudeSDKResult → standardised trace dict
# ---------------------------------------------------------------------------


class ClaudeSDKToTrace:
    """Convert a ``ClaudeSDKResult`` on ``ctx.trace`` to the standardised
    trace dict that the learning tail (``ReflectStep``) expects.

    Includes observability metadata (tokens, latency) in reasoning and
    feedback for the reflector to consider.
    """

    requires = frozenset({"trace"})
    provides = frozenset({"trace"})

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        r: ClaudeSDKResult = ctx.trace  # type: ignore[assignment]

        parts: List[str] = []
        status = "succeeded" if r.success else "failed"
        parts.append(f"Claude SDK call {status} ({r.model})")
        parts.append(
            f"Tokens: {r.input_tokens} in / {r.output_tokens} out / "
            f"{r.total_tokens} total"
        )
        parts.append(f"Latency: {r.latency_seconds}s")
        if r.stop_reason:
            parts.append(f"Stop reason: {r.stop_reason}")
        if r.tool_calls:
            parts.append(f"\nTool calls ({len(r.tool_calls)}):")
            for tc in r.tool_calls:
                parts.append(f"  - {tc.name}({tc.input})")
        if r.output:
            parts.append(f"\nOutput:\n{r.output}")
        if r.error:
            parts.append(f"\nError: {r.error}")
        reasoning = "\n".join(parts)

        feedback = f"Claude SDK call {status}"
        if r.error:
            feedback += f"\nError: {r.error}"
        feedback += (
            f"\nTokens: {r.input_tokens}+{r.output_tokens}={r.total_tokens}"
            f" | Latency: {r.latency_seconds}s"
        )

        trace: dict = {
            "question": r.task,
            "reasoning": reasoning,
            "answer": r.output,
            "skill_ids": r.cited_skill_ids,
            "feedback": feedback,
            "ground_truth": None,
        }
        return ctx.replace(trace=trace)


__all__ = [
    "ClaudeSDKExecuteStep",
    "ClaudeSDKResult",
    "ClaudeSDKToTrace",
    "ToolCall",
]
