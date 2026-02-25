"""Claude Code integration — execute step, result type, and trace converter."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

from ..core.context import ACEStepContext
from ..implementations.prompts import wrap_skillbook_for_external_agent

logger = logging.getLogger(__name__)

CLAUDE_CODE_AVAILABLE = shutil.which("claude") is not None


# ---------------------------------------------------------------------------
# Input / Output types
# ---------------------------------------------------------------------------


@dataclass
class ClaudeCodeResult:
    """Output from a Claude Code CLI execution.

    This is the integration-specific result — not yet in ACE trace format.
    Use ``ClaudeCodeToTrace`` to convert to a standardised trace dict.
    """

    task: str
    success: bool
    output: str = ""
    execution_trace: str = ""
    returncode: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Execute step
# ---------------------------------------------------------------------------


class ClaudeCodeExecuteStep:
    """INJECT skillbook context and EXECUTE via Claude Code CLI.

    Reads a task string from ``ctx.sample``, writes a ``ClaudeCodeResult``
    to ``ctx.trace``.
    """

    requires = frozenset({"sample", "skillbook"})
    provides = frozenset({"trace"})

    def __init__(
        self,
        working_dir: Optional[str] = None,
        timeout: int = 600,
        model: Optional[str] = None,
        allowed_tools: Optional[list[str]] = None,
    ) -> None:
        if not CLAUDE_CODE_AVAILABLE:
            raise RuntimeError(
                "Claude Code CLI not found. Install from: https://claude.ai/code"
            )
        self.working_dir = Path(working_dir).resolve() if working_dir else Path.cwd()
        self.timeout = timeout
        self.model = model
        self.allowed_tools = allowed_tools

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        task: str = ctx.sample

        # -- INJECT --
        prompt = self._inject(task, ctx.skillbook)

        # -- EXECUTE --
        result = self._execute(task, prompt)

        return ctx.replace(trace=result)

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    @staticmethod
    def _inject(task: str, skillbook: Any) -> str:
        if skillbook is None:
            return task
        context = wrap_skillbook_for_external_agent(skillbook)
        if not context:
            return task
        return f"{task}\n\n{context}"

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute(self, task: str, prompt: str) -> ClaudeCodeResult:
        cmd: list[str] = [
            "claude",
            "--print",
            "--output-format=stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]
        if self.model:
            cmd.extend(["--model", self.model])
        if self.allowed_tools:
            for tool in self.allowed_tools:
                cmd.extend(["--allowedTools", tool])

        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                text=True,
                cwd=str(self.working_dir),
                capture_output=True,
                timeout=self.timeout,
                env=env,
            )
            execution_trace, summary = self._parse_stream_json(result.stdout)
            return ClaudeCodeResult(
                task=task,
                success=result.returncode == 0,
                output=summary,
                execution_trace=execution_trace,
                returncode=result.returncode,
                error=result.stderr[:500] if result.returncode != 0 else None,
            )
        except subprocess.TimeoutExpired:
            return ClaudeCodeResult(
                task=task,
                success=False,
                returncode=-1,
                error=f"Execution timed out after {self.timeout}s",
            )
        except Exception as exc:
            return ClaudeCodeResult(
                task=task,
                success=False,
                returncode=-1,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Stream-JSON parser
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_stream_json(stdout: str) -> Tuple[str, str]:
        """Parse stream-json output. Returns ``(execution_trace, summary)``."""
        trace_parts: list[str] = []
        final_text = ""
        step_num = 0

        for line in stdout.split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if event.get("type") != "assistant":
                continue

            for block in event.get("message", {}).get("content", []):
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text = block.get("text", "")
                    if text.strip():
                        trace_parts.append(f"[Reasoning] {text[:300]}")
                        final_text = text
                elif block_type == "tool_use":
                    step_num += 1
                    tool_name = block.get("name", "unknown")
                    tool_input = block.get("input", {})
                    if tool_name in ("Read", "Glob", "Grep"):
                        target = tool_input.get("file_path") or tool_input.get(
                            "pattern", ""
                        )
                        trace_parts.append(f"[Step {step_num}] {tool_name}: {target}")
                    elif tool_name in ("Write", "Edit"):
                        target = tool_input.get("file_path", "")
                        trace_parts.append(f"[Step {step_num}] {tool_name}: {target}")
                    elif tool_name == "Bash":
                        cmd = tool_input.get("command", "")[:80]
                        trace_parts.append(f"[Step {step_num}] Bash: {cmd}")
                    else:
                        trace_parts.append(f"[Step {step_num}] {tool_name}")

        execution_trace = (
            "\n".join(trace_parts) if trace_parts else "(No trace captured)"
        )

        if final_text:
            paragraphs = [p.strip() for p in final_text.split("\n\n") if p.strip()]
            summary = paragraphs[-1][:300] if paragraphs else final_text[:300]
        else:
            summary = f"Completed {step_num} steps"

        return execution_trace, summary


# ---------------------------------------------------------------------------
# Convert step — ClaudeCodeResult → standardised trace dict
# ---------------------------------------------------------------------------


class ClaudeCodeToTrace:
    """Convert a ``ClaudeCodeResult`` on ``ctx.trace`` to the standardised
    trace dict that the learning tail (``ReflectStep``) expects.

    Trace dict keys: ``question``, ``reasoning``, ``answer``, ``skill_ids``,
    ``feedback``, ``ground_truth``.
    """

    requires = frozenset({"trace"})
    provides = frozenset({"trace"})

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        r: ClaudeCodeResult = ctx.trace  # type: ignore[assignment]

        status = "succeeded" if r.success else "failed"
        feedback = f"Claude Code task {status}"
        if r.error:
            feedback += f"\nError: {r.error}"

        trace: dict = {
            "question": r.task,
            "reasoning": r.execution_trace,
            "answer": r.output,
            "skill_ids": [],
            "feedback": feedback,
            "ground_truth": None,
        }
        return ctx.replace(trace=trace)


__all__ = [
    "ClaudeCodeExecuteStep",
    "ClaudeCodeResult",
    "ClaudeCodeToTrace",
]
