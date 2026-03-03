"""ClaudeRRStep — Recursive Reflector using the Claude Agent SDK.

Alternative to :class:`~ace_next.rr.runner.RRStep` that delegates the
iterative analysis loop to Claude's native agentic capabilities instead
of the custom REPL sandbox.  Trace data is written to files in a temp
directory that Claude reads with its built-in tools.

Public API::

    from ace_next.rr import ClaudeRRStep, ClaudeRRConfig

    rr = ClaudeRRStep(ClaudeRRConfig(model="claude-sonnet-4-5-20250929"))
    pipe = Pipeline([..., rr, ...])
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

from ace_next.core.context import ACEStepContext
from ace_next.core.outputs import ExtractedLearning, ReflectorOutput

from .steps import _parse_direct_response, _parse_final_value

if TYPE_CHECKING:
    from ace_next.core.outputs import AgentOutput

logger = logging.getLogger(__name__)

try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_ALLOWED_TOOLS = ["Read", "Bash", "Write", "Glob", "Grep"]


@dataclass
class ClaudeRRConfig:
    """Configuration for the Claude-based Recursive Reflector."""

    model: str | None = None
    env: dict[str, str] | None = None
    max_turns: int = 10
    max_budget_usd: float = 0.50
    permission_mode: str = "bypassPermissions"
    allowed_tools: list[str] | None = None
    extra_instructions: str = ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a trace analyst. You analyze agent execution traces and extract learnings \
that become strategies for future agents.

Your learnings will be added to a **skillbook** — a set of strategies injected into \
future agents' prompts before they execute similar tasks. A downstream SkillManager \
will refine, split, and curate your learnings. Your job is to identify WHAT the agent \
did that mattered and WHY.

**Priority:** Accuracy > Efficiency > Cost. Extract only real insights with evidence.

## Working Directory

- trace.json — The execution trace to analyze (question, ground_truth, feedback, \
steps with role/reasoning/answer/skill_ids). May contain `"batch": true` with \
multiple tasks — see Batch Analysis below.
- skillbook.md — The strategies available to the agent during execution.
- workspace/ — Scratch space for analysis scripts.

## Analysis Phases

1. **Discover** — Read trace.json. Understand the task, what was asked, what the \
ground truth expected.
2. **Survey** — Scan all steps. Note tool calls, decisions, errors, and the final \
answer. Read skillbook.md to see what strategies were available.
3. **Categorize** — Was this a success or failure? For failure: find the exact step \
where the agent diverged. For success: identify non-obvious correct decisions.
4. **Deep-dive** — Use Grep/Read on specific sections. Write scripts in workspace/ \
and run via Bash for pattern analysis when traces are large.
5. **Synthesize** — Produce the output JSON with learnings that pass all quality rules.

## Success vs Failure

**On failure:** Find the specific step where the agent diverged from the correct path. \
What tool call, decision, or response caused the failure? What should it have done \
instead?

**On success:** Was there anything non-obvious the agent did that a different agent \
might not? If the success was straightforward, it is fine to extract zero learnings.

**Key question for every potential learning:** Would a future agent benefit from \
having this as an explicit strategy in its prompt? If no — do not extract it.

## What to Extract — 5 Categories (priority order)

### 1. Tool Workflow Sequences (highest value)
The correct order of tool/API calls. Future agents fail when they call tools in the \
wrong order.
- "Call get_user_details BEFORE get_reservation_details"
- "Execute upgrade BEFORE cancellation on same reservation"

### 2. Policy Rules with Specific Values
Concrete domain rules with actual numbers, thresholds, and conditions.
- "Basic economy + no insurance + outside 24h = non-cancellable"
- "Compensation: $100/passenger for cancellation, $50 for delay"

### 3. Anti-Patterns (what NOT to do)
Specific failure modes observed in the trace. Frame as "Do X instead of Y".
- "Escalate after 2 policy re-explanations instead of repeating indefinitely"
- "Offer alternatives before escalating"

### 4. Decision Logic
Conditional rules that determine which action path to take.
- "Already-flown flights: reject outright, no escalation needed"
- "Check eligibility before offering compensation"

### 5. Communication Patterns (only if non-obvious)
Only extract if tied to a specific failure or non-obvious success in the trace.
- "Rotate empathy statements — repeating 'I understand' 5x becomes tone-deaf"
- "Explain WHY policy exists, not just WHAT it says"

**DO NOT extract generic communication advice** like "be empathetic" or "acknowledge \
concerns" — every agent already knows this. Only extract communication learnings tied \
to a specific observed failure or non-obvious success.

## Section Names
Tag each learning with a domain-specific section name reflecting the workflow it \
belongs to. Use names like `cancellation_policy`, `flight_booking`, \
`payment_processing`, `escalation_workflow`, `tool_usage` — NOT generic names like \
`customer_service`.

## Learning Quality Rules

### REQUIRED for every learning:
1. **Domain-specific** — Must reference actual tools, values, patterns from the task
2. **Evidence** — MUST cite specific trace detail (step number, tool output, agent quote)
3. **Atomic** — Single concept only, no "and" combining multiple ideas
4. **Actionable** — "Use X for Y" / "Call X before Y" format, not "consider" or "think about"
5. **Under 15 words** — Concise and specific

### FORBIDDEN learnings (will make your analysis worthless):
- "Be systematic" / "Think carefully" / "Step-by-step reasoning" → Too vague
- "Verify results" / "Validate input" → Generic with no specificity
- "Consider X" / "Be aware of Y" → Not actionable commands
- "Acknowledge the customer's feelings" → Generic empathy
- "Use clear communication" / "Be professional" → Surface-level noise
- Empty evidence field → No learning without proof from the trace

### GOOD learning examples:
```json
{"learning": "Call get_reservation_details before any cancellation action", \
"atomicity_score": 0.95, "evidence": "Agent cancelled at step 5 without checking \
reservation status, causing wrong refund amount"}
{"learning": "Basic economy + no insurance + outside 24h = must escalate", \
"atomicity_score": 0.93, "evidence": "Agent spent 4 turns re-explaining policy \
instead of escalating per rules"}
{"learning": "Execute cabin upgrade BEFORE cancellation on same reservation", \
"atomicity_score": 0.92, "evidence": "Step 8: agent cancelled first, lost ability \
to apply upgrade workaround"}
```

### BAD learning examples (DO NOT produce these):
```json
{"learning": "Systematic reasoning is important", "atomicity_score": 0.7, \
"evidence": ""}
{"learning": "Always verify your work", "atomicity_score": 0.8, "evidence": ""}
{"learning": "Consider edge cases and validate input", "atomicity_score": 0.6, \
"evidence": ""}
```

## Batch Analysis

When trace.json contains `"batch": true` with multiple tasks, analyze ALL tasks:
- Look for **cross-task patterns** — errors that repeat across multiple tasks reveal \
systemic gaps (these are highest-value learnings).
- Note **divergent outcomes** — same policy/tool handled differently across tasks \
reveals inconsistency the agent should resolve.
- Cite specific task indices in evidence: "Tasks 3,7,12 all failed at reservation \
lookup step."

## Output Format

Your FINAL message must contain a JSON block:
```json
{
    "reasoning": "What happened and why — your analysis",
    "key_insight": "Single most transferable learning",
    "extracted_learnings": [
        {
            "learning": "...",
            "atomicity_score": 0.9,
            "evidence": "REQUIRED: specific detail from trace",
            "section": "domain_specific_section_name"
        }
    ],
    "skill_tags": [
        {"id": "actual-skill-id-from-skillbook", "tag": "helpful|harmful|neutral"}
    ]
}
```

Optional fields (include for failures, skip for successes):
- `"error_identification"` — What went wrong
- `"root_cause_analysis"` — Why it went wrong
- `"correct_approach"` — What the correct approach should be

If skillbook is empty, return an empty `skill_tags` list. Never invent skill IDs.
Every learning MUST have a non-empty `evidence` field citing specific trace details."""

_INITIAL_PROMPT = """\
Analyze the execution trace in trace.json and the skillbook in skillbook.md.

Start by reading trace.json to understand the task and outcome, then read \
skillbook.md to see what strategies were available. Investigate the trace \
thoroughly before producing your final JSON output.

Remember: extract only domain-specific, evidence-backed, actionable learnings. \
If the task succeeded straightforwardly, zero learnings is acceptable."""


# ---------------------------------------------------------------------------
# ClaudeRRStep
# ---------------------------------------------------------------------------


class ClaudeRRStep:
    """Recursive Reflector using the Claude Agent SDK.

    Satisfies **StepProtocol** — can be placed directly in a Pipeline.
    Also satisfies **ReflectorLike** — can be passed to ``ReflectStep``
    or used standalone via ``reflect()``.

    Trace data is written to a temp directory as files that Claude reads
    with its built-in tools (Read, Bash, Grep, etc.).
    """

    requires = frozenset({"trace", "skillbook"})
    provides = frozenset({"reflection"})

    def __init__(self, config: ClaudeRRConfig | None = None) -> None:
        if not CLAUDE_SDK_AVAILABLE:
            raise ImportError(
                "claude-agent-sdk is required for ClaudeRRStep. "
                "Install it with: pip install 'ace-framework[claude-agent-sdk]'"
            )
        self.config = config or ClaudeRRConfig()

    # ------------------------------------------------------------------
    # StepProtocol entry
    # ------------------------------------------------------------------

    async def __call__(self, ctx: ACEStepContext) -> ACEStepContext:  # type: ignore[override]
        """Run the Claude-based Recursive Reflector and attach reflection to *ctx*."""
        traces = self._build_traces_dict_from_ctx(ctx)
        skillbook_text = self._extract_skillbook_text(ctx.skillbook)
        reflection = await self._run(traces, skillbook_text)
        return ctx.replace(reflection=reflection)

    # ------------------------------------------------------------------
    # ReflectorLike entry (also usable standalone)
    # ------------------------------------------------------------------

    def reflect(
        self,
        *,
        question: str = "",
        agent_output: Optional[AgentOutput] = None,
        skillbook: Any = None,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        """Synchronous reflect — matches ReflectorLike protocol.

        Uses ``asyncio.run()`` internally.  Safe to call from sync code
        (e.g. ``ReflectStep`` which runs behind ``async_boundary``).
        """
        trace = kwargs.pop("trace", None)
        if trace is not None and isinstance(trace, dict):
            traces = trace
        else:
            traces = self._build_traces_dict(
                question, agent_output, ground_truth, feedback
            )
        skillbook_text = self._extract_skillbook_text(skillbook)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self._run(traces, skillbook_text))
                return future.result()
        return asyncio.run(self._run(traces, skillbook_text))

    # ------------------------------------------------------------------
    # Core async logic
    # ------------------------------------------------------------------

    async def _run(
        self, traces: dict[str, Any], skillbook_text: str
    ) -> ReflectorOutput:
        """Create temp dir, call SDK query(), parse result."""
        tmp_dir = tempfile.mkdtemp(prefix="claude-rr-")
        try:
            self._write_sandbox_files(tmp_dir, traces, skillbook_text)
            final_text, cost_info = await self._query_sdk(tmp_dir)
            return self._parse_response(final_text, cost_info)
        except Exception as e:
            logger.error("Claude RR failed: %s", e, exc_info=True)
            return self._build_error_output(str(e))
        finally:
            self._cleanup(tmp_dir)

    def _write_sandbox_files(
        self,
        tmp_dir: str,
        traces: dict[str, Any],
        skillbook_text: str,
    ) -> None:
        """Write trace.json, skillbook.md, and create workspace/."""
        with open(os.path.join(tmp_dir, "trace.json"), "w") as f:
            json.dump(traces, f, indent=2, default=str)

        with open(os.path.join(tmp_dir, "skillbook.md"), "w") as f:
            f.write(skillbook_text)

        os.makedirs(os.path.join(tmp_dir, "workspace"), exist_ok=True)

    async def _query_sdk(self, tmp_dir: str) -> tuple[str, dict[str, Any]]:
        """Call the Claude Agent SDK and collect the final text + cost info."""
        cfg = self.config
        system_prompt = _SYSTEM_PROMPT
        if cfg.extra_instructions:
            system_prompt += "\n\n" + cfg.extra_instructions

        options = ClaudeAgentOptions(
            model=cfg.model,
            env=cfg.env,
            max_turns=cfg.max_turns,
            max_budget_usd=cfg.max_budget_usd,
            system_prompt=system_prompt,
            permission_mode=cfg.permission_mode,
            allowed_tools=cfg.allowed_tools or _DEFAULT_ALLOWED_TOOLS,
            cwd=tmp_dir,
        )

        text_parts: list[str] = []
        cost_info: dict[str, Any] = {}
        async for message in query(prompt=_INITIAL_PROMPT, options=options):
            if isinstance(message, AssistantMessage):
                # Collect text from the last assistant message
                text_parts = [
                    block.text
                    for block in message.content
                    if isinstance(block, TextBlock)
                ]
            elif isinstance(message, ResultMessage):
                cost_info = {
                    "total_cost_usd": message.total_cost_usd,
                    "num_turns": message.num_turns,
                    "duration_ms": message.duration_ms,
                    "is_error": message.is_error,
                    "session_id": message.session_id,
                }
        return "\n".join(text_parts), cost_info

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_response(
        self, final_text: str, cost_info: dict[str, Any]
    ) -> ReflectorOutput:
        """Parse the SDK's final text into a ReflectorOutput."""
        if not final_text:
            return self._build_error_output("Empty response from Claude SDK")

        # Path 1: text is pure JSON (optionally fenced)
        try:
            result = _parse_direct_response(final_text)
            result.raw["claude_rr"] = True
            result.raw["cost_info"] = cost_info
            return result
        except Exception:
            pass

        # Path 2: JSON embedded in prose
        json_str = self._extract_json_block(final_text)
        if json_str:
            try:
                data = json.loads(json_str)
                result = _parse_final_value(data)
                result.raw["claude_rr"] = True
                result.raw["cost_info"] = cost_info
                return result
            except Exception:
                pass

        return self._build_error_output(
            f"Could not parse Claude response: {final_text[:200]}",
            raw_text=final_text,
            cost_info=cost_info,
        )

    @staticmethod
    def _extract_json_block(text: str) -> str | None:
        """Extract a JSON block from markdown-fenced text."""
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # Try finding raw JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return None

    # ------------------------------------------------------------------
    # Trace building
    # ------------------------------------------------------------------

    def _build_traces_dict_from_ctx(self, ctx: ACEStepContext) -> dict[str, Any]:
        """Build traces dict from ACEStepContext.trace (may already be a dict)."""
        trace = ctx.trace
        if isinstance(trace, dict):
            return trace
        # Fallback: build minimal dict from context
        return {
            "question": getattr(trace, "question", ""),
            "ground_truth": getattr(trace, "ground_truth", None),
            "feedback": getattr(trace, "feedback", None),
            "steps": getattr(trace, "steps", []),
        }

    @staticmethod
    def _build_traces_dict(
        question: str,
        agent_output: Optional[AgentOutput],
        ground_truth: Optional[str],
        feedback: Optional[str],
    ) -> dict[str, Any]:
        """Build canonical traces dict from individual parameters."""
        ao = agent_output
        return {
            "question": question,
            "ground_truth": ground_truth,
            "feedback": feedback,
            "steps": [
                {
                    "role": "agent",
                    "reasoning": ao.reasoning if ao else "",
                    "answer": ao.final_answer if ao else "",
                    "skill_ids": ao.skill_ids if ao else [],
                }
            ],
        }

    @staticmethod
    def _extract_skillbook_text(skillbook: Any) -> str:
        """Extract text from a skillbook (SkillbookView, Skillbook, or str)."""
        if skillbook is None:
            return "(empty skillbook)"
        if hasattr(skillbook, "as_prompt"):
            return skillbook.as_prompt() or "(empty skillbook)"
        return str(skillbook)

    # ------------------------------------------------------------------
    # Error / cleanup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_error_output(
        error_msg: str,
        raw_text: str = "",
        cost_info: dict[str, Any] | None = None,
    ) -> ReflectorOutput:
        """Build a fallback ReflectorOutput on failure."""
        return ReflectorOutput(
            reasoning=f"Claude RR analysis failed: {error_msg}",
            error_identification="claude_rr_error",
            root_cause_analysis=error_msg,
            correct_approach="Retry with different configuration or fall back to RRStep",
            key_insight="Claude-based analysis was unable to complete",
            extracted_learnings=[
                ExtractedLearning(
                    learning="Claude RR encountered an error during analysis",
                    atomicity_score=0.3,
                    evidence=error_msg,
                )
            ],
            skill_tags=[],
            raw={
                "claude_rr": True,
                "error": error_msg,
                "raw_text": raw_text,
                "cost_info": cost_info or {},
            },
        )

    @staticmethod
    def _cleanup(tmp_dir: str) -> None:
        """Remove the temporary sandbox directory."""
        try:
            shutil.rmtree(tmp_dir)
        except OSError as e:
            logger.warning("Failed to clean up temp dir %s: %s", tmp_dir, e)
