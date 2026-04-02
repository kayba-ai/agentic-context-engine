"""RRStep — Recursive Reflector powered by PydanticAI.

Uses a PydanticAI agent with ``execute_code``, ``analyze``, and
``batch_analyze`` tools.  The agent explores traces via code execution
and sub-agent analysis, then produces ``ReflectorOutput`` as structured
output.

Satisfies both ``StepProtocol`` (for Pipeline composition) and
``ReflectorLike`` (drop-in replacement for simple Reflector).
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any, Optional

from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import UsageLimits

from ace.core.context import ACEStepContext
from ace.core.outputs import AgentOutput, ExtractedLearning, ReflectorOutput

from .agent import RRDeps, create_rr_agent, create_sub_agent
from .config import RecursiveConfig
from .prompts import REFLECTOR_RECURSIVE_PROMPT, REFLECTOR_RECURSIVE_SYSTEM
from .sandbox import TraceSandbox
from .trace_context import TraceContext

logger = logging.getLogger(__name__)


def _preview(text: str | None, max_len: int = 150) -> str:
    """Return a short preview safe for str.format()."""
    if not text:
        return "(empty)"
    snippet = text if len(text) <= max_len else text[:max_len]
    return snippet.replace("{", "{{").replace("}", "}}")


class RRStep:
    """Recursive Reflector as a pipeline step (PydanticAI agent).

    Satisfies **StepProtocol** (place directly in a Pipeline) and
    **ReflectorLike** (use as drop-in reflector in runners).

    Internally uses a PydanticAI agent with tools for code execution
    and sub-agent analysis.  The agent produces ``ReflectorOutput``
    as structured output when it has gathered enough evidence.

    Args:
        model: LiteLLM or PydanticAI model string.
        config: RR configuration (timeouts, limits, sub-agent settings).
        prompt_template: User prompt template with format placeholders.
        model_settings: Override PydanticAI model settings.
    """

    # StepProtocol
    requires = frozenset({"trace", "skillbook"})
    provides = frozenset({"reflections"})

    def __init__(
        self,
        model: str,
        config: Optional[RecursiveConfig] = None,
        prompt_template: str = REFLECTOR_RECURSIVE_PROMPT,
        model_settings: ModelSettings | None = None,
    ) -> None:
        self.config = config or RecursiveConfig()
        self.prompt_template = prompt_template
        self._model = model

        # Build PydanticAI agents
        self._agent = create_rr_agent(
            model,
            system_prompt=REFLECTOR_RECURSIVE_SYSTEM,
            config=self.config,
            model_settings=model_settings,
        )

        # Sub-agent for analyze/batch_analyze tools
        subagent_model = self.config.subagent_model or model
        self._sub_agent = (
            create_sub_agent(subagent_model, config=self.config)
            if self.config.enable_subagent
            else None
        )

    # ------------------------------------------------------------------
    # StepProtocol entry
    # ------------------------------------------------------------------

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        """Run the Recursive Reflector and attach the reflection(s).

        When the trace is a batch container (a raw list, ``"items"``,
        ``"tasks"``, or a legacy combined ``steps`` batch), a single
        session analyzes all items. Per-item results are parsed from
        the structured output.
        """
        trace = ctx.trace or {}
        if self._get_batch_items(trace) is not None:
            reflections = self._run_batch_reflections(trace, ctx.skillbook)
            return ctx.replace(reflections=reflections)
        elif isinstance(trace, dict):
            reflection = self._run_reflection(
                traces=trace,
                question=trace.get("question", ""),
                ground_truth=trace.get("ground_truth"),
                feedback=trace.get("feedback"),
                skillbook=ctx.skillbook,
            )
        else:
            reflection = self._run_reflection(
                skillbook=ctx.skillbook,
                trace=trace,
            )
        return ctx.replace(reflections=(reflection,))

    # ------------------------------------------------------------------
    # ReflectorLike protocol
    # ------------------------------------------------------------------

    def reflect(
        self,
        *,
        question: str,
        agent_output: AgentOutput,
        skillbook: Any = None,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        """ReflectorLike — delegates to the PydanticAI agent.

        Allows RRStep to be used as a drop-in replacement for Reflector
        in any runner or learning_tail pipeline.
        """
        return self._run_reflection(
            question=question,
            agent_output=agent_output,
            skillbook=skillbook,
            ground_truth=ground_truth,
            feedback=feedback,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Core reflection logic
    # ------------------------------------------------------------------

    def _run_reflection(
        self,
        *,
        question: str = "",
        agent_output: Optional[AgentOutput] = None,
        skillbook: Any = None,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        """Run the PydanticAI agent and return analysis."""
        trace_obj = kwargs.pop("trace", None)
        if trace_obj is None and agent_output is not None:
            trace_obj = getattr(agent_output, "trace_context", None)
            if trace_obj is None:
                trace_obj = TraceContext.from_agent_output(agent_output)  # type: ignore[arg-type]

        # Build traces dict — canonical data structure for sandbox code
        traces = kwargs.pop("traces", None)
        if traces is None:
            traces = self._build_traces_dict(
                question, agent_output, ground_truth, feedback, trace_obj
            )

        # Build sandbox with trace data (no ask_llm/FINAL injection)
        sandbox = self._create_sandbox(trace_obj, traces, skillbook)

        # Resolve skillbook text
        skillbook_text = ""
        if skillbook is not None:
            if hasattr(skillbook, "as_prompt"):
                skillbook_text = skillbook.as_prompt() or "(empty skillbook)"
            else:
                skillbook_text = str(skillbook)

        # Build deps for PydanticAI agent
        deps = RRDeps(
            sandbox=sandbox,
            trace_data=traces,
            skillbook_text=skillbook_text or "(empty skillbook)",
            config=self.config,
            sub_agent=self._sub_agent,
        )

        # Build user prompt
        initial_prompt = self._build_initial_prompt(traces, skillbook, trace_obj)

        # Run the PydanticAI agent with usage limits
        usage_limits = UsageLimits(request_limit=self.config.max_llm_calls)

        try:
            result = self._agent.run_sync(
                initial_prompt,
                deps=deps,
                usage_limits=usage_limits,
            )
            output = result.output

            # Merge execution metadata into raw (preserve LLM-populated fields)
            usage = result.usage()
            output.raw = {
                **output.raw,
                "max_llm_calls_scope": "main_agent_session",
                "usage": {
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "total_tokens": usage.total_tokens,
                    "requests": usage.requests,
                },
                "rr_trace": {
                    "total_iterations": deps.iteration,
                    "subagent_calls": deps.sub_agent_history,
                    "timed_out": False,
                },
            }

            logger.info(
                "RR completed: %d tool calls, %d sub-agent calls",
                deps.iteration,
                len(deps.sub_agent_history),
            )

            return output

        except UsageLimitExceeded:
            logger.warning(
                "RR usage limit reached (%d requests)",
                self.config.max_llm_calls,
            )
            return self._build_timeout_output(
                question, agent_output, ground_truth, feedback, deps
            )
        except Exception as e:
            logger.error("RR agent failed: %s", e, exc_info=True)
            return ReflectorOutput(
                reasoning=f"Recursive analysis failed: {e}",
                correct_approach="",
                key_insight="",
                raw={"error": str(e)},
            )

    # ------------------------------------------------------------------
    # Batch reflection
    # ------------------------------------------------------------------

    def _run_batch_reflections(
        self,
        batch_trace: Any,
        skillbook: Any,
    ) -> tuple[ReflectorOutput, ...]:
        """Run a single session for all items in a batch."""
        items = self._get_batch_items(batch_trace) or []

        reflection = self._run_reflection(
            traces=batch_trace,
            skillbook=skillbook,
        )

        return self._split_batch_reflection(reflection, items)

    def _split_batch_reflection(
        self,
        reflection: ReflectorOutput,
        items: list[Any],
    ) -> tuple[ReflectorOutput, ...]:
        """Extract per-item ReflectorOutputs from batch output."""
        item_results = reflection.raw.get("items")
        if not item_results:
            item_results = reflection.raw.get("tasks", [])

        if not item_results or len(item_results) != len(items):
            logger.warning(
                "Batch missing per-item results (got %d, expected %d); "
                "duplicating single reflection",
                len(item_results) if item_results else 0,
                len(items),
            )
            fallback = []
            for i, item in enumerate(items):
                r = reflection.model_copy(deep=True)
                r.raw["item_id"] = self._get_batch_item_id(item, i)
                fallback.append(r)
            return tuple(fallback)

        reflections: list[ReflectorOutput] = []
        rr_trace = reflection.raw.get("rr_trace", {})
        for i, (item, tr) in enumerate(zip(items, item_results)):
            item_id = self._get_batch_item_id(item, i)
            if not isinstance(tr, dict):
                tr = {}
            learnings = tr.get("extracted_learnings", tr.get("learnings", []))
            reflections.append(
                ReflectorOutput(
                    reasoning=tr.get("reasoning", reflection.reasoning),
                    error_identification=str(tr.get("error_identification", "")),
                    root_cause_analysis=tr.get("root_cause_analysis", ""),
                    correct_approach=tr.get(
                        "correct_approach", reflection.correct_approach
                    ),
                    key_insight=tr.get("key_insight", reflection.key_insight),
                    extracted_learnings=[
                        ExtractedLearning(
                            learning=l.get("learning", ""),
                            atomicity_score=float(l.get("atomicity_score", 0.0)),
                            evidence=l.get("evidence", ""),
                        )
                        for l in learnings
                        if isinstance(l, dict)
                    ],
                    raw={
                        **tr,
                        "item_id": item_id,
                        "rr_trace": rr_trace,
                    },
                )
            )
        return tuple(reflections)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _build_traces_dict(
        self,
        question: str,
        agent_output: Optional[AgentOutput],
        ground_truth: Optional[str],
        feedback: Optional[str],
        trace_obj: Any,
    ) -> dict[str, Any]:
        """Build the canonical ``traces`` dict from individual parameters."""
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

    def _create_sandbox(
        self,
        trace_obj: Any,
        traces: Any,
        skillbook: Any,
    ) -> TraceSandbox:
        """Create a simplified sandbox for code execution.

        Unlike the old RRStep, does NOT inject ask_llm, FINAL, or
        parallel_map — those are now PydanticAI tools.
        """
        sandbox = TraceSandbox(trace=trace_obj, llm_query_fn=None)

        # Skillbook text
        skillbook_text = ""
        if skillbook is not None:
            if hasattr(skillbook, "as_prompt"):
                skillbook_text = skillbook.as_prompt() or "(empty skillbook)"
            else:
                skillbook_text = str(skillbook)
        sandbox.inject("skillbook", skillbook_text or "(empty skillbook)")

        # Traces data
        sandbox.inject("traces", traces)
        self._inject_helper_variables(sandbox, traces)

        return sandbox

    def _looks_like_combined_steps_batch(self, traces: dict[str, Any]) -> bool:
        """Return True for legacy combined batches encoded inside ``steps``."""
        steps = traces.get("steps")
        if not isinstance(steps, list) or not steps:
            return False

        return all(
            isinstance(step, dict)
            and step.get("role") == "conversation"
            and isinstance(step.get("content"), dict)
            for step in steps
        )

    def _get_batch_items(self, traces: Any) -> list[Any] | None:
        """Return batch items from a raw list or common batch container shapes."""
        if isinstance(traces, list):
            return traces
        if not isinstance(traces, dict):
            return None
        for key in ("items", "tasks"):
            items = traces.get(key)
            if isinstance(items, list):
                return items
        if self._looks_like_combined_steps_batch(traces):
            return traces.get("steps")
        return None

    def _extract_batch_item_payload(self, item: Any) -> Any:
        """Return the analysis payload for a batch item without rewriting it."""
        if (
            isinstance(item, dict)
            and item.get("role") == "conversation"
            and isinstance(item.get("content"), dict)
        ):
            return item["content"]
        return item

    def _get_batch_item_id(self, item: Any, index: int) -> str:
        """Choose a stable identifier for a batch item."""
        payload = self._extract_batch_item_payload(item)
        if isinstance(item, dict):
            for key in ("item_id", "task_id", "id"):
                value = item.get(key)
                if value is not None:
                    return str(value)
        if isinstance(payload, dict):
            for key in ("item_id", "task_id", "id"):
                value = payload.get(key)
                if value is not None:
                    return str(value)
        return f"item_{index}"

    def _extract_batch_messages(self, item: Any) -> list[Any]:
        """Return a best-effort message/step list for previews and summaries."""
        payload = self._extract_batch_item_payload(item)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            trace_value = payload.get("trace")
            if isinstance(trace_value, list):
                return trace_value
            if isinstance(trace_value, dict):
                for key in ("messages", "steps", "trace"):
                    nested = trace_value.get(key)
                    if isinstance(nested, list):
                        return nested
            for key in ("steps", "messages"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
        return []

    def _extract_batch_field(self, item: Any, field: str) -> str:
        """Extract a string field from a batch item payload when present."""
        payload = self._extract_batch_item_payload(item)
        if isinstance(payload, dict):
            value = payload.get(field)
            if value is not None:
                return str(value)
        return ""

    def _inject_helper_variables(
        self,
        sandbox: TraceSandbox,
        traces: Any,
    ) -> None:
        """Inject generic helper data for raw batch traces."""
        batch_items = self._get_batch_items(traces)
        if batch_items is None:
            return

        item_ids = [
            self._get_batch_item_id(item, index)
            for index, item in enumerate(batch_items)
        ]
        item_id_to_index = {item_id: index for index, item_id in enumerate(item_ids)}
        item_preview_by_id = {
            item_id: self._build_item_preview(item)
            for item_id, item in zip(item_ids, batch_items)
        }
        survey_items = self._build_survey_items(batch_items)

        sandbox.inject("batch_items", batch_items)
        sandbox.inject("item_ids", item_ids)
        sandbox.inject("item_id_to_index", item_id_to_index)
        sandbox.inject("item_preview_by_id", item_preview_by_id)
        sandbox.inject("survey_items", survey_items)

    def _build_item_preview(self, item: Any) -> dict[str, Any]:
        """Build a compact preview row for a generic batch item."""
        item_trace = self._extract_batch_messages(item)
        first_message = ""
        if item_trace and isinstance(item_trace[0], dict):
            first_message = str(item_trace[0].get("content", ""))
        elif item_trace:
            first_message = str(item_trace[0])

        feedback = self._extract_batch_field(item, "feedback")
        question = self._extract_batch_field(item, "question")
        payload_type = type(self._extract_batch_item_payload(item)).__name__

        return {
            "question_preview": _preview(question, 120),
            "feedback_preview": _preview(feedback, 120),
            "first_message_preview": _preview(first_message, 120),
            "message_count": len(item_trace) if isinstance(item_trace, list) else 0,
            "payload_type": payload_type,
        }

    def _build_survey_items(self, batch_items: list[Any]) -> list[str]:
        """Precompute explicit batch_analyze items for grouped item survey."""
        survey_items: list[str] = []
        group_size = 3
        for start in range(0, len(batch_items), group_size):
            stop = min(start + group_size, len(batch_items))
            refs = []
            for index in range(start, stop):
                item_id = self._get_batch_item_id(batch_items[index], index)
                refs.append(f"batch_items[{index}] (item_id='{item_id}')")
            survey_items.append("Inspect " + ", ".join(refs))
        return survey_items

    def _build_data_summary(self, traces: Any) -> str:
        """Pre-compute a data summary so the agent doesn't waste calls exploring structure."""
        batch_items = self._get_batch_items(traces)
        is_batch = batch_items is not None

        if is_batch:
            assert batch_items is not None
            survey_items = self._build_survey_items(batch_items)
            # Compute pass/fail breakdown
            pass_count = 0
            fail_count = 0
            item_summaries = []
            for index, item in enumerate(batch_items):
                item_id = self._get_batch_item_id(item, index)
                trace = self._extract_batch_messages(item)
                feedback = self._extract_batch_field(item, "feedback")
                reward_str = ""
                if "reward=1.0" in str(feedback) or "PASSED" in str(feedback).upper():
                    pass_count += 1
                    reward_str = "PASS"
                elif "reward=0.0" in str(feedback) or "FAILED" in str(feedback).upper():
                    fail_count += 1
                    reward_str = "FAIL"
                item_summaries.append(
                    f"  {item_id}: {reward_str}, {len(trace)} messages"
                )

            lines = [
                "### Data Summary (pre-computed)",
                f"- **{len(batch_items)} batch items**: {pass_count} PASS, {fail_count} FAIL",
                f"- Item list:",
            ]
            # Show all tasks compactly
            lines.extend(item_summaries[:50])  # cap at 50
            if len(item_summaries) > 50:
                lines.append(f"  ... and {len(item_summaries) - 50} more")
            if survey_items:
                lines.append(
                    "- **Precomputed survey_items**: pass these strings directly "
                    "to `batch_analyze`; they already reference `batch_items[i]`."
                )
                lines.extend(f"  - {item}" for item in survey_items[:10])
                if len(survey_items) > 10:
                    lines.append(f"  - ... and {len(survey_items) - 10} more groups")

            # Check for policy/rules in the first batch item
            if batch_items:
                first_trace = self._extract_batch_messages(batch_items[0])
                for msg in first_trace[:3]:
                    content = (
                        str(msg.get("content", ""))
                        if isinstance(msg, dict)
                        else str(msg)
                    )
                    if len(content) > 500 and any(
                        kw in content.lower()
                        for kw in [
                            "policy",
                            "rule",
                            "instruction",
                            "you must",
                            "you should",
                        ]
                    ):
                        lines.append(
                            "- **Potential embedded policy/rules** detected in early "
                            f"content ({len(content)} chars) — inspect via helpers "
                            "or a small `execute_code` call."
                        )
                        break

            return "\n".join(lines)

        else:
            # Single trace
            if not isinstance(traces, dict):
                return "\n".join(
                    [
                        "### Data Summary (pre-computed)",
                        f"- **Trace type**: {type(traces).__name__}",
                        f'- **Preview**: "{_preview(str(traces), 200)}"',
                    ]
                )

            steps = traces.get("steps", [])
            question = traces.get("question", "")
            feedback = traces.get("feedback", "")
            ground_truth = traces.get("ground_truth", "")

            lines = ["### Data Summary (pre-computed)"]
            if feedback:
                lines.append(f"- **Feedback**: {_preview(feedback, 200)}")
            if ground_truth:
                lines.append(f"- **Ground truth**: {_preview(ground_truth, 200)}")
            lines.append(f"- **Steps**: {len(steps)}")
            if question:
                lines.append(f"- **Task**: {_preview(question, 200)}")

            # Check for messages in trace
            messages = traces.get("messages", [])
            if messages:
                lines.append(f"- **Messages**: {len(messages)} conversation turns")
                # Count tool calls
                tool_calls = sum(
                    1 for m in messages if isinstance(m, dict) and m.get("tool_calls")
                )
                if tool_calls:
                    lines.append(f"- **Tool calls**: {tool_calls}")

            return "\n".join(lines)

    def _build_initial_prompt(
        self,
        traces: Any,
        skillbook: Any,
        trace_obj: Any,
    ) -> str:
        """Format the prompt template with previews and metadata."""
        batch_items = self._get_batch_items(traces)
        is_batch = batch_items is not None
        t_steps = (
            traces.get("steps", []) if isinstance(traces, dict) and not is_batch else []
        )

        trace_size_chars = len(_json.dumps(traces, default=str))

        skillbook_text = ""
        if skillbook is not None:
            if hasattr(skillbook, "as_prompt"):
                skillbook_text = skillbook.as_prompt() or ""
            else:
                skillbook_text = str(skillbook)

        if is_batch:
            assert batch_items is not None
            total_steps = sum(
                len(self._extract_batch_messages(item)) for item in batch_items
            )
            preview_rows = []
            for index, item in enumerate(batch_items):
                item_id = self._get_batch_item_id(item, index)
                tr = self._extract_batch_messages(item)
                first_msg = ""
                if tr and isinstance(tr[0], dict):
                    first_msg = tr[0].get("content", "")
                elif tr:
                    first_msg = str(tr[0])
                preview_rows.append(
                    f"| `{item_id}` | {len(tr)} messages | "
                    f'"{_preview(first_msg, 80)}" |'
                )

            if isinstance(traces, list):
                traces_description = f"List of {len(batch_items)} raw trace items"
            elif isinstance(traces, dict):
                traces_description = (
                    f"Batch container with keys {sorted(traces.keys())}. "
                    "Use the injected `batch_items` helper list for iteration."
                )
            else:
                traces_description = (
                    f"Batch-like trace container of type {type(traces).__name__}"
                )

            fmt_kwargs = dict(
                traces_description=traces_description,
                batch_variables=(
                    f"| `batch_items` | Ordered batch view over the raw trace items | "
                    f"{len(batch_items)} items |\n"
                ),
                helper_variables=(
                    "| `helper_registry` | Registered reusable helper functions and descriptions | dynamic |\n"
                    "| `register_helper` | Persist helper code for this run and sub-agent snapshots | callable |\n"
                    "| `list_helpers()` | List registered helper names/descriptions | callable |\n"
                    "| `run_helper(name, *args, **kwargs)` | Invoke a registered helper by name | callable |\n"
                    "| `get_batch_item(index)` | Convenience accessor for `batch_items[index]` | callable |\n"
                    f"| `item_ids` | Ordered item ids for this batch | {len(batch_items)} ids |\n"
                    f"| `item_id_to_index` | Maps item id to `batch_items[i]` | {len(batch_items)} entries |\n"
                    f"| `item_preview_by_id` | Compact previews per batch item | {len(batch_items)} entries |\n"
                    f"| `survey_items` | Precomputed `batch_analyze` items with explicit item references | {len(self._build_survey_items(batch_items))} items |\n"
                ),
                traces_previews=(
                    f"| Item | Steps | First message |\n"
                    f"|------|-------|---------------|\n" + "\n".join(preview_rows)
                ),
                step_count=total_steps,
                skillbook_length=len(skillbook_text),
                trace_size_chars=trace_size_chars,
                max_iterations=self.config.max_llm_calls,
                task_count=len(batch_items),
                data_summary=self._build_data_summary(traces),
            )
        else:
            single_trace_type = type(traces).__name__
            if not isinstance(traces, dict):
                fmt_kwargs = dict(
                    traces_description=f"Raw trace object of type {single_trace_type}",
                    batch_variables="",
                    helper_variables=(
                        "| `helper_registry` | Registered reusable helper functions and descriptions | dynamic |\n"
                        "| `register_helper` | Persist helper code for this run and sub-agent snapshots | callable |\n"
                        "| `list_helpers()` | List registered helper names/descriptions | callable |\n"
                        "| `run_helper(name, *args, **kwargs)` | Invoke a registered helper by name | callable |\n"
                    ),
                    traces_previews=(
                        "| Field | Preview | Size |\n"
                        "|-------|---------|------|\n"
                        f'| raw trace | "{_preview(str(traces), 120)}" | {len(str(traces))} chars |'
                    ),
                    step_count=0,
                    skillbook_length=len(skillbook_text),
                    trace_size_chars=trace_size_chars,
                    max_iterations=self.config.max_llm_calls,
                    task_count=1,
                    data_summary=self._build_data_summary(traces),
                )
                return self.prompt_template.format(**fmt_kwargs)

            t_question = traces.get("question", "")
            t_ground_truth = traces.get("ground_truth")
            t_feedback = traces.get("feedback")
            first_agent: dict[str, str] = next(
                (s for s in t_steps if s.get("role") == "agent"), {}
            )
            t_reasoning = first_agent.get("reasoning", "")

            fmt_kwargs = dict(
                traces_description=(
                    "Dict with keys: question, ground_truth, feedback, "
                    "steps (List[Dict])"
                ),
                batch_variables="",
                helper_variables=(
                    "| `helper_registry` | Registered reusable helper functions and descriptions | dynamic |\n"
                    "| `register_helper` | Persist helper code for this run and sub-agent snapshots | callable |\n"
                    "| `list_helpers()` | List registered helper names/descriptions | callable |\n"
                    "| `run_helper(name, *args, **kwargs)` | Invoke a registered helper by name | callable |\n"
                ),
                traces_previews=(
                    f"| Field | Preview | Size |\n"
                    f"|-------|---------|------|\n"
                    f'| `traces["question"]` | "{_preview(t_question)}" '
                    f"| {len(t_question)} chars |\n"
                    f'| first step | "{_preview(t_reasoning)}..." '
                    f"| {len(t_reasoning) if t_reasoning else 0} chars |\n"
                    f'| `traces["ground_truth"]` | "{_preview(t_ground_truth)}" '
                    f"| {len(t_ground_truth) if t_ground_truth else 0} chars |\n"
                    f'| `traces["feedback"]` | "{_preview(t_feedback)}..." '
                    f"| {len(t_feedback) if t_feedback else 0} chars |"
                ),
                step_count=(
                    len(t_steps) if t_steps else (len(trace_obj) if trace_obj else 0)
                ),
                skillbook_length=len(skillbook_text),
                trace_size_chars=trace_size_chars,
                max_iterations=self.config.max_llm_calls,
                task_count=1,
                data_summary=self._build_data_summary(traces),
            )

        return self.prompt_template.format(**fmt_kwargs)

    # ------------------------------------------------------------------
    # Timeout / error fallback
    # ------------------------------------------------------------------

    def _build_timeout_output(
        self,
        question: str,
        agent_output: Optional[AgentOutput],
        ground_truth: Optional[str],
        feedback: Optional[str],
        deps: RRDeps,
    ) -> ReflectorOutput:
        """Build a ReflectorOutput when usage limits are reached."""
        is_correct = False
        if ground_truth and agent_output:
            is_correct = (
                agent_output.final_answer.strip().lower()
                == ground_truth.strip().lower()
            )

        return ReflectorOutput(
            reasoning=(
                f"Recursive analysis reached the main-agent usage limit "
                f"({self.config.max_llm_calls} requests). "
                f"Basic analysis: Answer was "
                f"{'correct' if is_correct else 'incorrect'}."
            ),
            error_identification="timeout" if not is_correct else "none",
            root_cause_analysis="Analysis incomplete due to request limit",
            correct_approach=(
                "Consider increasing max_llm_calls or simplifying the analysis"
            ),
            key_insight=(
                "Complex traces may require more requests for thorough analysis"
            ),
            extracted_learnings=[
                ExtractedLearning(
                    learning="Usage limit reached during recursive analysis",
                    atomicity_score=0.5,
                )
            ],
            skill_tags=[],
            raw={
                "timeout": True,
                "max_llm_calls": self.config.max_llm_calls,
                "max_llm_calls_scope": "main_agent_session",
                "question": question,
                "feedback": feedback,
                "rr_trace": {
                    "total_iterations": deps.iteration,
                    "subagent_calls": deps.sub_agent_history,
                    "timed_out": True,
                },
            },
        )
