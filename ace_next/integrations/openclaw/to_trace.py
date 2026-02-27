"""OpenClawToTraceStep — convert raw JSONL events to a structured trace dict."""

from __future__ import annotations

from typing import Any

from ...core.context import ACEStepContext


class OpenClawToTraceStep:
    """Convert raw OpenClaw JSONL events into a structured trace dict.

    This step receives ``ctx.trace`` as a ``list[dict]`` of raw JSONL events
    (placed by ``LoadTracesStep``) and converts them into the trace dict
    format expected by ``ReflectStep``::

        {
            "question": str,      # reconstructed conversation
            "reasoning": str,     # full execution trace (thinking + tool calls)
            "answer": str,        # last assistant text
            "skill_ids": list,    # always [] for OpenClaw
            "feedback": str,      # session summary
            "ground_truth": None,
        }

    Follows the same pattern as ``BrowserToTrace``, ``LangChainToTrace``,
    and ``ClaudeCodeToTrace``.
    """

    requires = frozenset({"trace"})
    provides = frozenset({"trace"})

    def __call__(self, ctx: ACEStepContext) -> ACEStepContext:
        events: list[dict[str, Any]] = ctx.trace  # type: ignore[assignment]
        if not events:
            return ctx

        trace_dict = _events_to_trace(events)
        return ctx.replace(trace=trace_dict)


def _events_to_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert a list of OpenClaw JSONL events into the standardised trace dict."""
    user_messages: list[str] = []
    assistant_texts: list[str] = []
    reasoning_parts: list[str] = []
    model = ""
    total_tokens = 0

    for event in events:
        etype = event.get("type")

        if etype == "session":
            model = event.get("cwd", "")
            continue

        if etype == "custom":
            data = event.get("data", {})
            if data.get("modelId"):
                model = data["modelId"]
            continue

        if etype != "message":
            continue

        msg = event.get("message", {})
        role = msg.get("role")
        content_blocks = msg.get("content", [])

        # Track token usage
        usage = msg.get("usage", {})
        total_tokens += usage.get("totalTokens", 0)

        # Track model
        if msg.get("model"):
            model = msg["model"]

        if role == "user":
            for block in content_blocks:
                if block.get("type") == "text":
                    user_messages.append(block["text"])

        elif role == "assistant":
            for block in content_blocks:
                btype = block.get("type")
                if btype == "thinking":
                    reasoning_parts.append(
                        f"[thinking] {block.get('thinking', '')}"
                    )
                elif btype == "text":
                    text = block.get("text", "")
                    assistant_texts.append(text)
                    reasoning_parts.append(f"[response] {text}")
                elif btype == "toolCall":
                    name = block.get("name", "unknown")
                    args = block.get("arguments", {})
                    reasoning_parts.append(f"[tool:{name}] {args}")

        elif role == "toolResult":
            tool_name = msg.get("toolName", "unknown")
            for block in content_blocks:
                if block.get("type") == "text":
                    text = block["text"]
                    # Truncate long tool results
                    if len(text) > 500:
                        text = text[:500] + "..."
                    reasoning_parts.append(f"[tool_result:{tool_name}] {text}")

    # Build the conversation as the "question"
    question = "\n\n".join(f"User: {m}" for m in user_messages) if user_messages else ""

    # Last assistant text as the "answer"
    answer = assistant_texts[-1] if assistant_texts else ""

    # Build feedback summary
    n_user = len(user_messages)
    n_assistant = len(assistant_texts)
    feedback = (
        f"OpenClaw session: {n_user} user messages, {n_assistant} assistant responses"
    )
    if model:
        feedback += f", model: {model}"
    if total_tokens:
        feedback += f", {total_tokens} tokens"

    return {
        "question": question,
        "reasoning": "\n".join(reasoning_parts),
        "answer": answer,
        "skill_ids": [],
        "feedback": feedback,
        "ground_truth": None,
    }
