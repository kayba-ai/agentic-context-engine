"""Pure functions for trimming REPL message history to fit context budgets."""

from __future__ import annotations

from typing import Optional


def trim_messages(
    messages: list[dict[str, str]], max_chars: int
) -> list[dict[str, str]]:
    """Trim messages to fit within *max_chars* using semantic scoring.

    Scores each iteration by importance and keeps highest-value ones:
    - Errors are high value (debugging context)
    - Findings/insights are high value
    - Substantive output is medium value
    - Empty output is low value

    Always keeps the first message (instructions) and ensures chronological
    order of kept messages.
    """
    total = sum(len(m["content"]) for m in messages)
    if total <= max_chars:
        return messages

    first = messages[0]
    remaining_budget = max_chars - len(first["content"])

    # Group into iteration pairs (assistant + user output)
    iterations: list[tuple[int, dict, dict | None]] = []
    i = 1
    while i < len(messages):
        if i + 1 < len(messages):
            iterations.append((i, messages[i], messages[i + 1]))
            i += 2
        else:
            iterations.append((i, messages[i], None))
            i += 1

    # Score and sort by importance
    scored: list[tuple[float, int, dict, dict | None, int]] = []
    for idx, asst_msg, user_msg in iterations:
        score = score_iteration(asst_msg, user_msg)
        pair_size = len(asst_msg["content"])
        if user_msg:
            pair_size += len(user_msg["content"])
        scored.append((score, idx, asst_msg, user_msg, pair_size))

    scored.sort(key=lambda x: (-x[0], x[1]))

    kept_indices: list[tuple[int, dict, dict | None]] = []
    used_budget = 0
    for score, idx, asst_msg, user_msg, pair_size in scored:
        if used_budget + pair_size <= remaining_budget:
            kept_indices.append((idx, asst_msg, user_msg))
            used_budget += pair_size

    # Restore chronological order
    kept_indices.sort(key=lambda x: x[0])

    kept: list[dict[str, str]] = []
    for _, asst_msg, user_msg in kept_indices:
        kept.append(asst_msg)
        if user_msg:
            kept.append(user_msg)

    dropped_count = len(iterations) - len(kept_indices)
    if dropped_count > 0:
        dropped_pairs = [
            (asst, user) for _, _, asst, user, _ in scored[len(kept_indices) :]
        ]
        summary_text = summarize_dropped(dropped_pairs)
        summary = {
            "role": "user",
            "content": f"[{dropped_count} earlier iterations omitted: {summary_text}]",
        }
        return [first, summary] + kept

    return [first] + kept


def score_iteration(
    assistant_msg: dict[str, str], user_msg: Optional[dict[str, str]]
) -> float:
    """Score iteration importance for retention priority.

    Higher scores = more valuable context to keep.
    """
    score = 0.0

    if user_msg:
        content = user_msg["content"]

        error_indicators = ["Error", "Exception", "Traceback", "stderr:"]
        if any(ind in content for ind in error_indicators):
            score += 3.0

        finding_indicators = ["found", "pattern", "insight", "discovered", "result:"]
        if any(ind.lower() in content.lower() for ind in finding_indicators):
            score += 2.0

        if len(content) > 500:
            score += 1.0

        if "(no output)" in content:
            score -= 1.0

    if "FINAL(" in assistant_msg["content"]:
        score += 2.0

    if (
        "ask_llm(" in assistant_msg["content"]
        or "llm_query(" in assistant_msg["content"]
    ):
        score += 1.0

    return score


def summarize_dropped(dropped: list[tuple[dict, dict | None]]) -> str:
    """Generate a brief semantic summary of dropped iterations."""
    if not dropped:
        return "no significant findings"

    summaries: list[str] = []
    error_count = 0
    explore_count = 0

    for asst_msg, user_msg in dropped:
        if user_msg and any(
            ind in user_msg["content"] for ind in ["Error", "Exception", "stderr:"]
        ):
            error_count += 1
        if "print(" in asst_msg["content"]:
            explore_count += 1

    if error_count:
        summaries.append(f"{error_count} error(s)")
    if explore_count:
        summaries.append(f"{explore_count} exploration(s)")

    return ", ".join(summaries) if summaries else "exploration iterations"
