"""Configuration for recursive reflector."""

from dataclasses import dataclass

from pydantic_ai.usage import UsageLimits


@dataclass
class RecursiveConfig:
    """Configuration for the RecursiveReflector.

    Attributes:
        timeout: Timeout in seconds for each code execution (default: 30.0)
        max_tokens: Total token budget (input + output) per PydanticAI agent run (default: 500_000)
        max_requests: Safety cap on LLM requests per PydanticAI agent run (default: 50)
        context_window: Model context window size; compaction triggers at 85% of this (default: 128_000)
        max_output_chars: Maximum characters per code execution output before truncation (default: 20000)
        max_depth: Maximum recursion depth (0=root, max_depth=leaf with no recurse tool) (default: 2)
        child_budget_fraction: Fraction of remaining token budget given to each child (default: 0.5)
        max_compactions: Safety cap on full summarization rounds per session (default: 3)
        microcompact_keep_recent: Number of most recent tool results to preserve during microcompaction (default: 3)
    """

    timeout: float = 30.0
    # Budget (wired to PydanticAI UsageLimits)
    max_tokens: int = 500_000
    max_requests: int = 50
    context_window: int = 128_000
    max_output_chars: int = 20_000
    # Recursion
    max_depth: int = 2
    child_budget_fraction: float = 0.5
    # Compaction
    max_compactions: int = 3
    microcompact_keep_recent: int = 3

    def build_usage_limits(self, remaining_tokens: int | None = None) -> UsageLimits:
        """Build PydanticAI UsageLimits from this config.

        Args:
            remaining_tokens: If provided, use this as total_tokens_limit
                             instead of self.max_tokens (for child sessions).
        """
        return UsageLimits(
            total_tokens_limit=remaining_tokens or self.max_tokens,
            request_limit=self.max_requests,
            request_tokens_limit=int(self.context_window * 0.85),
            count_tokens_before_request=True,
        )
