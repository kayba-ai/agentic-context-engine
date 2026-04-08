"""Configuration for recursive reflector."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RecursiveConfig:
    """Configuration for the RecursiveReflector.

    Attributes:
        max_iterations: Maximum number of REPL iterations before timing out (default: 20)
        timeout: Timeout in seconds for each code execution (default: 30.0)
        enable_llm_query: Whether to allow llm_query() function in sandbox (default: True)
        max_llm_calls: Maximum number of LLM calls allowed for the main RR agent session (default: 30)
        max_context_chars: Maximum total characters in message history before trimming (default: 50000)
        max_output_chars: Maximum characters per code execution output before truncation (default: 20000)
        enable_subagent: Whether to enable the analyze()/batch_analyze() sub-agent tools (default: True)
        subagent_model: Model to use for sub-agent (None = same as main reflector)
        subagent_max_tokens: Max tokens for sub-agent responses (default: 8192)
        subagent_temperature: Temperature for sub-agent responses (default: 0.3)
        subagent_system_prompt: Custom system prompt for sub-agent (None = default)
        subagent_max_requests: Max requests per sub-agent run (default: 15)
        subagent_max_parallel: Max concurrent sub-agents in batch_analyze() (default: 10)
        enable_fallback_synthesis: Whether to attempt LLM synthesis on timeout (default: True)
        orchestrator_max_llm_calls: Max LLM calls for the orchestrator session (default: 50)
        max_cluster_workers: Max concurrent worker RR sessions (default: 5)
        worker_collect_timeout: Seconds to wait when collecting worker results (default: 120.0)
        worker_model: Model to use for delegated worker RR sessions (default: None = same as main)
        worker_enable_subagent: Whether workers can use analyze/batch_analyze (default: False)
        worker_max_llm_calls: Hard cap for each delegated worker RR session (default: 12)
        worker_max_items: Max traces allowed in a single delegated worker assignment (default: 6)
        worker_subagent_max_parallel: Max concurrent sub-agents per worker (default: 2)
        local_parallel_max_concurrency: Max threads for parallel_map extraction (default: 8)
        local_parallel_timeout: Per-item timeout for parallel_map (default: 30.0)
    """

    max_iterations: int = 20
    timeout: float = 30.0
    enable_llm_query: bool = True
    max_llm_calls: int = 30
    max_context_chars: int = 50_000
    max_output_chars: int = 20_000
    # Sub-agent configuration
    enable_subagent: bool = True
    subagent_model: Optional[str] = None
    subagent_max_tokens: int = 8192
    subagent_temperature: float = 0.3
    subagent_system_prompt: Optional[str] = None
    subagent_max_requests: int = 15
    subagent_max_parallel: int = 10
    enable_fallback_synthesis: bool = True
    # Orchestration settings (batch RR manager/worker pattern)
    orchestrator_max_llm_calls: int = 50
    max_cluster_workers: int = 5
    worker_collect_timeout: float = 120.0
    worker_model: Optional[str] = None
    worker_enable_subagent: bool = False
    worker_max_llm_calls: int = 12
    worker_max_items: int = 6
    worker_subagent_max_parallel: int = 2
    local_parallel_max_concurrency: int = 8
    local_parallel_timeout: float | None = 30.0
