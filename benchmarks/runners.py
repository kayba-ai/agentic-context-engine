"""
Benchmark execution runners for different execution modes.

This module provides specialized runners for different benchmark types:
- StandardRunner: Single-turn Q&A evaluation (default)
- HALRunner: Delegates to HAL harness for agentic benchmarks (AppWorld, SWE-bench)

The runner is selected automatically based on the benchmark's execution_mode
in its YAML configuration.
"""

from __future__ import annotations

import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from ace import EnvironmentResult, Skillbook

if TYPE_CHECKING:
    from ace import Agent
    from .base import BenchmarkConfig

logger = logging.getLogger(__name__)

# Path to HAL agents directory
AGENTS_DIR = Path(__file__).parent.parent / "agents"


@dataclass
class ExecutionStep:
    """A single step in an iterative execution."""

    step_number: int
    code: str
    output: str
    error: Optional[str] = None


class BenchmarkRunner(ABC):
    """Abstract base class for benchmark runners."""

    @abstractmethod
    def run_task(
        self,
        task_data: Dict[str, Any],
        agent: "Agent",
        skillbook: Skillbook,
        config: "BenchmarkConfig",
    ) -> EnvironmentResult:
        """
        Execute a single benchmark task.

        Args:
            task_data: Task metadata from the loader
            agent: ACE Agent for generating responses
            skillbook: Current skillbook context
            config: Benchmark configuration

        Returns:
            EnvironmentResult with feedback, ground_truth, and metrics
        """
        pass


class StandardRunner(BenchmarkRunner):
    """Runner for standard single-turn Q&A benchmarks."""

    def run_task(
        self,
        task_data: Dict[str, Any],
        agent: "Agent",
        skillbook: Skillbook,
        config: "BenchmarkConfig",
    ) -> EnvironmentResult:
        """
        Execute a standard Q&A task.

        This is the default execution mode for most benchmarks.
        The agent generates a single response which is then evaluated.
        """
        question = task_data.get("question", "")
        context = task_data.get("context", "")
        ground_truth = task_data.get("ground_truth", "")

        # Generate agent response
        output = agent.generate(
            question=question,
            context=context,
            skillbook=skillbook,
        )

        # Return result (actual evaluation happens in environment)
        return EnvironmentResult(
            feedback=f"Generated answer: {output.final_answer}",
            ground_truth=ground_truth,
            metrics={"generated": 1.0},
        )


class HALRunner(BenchmarkRunner):
    """
    Runner that delegates to HAL harness for agentic benchmarks.

    HAL (Holistic Agent Leaderboard) provides isolated execution environments
    for benchmarks like AppWorld, SWE-bench, etc. This runner wraps HAL
    to provide a consistent interface with ACE.

    HAL Setup Required:
        git clone --recursive https://github.com/princeton-pli/hal-harness.git
        cd hal-harness && pip install -e .

    See: https://github.com/princeton-pli/hal-harness
    See: agents/README.md for detailed setup instructions
    """

    def __init__(self, benchmark_name: str = "appworld_test_normal"):
        """
        Initialize HAL runner.

        Args:
            benchmark_name: HAL benchmark name (e.g., appworld_test_normal)
        """
        self.benchmark_name = benchmark_name
        self.agent_dir = AGENTS_DIR / "ace_agent"
        self._hal_available = None

    def _check_hal_available(self) -> bool:
        """Check if HAL is properly installed."""
        if self._hal_available is None:
            try:
                result = subprocess.run(
                    ["hal-eval", "--help"],
                    capture_output=True,
                    timeout=10,
                )
                self._hal_available = result.returncode == 0
            except Exception:
                self._hal_available = False
        return self._hal_available

    def run_task(
        self,
        task_data: Dict[str, Any],
        agent: "Agent",
        skillbook: Skillbook,
        config: "BenchmarkConfig",
    ) -> EnvironmentResult:
        """
        Execute task via HAL harness.

        Note: For efficiency, HAL is typically run on batches of tasks.
        This method runs a single task which may not be optimal.
        Consider using run_benchmark() for batch execution.
        """
        # Check HAL availability
        if not self._check_hal_available():
            return EnvironmentResult(
                feedback=(
                    "HAL harness not available. Install from: "
                    "git clone --recursive https://github.com/princeton-pli/hal-harness.git && "
                    "cd hal-harness && pip install -e . "
                    "See agents/README.md for details."
                ),
                ground_truth="Task completion",
                metrics={"task_success": 0.0, "error": 1.0},
            )

        task_id = task_data.get("task_id", "unknown")

        try:
            # Run HAL for single task
            result = self._run_hal_single(
                task_id=task_id,
                config=config,
            )

            return EnvironmentResult(
                feedback=f"HAL result: {result.get('status', 'unknown')}",
                ground_truth="Task completion",
                metrics={
                    "task_success": float(result.get("success", False)),
                },
            )
        except Exception as e:
            logger.error(f"HAL execution failed: {e}")
            return EnvironmentResult(
                feedback=f"HAL error: {e}",
                ground_truth="Task completion",
                metrics={"task_success": 0.0, "error": 1.0},
            )

    def _run_hal_single(
        self,
        task_id: str,
        config: "BenchmarkConfig",
    ) -> Dict[str, Any]:
        """Run HAL for a single task."""
        # Build HAL command
        cmd = [
            "hal-eval",
            "--benchmark",
            self.benchmark_name,
            "--agent_dir",
            str(self.agent_dir),
            "--agent_function",
            "main.run",
            "--agent_name",
            "ACE Agent",
            "-A",
            f"max_steps={config.max_steps}",
        ]

        # Run HAL
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"HAL failed: {result.stderr}")

        return {"status": "completed", "success": True}

    @classmethod
    def run_benchmark(
        cls,
        benchmark_name: str,
        model: str = "gpt-4o-mini",
        skillbook_path: Optional[str] = None,
        max_concurrent: int = 5,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run full benchmark via HAL harness.

        This is the recommended way to run agentic benchmarks.

        Args:
            benchmark_name: HAL benchmark (appworld_test_normal, swebench_lite, etc.)
            model: LLM model to use
            skillbook_path: Path to skillbook JSON
            max_concurrent: Maximum concurrent tasks
            output_dir: Directory for results

        Returns:
            Benchmark results from HAL

        Example:
            results = HALRunner.run_benchmark(
                "appworld_test_normal",
                model="gpt-4o",
                max_concurrent=10,
            )
        """
        agent_dir = AGENTS_DIR / "ace_agent"

        # Build command
        cmd = [
            "hal-eval",
            "--benchmark",
            benchmark_name,
            "--agent_dir",
            str(agent_dir),
            "--agent_function",
            "main.run",
            "--agent_name",
            f"ACE Agent ({model})",
            "--max_concurrent",
            str(max_concurrent),
            "-A",
            f"model={model}",
        ]

        if skillbook_path:
            cmd.extend(["-A", f"skillbook_path={skillbook_path}"])

        if output_dir:
            cmd.extend(["--output_dir", output_dir])

        logger.info(f"Running HAL: {' '.join(cmd)}")

        # Run HAL
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"HAL failed: {result.stderr}")

        return {
            "status": "completed",
            "stdout": result.stdout,
        }


# Backwards compatibility alias
IterativeRunner = HALRunner


def get_runner(
    execution_mode: str, benchmark_name: str = "appworld"
) -> BenchmarkRunner:
    """
    Get the appropriate runner for an execution mode.

    Args:
        execution_mode: One of 'standard', 'iterative', 'hal'
        benchmark_name: Benchmark name for HAL runner

    Returns:
        Appropriate BenchmarkRunner instance
    """
    if execution_mode in ("iterative", "hal"):
        # Map benchmark names to HAL benchmark names
        hal_benchmarks = {
            "appworld": "appworld_test_normal",
            "swe_bench": "swebench_lite",
        }
        hal_name = hal_benchmarks.get(benchmark_name, benchmark_name)
        return HALRunner(benchmark_name=hal_name)

    return StandardRunner()
