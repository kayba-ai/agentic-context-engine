#!/usr/bin/env python3
"""
Run TAU-bench (τ²-bench) evaluation with ACE framework.

TAU-bench evaluates tool-calling agents in customer service domains
(airline, retail, telecom) using multi-turn conversations and database
state assertions.

Key metrics:
- pass^k: Run each task k times, pass only if ALL k succeed (consistency)
- ACE epochs: Train skillbook on subset before evaluation

Workflow: Train on subset (ACE epochs) → Evaluate on test set with pass^k (frozen skillbook)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from ace import (
    Reflector,
    ReflectorMode,
    SkillManager,
    Skillbook,
    AgentOutput,
)
from ace.llm_providers import LiteLLMClient
from ace.reflector.trace_context import TraceContext

# Suppress LiteLLM debug messages
import litellm

litellm.suppress_debug_info = True

# TAU2 imports
from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, ToolMessage
from tau2.data_model.simulation import SimulationRun
from tau2.metrics.agent_metrics import pass_hat_k
from tau2.registry import registry
from tau2.run import run_task


class ACELLMAgent(LLMAgent):
    """LLMAgent with ACE skillbook injection into the system prompt."""

    # Class-level skillbook (set before each run)
    _skillbook: Optional[Skillbook] = None
    _playbook_text: Optional[str] = None

    @classmethod
    def set_skillbook(cls, skillbook: Optional[Skillbook]):
        """Set the skillbook to inject into the system prompt."""
        cls._skillbook = skillbook

    @classmethod
    def set_playbook_text(cls, text: Optional[str]):
        """Set raw playbook text for direct system prompt injection."""
        cls._playbook_text = text

    @property
    def system_prompt(self) -> str:
        """Return system prompt with skillbook/playbook strategies appended."""
        base_prompt = super().system_prompt

        # Raw playbook injection takes priority
        if self._playbook_text:
            return (
                base_prompt
                + f"\n\n<learned_strategies>\n{self._playbook_text}\n</learned_strategies>\n"
            )

        if self._skillbook and len(self._skillbook.skills()) > 0:
            from ace.prompts_v3 import wrap_skillbook_for_external_agent

            wrapped = wrap_skillbook_for_external_agent(self._skillbook)
            return (
                base_prompt
                + f"\n\n<learned_strategies>\n{wrapped}\n</learned_strategies>\n"
            )

        return base_prompt


# Register the custom agent with tau2's registry
try:
    registry.register_agent(ACELLMAgent, "ace_llm_agent")
except ValueError:
    # Already registered (e.g., when running multiple times in same process)
    pass


# --- Opik tracing setup ---


def setup_opik_tracing(
    domain: str, model: str, project_name: Optional[str] = None
) -> Optional["OpikIntegration"]:
    """Set up Opik tracing for all LiteLLM calls (agent + user simulator).

    Registers OpikLogger on litellm.callbacks so every litellm.completion()
    call from tau2 is automatically traced — zero tau2 modifications needed.

    Returns the OpikIntegration instance, or None if unavailable/disabled.
    """
    try:
        from ace.observability.opik_integration import (
            OpikIntegration,
            _should_skip_opik,
        )
    except ImportError:
        return None

    if _should_skip_opik():
        return None

    try:
        integration = OpikIntegration(
            project_name=project_name or "tau-bench",
            tags=["tau-bench", domain, model],
        )
        if not integration.enabled:
            return None
        integration.setup_litellm_callback()
        return integration
    except Exception:
        return None


def run_single_task_traced(
    task: Dict[str, Any],
    skillbook: Skillbook,
    args: argparse.Namespace,
    *,
    phase: str = "eval",
    trial: int = 0,
    experiment_name: str = "",
) -> Tuple[Dict[str, Any], Optional[SimulationRun]]:
    """Wrap run_single_task with an Opik trace per task execution.

    Each call becomes a parent trace; all litellm.completion() calls inside
    become child spans automatically via the OpikLogger callback.
    Falls back to plain run_single_task when Opik is not installed.
    """
    try:
        from opik import track as opik_track

        @opik_track(
            name=f"tau_{task['domain']}_{task['task_id']}",
            project_name="tau-bench",
            tags=[
                f"domain:{task['domain']}",
                f"phase:{phase}",
                f"trial:{trial}",
                f"model:{args.model}",
            ],
            metadata={
                "task_id": task["task_id"],
                "domain": task["domain"],
                "phase": phase,
                "trial": trial,
                "experiment_name": experiment_name,
                "model": args.model,
            },
        )
        def _traced_run() -> Tuple[Dict[str, Any], Optional[SimulationRun]]:
            result, sim = run_single_task(task, skillbook, args)
            # Attach reward as Opik feedback score
            try:
                from opik import opik_context

                opik_context.update_current_trace(
                    feedback_scores=[
                        {
                            "name": "reward",
                            "value": result.get("reward", 0.0),
                            "reason": f"tau2 reward for {task['task_id']}",
                        }
                    ],
                )
            except Exception:
                pass
            return result, sim

        return _traced_run()
    except ImportError:
        return run_single_task(task, skillbook, args)


CONFIG_DIR = ROOT / "benchmarks" / "tasks" / "tau_bench"


def load_config(name_or_path: str) -> Dict[str, Any]:
    """Load a YAML config profile, resolving inheritance.

    Args:
        name_or_path: Profile name (e.g. "sonnet") or path to YAML file.

    Returns:
        Merged config dict (parent values overridden by child values).
    """
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"

    if not path.exists():
        print(f"Error: Config not found: {path}")
        sys.exit(1)

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    # Resolve inheritance
    parent_name = cfg.pop("inherits", None)
    if parent_name:
        parent = load_config(parent_name)
        # Deep-merge ace section
        parent_ace = parent.pop("ace", {})
        child_ace = cfg.pop("ace", {})
        merged = {**parent, **cfg}
        merged["ace"] = {**parent_ace, **child_ace}
        merged["_config_file"] = str(path)
        return merged

    cfg["_config_file"] = str(path)
    return cfg


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Config profile
    parser.add_argument(
        "--config",
        default="default",
        help="Config profile name or path to YAML (default: default)",
    )

    # Domain configuration — defaults are None so CLI overrides are detectable
    parser.add_argument(
        "--domain",
        choices=["airline", "retail", "telecom", "all"],
        default=None,
        help="Domain to evaluate (default: from config)",
    )
    parser.add_argument(
        "--task-split",
        choices=["base", "train", "test", "human", "gpt4o"],
        default=None,
        help="Task split to use (default: from config)",
    )

    # Data configuration
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks to evaluate (default: all)",
    )

    # Pass^k configuration
    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=None,
        help="K value for pass^k metric (default: from config)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps per task (default: from config)",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=None,
        help="Maximum errors before termination (default: from config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: from config)",
    )

    # ACE configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of ACE training epochs (default: from config)",
    )
    parser.add_argument(
        "--max-refinement-rounds",
        type=int,
        default=None,
        help="Maximum refinement rounds per sample (default: from config)",
    )
    parser.add_argument(
        "--skip-ace",
        action="store_true",
        help="Skip ACE training, run baseline only",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both baseline and ACE, then compare results",
    )
    parser.add_argument(
        "--skillbook",
        type=str,
        default=None,
        help="Path to pre-trained skillbook JSON. Skips training, evaluates directly.",
    )
    parser.add_argument(
        "--playbook",
        type=str,
        default=None,
        help="Path to markdown/text playbook for direct system prompt injection.",
    )
    parser.add_argument(
        "--batch-reflect",
        action="store_true",
        default=None,
        help="Defer learning until all tasks complete, then reflect on all traces together",
    )
    parser.add_argument(
        "--trace-limit",
        type=int,
        default=None,
        help="Max tokens per trace in batch mode (default: from config)",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default=None,
        help="Agent model to use (default: from config)",
    )
    parser.add_argument(
        "--user-llm",
        default=None,
        help="User simulator model (default: from config)",
    )
    parser.add_argument(
        "--reflector-model",
        type=str,
        default=None,
        help="Model for Reflector/SkillManager (default: same as --model)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: from config)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate (default: from config)",
    )

    # Output configuration
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for results (default: tau_benchmark_results)",
    )
    parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="Save detailed per-task results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--opik-project",
        type=str,
        default=None,
        help="Opik project name for tracing (e.g. 'haiku-cleaned-50-airline')",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        default=None,
        help="Comma-separated task IDs to run (e.g. '35,37,44,45,48'). Filters loaded tasks.",
    )
    parser.add_argument(
        "--feedback-level",
        choices=["trace", "outcome", "full"],
        default=None,
        help="What Reflector sees: trace=conversation only, outcome=+reward (default), full=+assertions",
    )

    args = parser.parse_args()

    # Load config and merge: config < CLI overrides
    cfg = load_config(args.config)
    ace_cfg = cfg.pop("ace", {})

    # Mapping from config keys to argparse dest names
    flat = {**cfg, **ace_cfg}
    # Normalize config keys: YAML uses underscores, argparse uses underscores too
    key_map = {
        "user_llm": "user_llm",
        "task_split": "task_split",
        "max_steps": "max_steps",
        "max_errors": "max_errors",
        "max_tokens": "max_tokens",
        "batch_reflect": "batch_reflect",
        "trace_limit": "trace_limit",
        "max_refinement_rounds": "max_refinement_rounds",
        "feedback_level": "feedback_level",
    }

    for cfg_key, value in flat.items():
        if cfg_key.startswith("_"):
            continue
        attr = key_map.get(cfg_key, cfg_key)
        # Only apply config value if CLI didn't set it (CLI value is None)
        if hasattr(args, attr) and getattr(args, attr) is None:
            setattr(args, attr, value)

    # Store config metadata for banner
    args._config_name = args.config
    args._config_file = cfg.get("_config_file", "")

    # Final fallback defaults for anything still None
    _fallbacks = {
        "domain": "airline",
        "task_split": "test",
        "k": 4,
        "max_steps": 200,
        "max_errors": 10,
        "seed": 300,
        "epochs": 1,
        "max_refinement_rounds": 3,
        "batch_reflect": False,
        "trace_limit": 500,
        "feedback_level": "outcome",
        "model": "gpt-4.1-mini-2025-04-14",
        "user_llm": "gpt-4.1-2025-04-14",
        "temperature": 0.0,
        "max_tokens": 2048,
        "output": "tau_benchmark_results",
    }
    for attr, default in _fallbacks.items():
        if getattr(args, attr, None) is None:
            setattr(args, attr, default)

    return args


def create_llm_client(
    args: argparse.Namespace, model: Optional[str] = None
) -> LiteLLMClient:
    """Create LLM client with specified configuration."""
    return LiteLLMClient(
        model=model or args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=120,
    )


def load_tau_tasks(
    args: argparse.Namespace, split: str = "base"
) -> List[Dict[str, Any]]:
    """Load TAU-bench tasks for the specified domain and split.

    Args:
        args: Command line arguments
        split: Task split to load ("base", "train", "test", "human", "gpt4o")
    """
    try:
        from benchmarks.loaders.tau2 import Tau2Loader
    except ImportError:
        print("Error: tau2-bench is not installed.")
        print(
            "Install with: pip install tau2-bench or pip install ace-framework[tau-bench]"
        )
        sys.exit(1)

    loader = Tau2Loader()
    domains = (
        ["airline", "retail", "telecom"] if args.domain == "all" else [args.domain]
    )

    all_tasks = []
    for domain in domains:
        if not args.quiet:
            print(f"Loading {domain} tasks (split: {split})...")

        tasks = list(
            loader.load(
                domain=domain,
                task_split=split,
                limit=args.limit,
                user_llm=args.user_llm,
            )
        )
        all_tasks.extend(tasks)

        if not args.quiet:
            print(f"  Loaded {len(tasks)} tasks from {domain}")

    return all_tasks


def extract_conversation_trace(
    simulation: SimulationRun, feedback_level: str = "outcome"
) -> str:
    """Extract agent-only trace from tau2 simulation for Reflector.

    Only keeps AssistantMessage (agent actions) and ToolMessage (environment
    responses). UserMessage turns are excluded because they encode
    task-specific knowledge from the user simulator's scenario prompt —
    the reflector should learn from the agent's behavior, not the task setup.

    Feedback levels control data leakage:
    - trace: conversation only (no reward, no assertions)
    - outcome: conversation + reward/steps (default, no ground truth leakage)
    - full: conversation + failed assertions + action checks (data leakage!)
    """
    lines = ["## Conversation Trace\n"]

    for msg in simulation.messages:
        if isinstance(msg, AssistantMessage):
            if msg.tool_calls:
                for tool in msg.tool_calls:
                    args_str = str(tool.arguments)
                    if len(args_str) > 500:
                        args_str = args_str[:500] + "..."
                    lines.append(f"Agent: [TOOL] {tool.name}({args_str})")
            elif msg.content:
                content = msg.content[:1000] if len(msg.content) > 1000 else msg.content
                lines.append(f"Agent: {content}")
        elif isinstance(msg, ToolMessage):
            status = "[ERROR]" if msg.error else "[OK]"
            content = msg.content[:1000] if len(msg.content) > 1000 else msg.content
            lines.append(f"Tool {status}: {content}")

    if feedback_level == "full":
        # WARNING: Including assertions causes data leakage - assertion text
        # encodes expected correct behavior (effectively ground truth).
        # Use --feedback-level full only for A/B leakage comparison.
        if simulation.reward_info and simulation.reward_info.nl_assertions:
            failed = [a for a in simulation.reward_info.nl_assertions if not a.met]
            if failed:
                lines.append("\n## Failed Assertions")
                for a in failed:
                    lines.append(f"- {a.nl_assertion}")
                    if a.justification:
                        lines.append(f"  Reason: {a.justification}")

        if simulation.reward_info and simulation.reward_info.action_checks:
            failed_actions = [
                a for a in simulation.reward_info.action_checks if not a.action_match
            ]
            if failed_actions:
                lines.append("\n## Failed Action Checks")
                for a in failed_actions:
                    lines.append(f"- Action: {a.action}")

        # Add termination reason if abnormal (only for full level)
        if (
            simulation.termination_reason
            and simulation.termination_reason.value != "success"
        ):
            lines.append(f"\n## Termination: {simulation.termination_reason.value}")

    return "\n".join(lines)


def run_single_task(
    task: Dict[str, Any],
    skillbook: Skillbook,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Optional[SimulationRun]]:
    """
    Run a single TAU task using tau2's run_task with skillbook injection.

    This uses tau2's proper tool-calling LLMAgent (via ACELLMAgent subclass)
    instead of ACE's simple text-based Agent.

    Returns:
        Tuple of (result_dict, simulation_object) where simulation_object
        contains the full conversation trace for meso-level learning.
    """
    # Set skillbook on the custom agent class before running
    ACELLMAgent.set_skillbook(skillbook)

    try:
        # Run with our custom agent that has skillbook injected
        simulation = run_task(
            domain=task["domain"],
            task=task["task"],  # The actual tau2 Task object
            agent="ace_llm_agent",  # Our registered custom agent
            user="user_simulator",
            llm_agent=args.model,
            llm_args_agent={"temperature": args.temperature},
            llm_user=args.user_llm,
            llm_args_user={"temperature": 0.0},
            max_steps=args.max_steps,
            max_errors=args.max_errors,
            seed=args.seed,
        )

        reward = simulation.reward_info.reward if simulation.reward_info else 0.0
        result = {
            "task_id": task["task_id"],
            "domain": task["domain"],
            "reward": reward,
            "success": reward >= 1.0,
            "steps": len(simulation.messages) if simulation.messages else 0,
            "cost": getattr(simulation, "agent_cost", None),
        }
        return result, simulation
    except Exception as e:
        result = {
            "task_id": task["task_id"],
            "domain": task["domain"],
            "reward": 0.0,
            "success": False,
            "steps": 0,
            "cost": None,
            "error": str(e),
        }
        return result, None


def evaluate_pass_k(
    tasks: List[Dict[str, Any]],
    skillbook: Skillbook,
    args: argparse.Namespace,
    k: int = 1,
    quiet: bool = False,
    phase: str = "eval",
    experiment_name: str = "",
) -> Dict[str, Any]:
    """
    Evaluate tasks using pass^k metric per TAU-bench paper (arXiv:2406.12045).

    pass^k = average of C(successes, k) / C(trials, k) across all tasks.

    This is a combinatorial probability: the chance that all k randomly
    selected trials from the pool would be successes.
    """
    results = []
    pass_sums = {i: 0.0 for i in range(1, k + 1)}  # Accumulate pass^1, ..., pass^k

    for i, task in enumerate(tasks):
        if not quiet:
            print(
                f"  Task {i + 1}/{len(tasks)}: {task['task_id']}", end=" ", flush=True
            )

        # Run k trials
        trial_results = []
        for trial_idx in range(k):
            trial_result, simulation = run_single_task_traced(
                task,
                skillbook,
                args,
                phase=phase,
                trial=trial_idx,
                experiment_name=experiment_name,
            )
            # Preserve simulation summary in trial result
            if simulation is not None:
                trial_result["simulation_summary"] = {
                    "id": getattr(simulation, "id", None),
                    "duration": getattr(simulation, "duration", None),
                    "termination_reason": (
                        str(simulation.termination_reason)
                        if simulation.termination_reason
                        else None
                    ),
                    "num_messages": (
                        len(simulation.messages) if simulation.messages else 0
                    ),
                    "agent_cost": getattr(simulation, "agent_cost", None),
                    "user_cost": getattr(simulation, "user_cost", None),
                }
            trial_results.append(trial_result)

        # Record final pass^k for this task
        task_passed_all = all(tr["success"] for tr in trial_results)

        # Compute pass^j for each j using combinatorial formula
        num_successes = sum(1 for tr in trial_results if tr["success"])
        task_pass_k = {}
        for j in range(1, k + 1):
            task_pass_k[j] = pass_hat_k(k, num_successes, j)

        results.append(
            {
                "task_id": task["task_id"],
                "domain": task["domain"],
                "trials": trial_results,
                "passed_all": task_passed_all,
                "pass_k_values": task_pass_k,
            }
        )

        # Accumulate for averaging
        for j in range(1, k + 1):
            pass_sums[j] += task_pass_k[j]

        if not quiet:
            status = "✓" if task_passed_all else "✗"
            reward = trial_results[0]["reward"] if trial_results else 0.0
            print(f"{status} (reward={reward:.2f}, pass^k={task_pass_k})")

    # Average pass^k across all tasks
    n_tasks = len(tasks)
    metrics = {}
    for j in range(1, k + 1):
        metrics[f"pass_{j}"] = pass_sums[j] / n_tasks if n_tasks > 0 else 0.0

    return {
        "tasks_evaluated": n_tasks,
        "k": k,
        "pass_sums": pass_sums,
        "metrics": metrics,
        "results": results,
    }


def run_ace_training(
    train_tasks: List[Dict[str, Any]],
    args: argparse.Namespace,
    quiet: bool = False,
    experiment_name: str = "",
) -> Skillbook:
    """
    Run ACE training on train tasks.

    Uses tau2's run_task with our ACELLMAgent to execute tasks,
    then learns from the results using ACE's Reflector and SkillManager.
    """
    if not quiet:
        print(
            f"\n📚 ACE Training Phase ({len(train_tasks)} tasks × {args.epochs} epochs)"
        )

    from ace.prompt_manager import PromptManager

    pm = PromptManager()

    # Use reflector-model for Reflector/SkillManager if specified
    reflector_model = getattr(args, "reflector_model", None) or args.model
    reflector_client = create_llm_client(args, model=reflector_model)
    reflector = Reflector(reflector_client, mode=ReflectorMode.RECURSIVE)
    skill_manager = SkillManager(
        reflector_client, prompt_template=pm.get_skill_manager_prompt(version="3.0")
    )
    skillbook = Skillbook()

    # Fetch domain policy so the reflector can learn policy-aware strategies
    env_constructor = registry.get_env_constructor(args.domain)
    domain_policy = env_constructor().get_policy()

    # Run adaptation with meso-level learning (full conversation trace)
    for epoch in range(1, args.epochs + 1):
        if not quiet:
            print(f"  Epoch {epoch}/{args.epochs}")

        for i, task in enumerate(train_tasks):
            try:
                # Run task with current skillbook using tau2's proper tool-calling agent
                result, simulation = run_single_task_traced(
                    task,
                    skillbook,
                    args,
                    phase="train",
                    trial=0,
                    experiment_name=experiment_name,
                )

                # Extract agent-only conversation trace for meso-level learning
                feedback_level = getattr(args, "feedback_level", "outcome")
                if simulation:
                    trace_str = extract_conversation_trace(
                        simulation,
                        feedback_level=feedback_level,
                    )
                    trace_ctx = TraceContext.from_tau_simulation(simulation.messages)
                else:
                    trace_str = "No trace available"
                    trace_ctx = None

                if feedback_level == "trace":
                    feedback = f"## Agent Domain Policy\n{domain_policy}"
                else:
                    outcome = "SUCCEEDED" if result["success"] else "FAILED"
                    feedback = f"Task {outcome}. Reward: {result['reward']:.2f}, Steps: {result['steps']}\n\n## Agent Domain Policy\n{domain_policy}"

                # Trace goes in reasoning (Model Reasoning), outcome+policy in feedback (Environment Feedback)
                agent_output = AgentOutput(
                    final_answer=(
                        "task completed"
                        if feedback_level == "trace"
                        else f"reward={result['reward']:.2f}"
                    ),
                    reasoning=trace_str,
                    skill_ids=[],
                    trace_context=trace_ctx,
                )

                # Learn from result with full conversation context
                reflection = reflector.reflect(
                    question=task["instruction"],
                    agent_output=agent_output,
                    skillbook=skillbook,
                    ground_truth=None,
                    feedback=feedback,
                )

                skill_manager_output = skill_manager.update_skills(
                    reflection=reflection,
                    skillbook=skillbook,
                    question_context=task["instruction"],
                    progress=f"epoch {epoch}/{args.epochs} · task {i + 1}/{len(train_tasks)}",
                )

                skillbook.apply_update(skill_manager_output.update)

                if not quiet:
                    status = "✓" if result["success"] else "✗"
                    print(
                        f"    [{i + 1}/{len(train_tasks)}] {task['task_id']} {status} (reward={result['reward']:.2f})"
                    )

            except Exception as e:
                if not quiet:
                    print(
                        f"    [{i + 1}/{len(train_tasks)}] {task['task_id']} ERROR: {e}"
                    )
                continue

    if not quiet:
        print(f"  Training complete. Skillbook has {len(skillbook.skills())} skills")

    return skillbook


def run_ace_batch_training(
    train_tasks: List[Dict[str, Any]],
    args: argparse.Namespace,
    quiet: bool = False,
    experiment_name: str = "",
) -> Skillbook:
    """
    Run ACE batch training: execute all tasks first, then reflect on all traces together.

    This defers learning until all tasks complete, then performs a single reflection
    on the combined traces. This enables cross-task pattern recognition.

    Flow:
        Task 1, Task 2, ..., Task N → Reflect on ALL → Single Update
    """
    if not quiet:
        print(f"\n📚 ACE Batch Training ({len(train_tasks)} tasks)")
        print("  Phase 1: Execute all tasks and collect traces...")

    from ace.prompt_manager import PromptManager

    pm = PromptManager()

    # Use reflector-model for Reflector/SkillManager if specified
    reflector_model = getattr(args, "reflector_model", None) or args.model
    reflector_client = create_llm_client(args, model=reflector_model)
    reflector = Reflector(reflector_client, mode=ReflectorMode.RECURSIVE)
    skill_manager = SkillManager(
        reflector_client, prompt_template=pm.get_skill_manager_prompt(version="3.0")
    )
    skillbook = Skillbook()

    # Fetch domain policy so the reflector can learn policy-aware strategies
    env_constructor = registry.get_env_constructor(args.domain)
    domain_policy = env_constructor().get_policy()

    # Phase 1: Execute all tasks and collect traces
    traces: List[Tuple[Dict[str, Any], Dict[str, Any], str, str]] = []
    success_count = 0

    for i, task in enumerate(train_tasks):
        try:
            result, simulation = run_single_task_traced(
                task,
                skillbook,
                args,
                phase="train",
                trial=0,
                experiment_name=experiment_name,
            )

            # Extract trace for batch learning
            feedback_level = getattr(args, "feedback_level", "outcome")
            if simulation:
                trace = extract_conversation_trace(
                    simulation,
                    feedback_level=feedback_level,
                )
                # Truncate trace if needed (configurable via --trace-limit)
                trace_lines = trace.split("\n")
                if len(trace_lines) > args.trace_limit // 10:  # ~10 chars per line
                    half = args.trace_limit // 20
                    trace = "\n".join(
                        trace_lines[:half] + ["..."] + trace_lines[-half:]
                    )
            else:
                trace = f"Error: {result.get('error', 'Unknown')}"

            if feedback_level == "trace":
                feedback = "Task completed"
            else:
                outcome = "SUCCEEDED" if result["success"] else "FAILED"
                feedback = f"Task {outcome}. Reward: {result['reward']:.2f}, Steps: {result['steps']}"

            traces.append((task, result, trace, feedback))

            if result["success"]:
                success_count += 1

            if not quiet:
                status = "✓" if result["success"] else "✗"
                print(
                    f"    [{i + 1}/{len(train_tasks)}] {task['task_id']} {status} (reward={result['reward']:.2f})"
                )

        except Exception as e:
            if not quiet:
                print(f"    [{i + 1}/{len(train_tasks)}] {task['task_id']} ERROR: {e}")
            traces.append(
                (
                    task,
                    {"reward": 0.0, "success": False, "error": str(e)},
                    f"Error: {e}",
                    "Task FAILED",
                )
            )

    if not quiet:
        print(f"\n  Phase 1 complete: {success_count}/{len(traces)} tasks succeeded")
        print("  Phase 2: Batch reflection on all traces...")

    # Phase 2: Combine traces into mega-context
    combined_reasoning = []
    combined_feedback = []

    for i, (task, result, trace, feedback) in enumerate(traces):
        task_header = f"### Task {i + 1}: {task['task_id']}"
        status = "✓ SUCCESS" if result.get("success") else "✗ FAILED"
        combined_reasoning.append(f"{task_header} ({status})\n{trace}")
        combined_feedback.append(f"Task {i + 1} ({task['task_id']}): {feedback}")

    mega_trace = "\n\n---\n\n".join(combined_reasoning)
    mega_feedback = f"## Agent Domain Policy\n{domain_policy}\n\n" + "\n".join(
        combined_feedback
    )

    # Phase 3: Single reflection on all traces
    agent_output = AgentOutput(
        final_answer=f"{success_count}/{len(traces)} tasks succeeded",
        reasoning=mega_trace,
        skill_ids=[],
    )

    if not quiet:
        print(f"    Combined trace size: ~{len(mega_trace)} chars")

    try:
        reflection = reflector.reflect(
            question="Analyze patterns across all training tasks. Look for common failure modes, successful strategies, and cross-task patterns.",
            agent_output=agent_output,
            skillbook=skillbook,
            ground_truth=None,
            feedback=mega_feedback,
        )

        skill_manager_output = skill_manager.update_skills(
            reflection=reflection,
            skillbook=skillbook,
            question_context=f"Batch analysis of {len(traces)} {args.domain} customer service tasks",
            progress="batch learning",
        )

        skillbook.apply_update(skill_manager_output.update)

        if not quiet:
            print(f"  Phase 2 complete. Skillbook has {len(skillbook.skills())} skills")

    except Exception as e:
        if not quiet:
            print(f"  ERROR in batch reflection: {e}")

    return skillbook


def run_evaluation(
    args: argparse.Namespace,
    tasks: List[Dict[str, Any]],
    skillbook: Skillbook,
    phase_name: str = "Evaluation",
    experiment_name: str = "",
) -> Dict[str, Any]:
    """Run pass^k evaluation on tasks."""
    if not args.quiet:
        print(f"\n🧪 {phase_name} Phase (k={args.k})")

    # Run pass^k evaluation using tau2's run_task
    eval_results = evaluate_pass_k(
        tasks=tasks,
        skillbook=skillbook,
        args=args,
        k=args.k,
        quiet=args.quiet,
        phase=phase_name.lower().replace(" ", "_"),
        experiment_name=experiment_name,
    )

    return eval_results


def print_results(
    results: Dict[str, Any],
    title: str,
    args: argparse.Namespace,
) -> None:
    """Print evaluation results summary."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(f"Domain: {args.domain}")
    print(f"Tasks evaluated: {results['tasks_evaluated']}")
    print(f"K value: {results['k']}")
    print()
    print("Pass^k Metrics (TAU-bench formula: C(successes,k)/C(trials,k)):")
    for j in range(1, results["k"] + 1):
        metric = results["metrics"][f"pass_{j}"]
        print(f"  pass^{j}: {metric:.2%}")
    print("=" * 60)


def save_results(
    args: argparse.Namespace,
    results: Dict[str, Any],
    skillbook: Skillbook,
    phase: str,
) -> None:
    """Save evaluation results to files."""
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config_name = getattr(args, "_config_name", args.model)
    base_name = f"tau_{args.domain}_{config_name}_{phase}_{timestamp}"

    # Save summary
    summary_file = output_dir / f"{base_name}_summary.json"
    summary = {
        "benchmark": "tau_bench",
        "domain": args.domain,
        "task_split": args.task_split,
        "model": args.model,
        "user_llm": args.user_llm,
        "phase": phase,
        "timestamp": timestamp,
        "config_profile": getattr(args, "_config_name", None),
        "config_file": getattr(args, "_config_file", None),
        "configuration": {
            "k": args.k,
            "epochs": args.epochs,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "max_steps": args.max_steps,
            "max_errors": args.max_errors,
            "seed": args.seed,
            "batch_reflect": getattr(args, "batch_reflect", False),
            "trace_limit": getattr(args, "trace_limit", 500),
            "skillbook_path": getattr(args, "skillbook", None),
            "reflector_model": getattr(args, "reflector_model", None),
        },
        "results": {
            "tasks_evaluated": results["tasks_evaluated"],
            "pass_sums": results["pass_sums"],
            "metrics": results["metrics"],
        },
        "skillbook_stats": skillbook.stats() if skillbook else {},
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    if not args.quiet:
        print(f"\n💾 Results saved to: {summary_file}")

    # Save detailed results if requested
    if args.save_detailed:
        detailed_file = output_dir / f"{base_name}_detailed.json"
        with open(detailed_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        if not args.quiet:
            print(f"   Detailed results: {detailed_file}")

    # Save skillbook
    if skillbook and len(skillbook.skills()) > 0:
        skillbook_file = output_dir / f"{base_name}_skillbook.json"
        skillbook.save_to_file(str(skillbook_file))

        if not args.quiet:
            print(f"   Skillbook: {skillbook_file}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Validate --skillbook flag
    if args.skillbook:
        skillbook_path = Path(args.skillbook)
        if not skillbook_path.exists():
            print(f"Error: Skillbook file not found: {args.skillbook}")
            sys.exit(1)
        if args.skip_ace:
            print(
                "Warning: --skip-ace is redundant with --skillbook (training is already skipped)"
            )

    # Validate --playbook flag
    playbook_text: Optional[str] = None
    if getattr(args, "playbook", None):
        playbook_path = Path(args.playbook)
        if not playbook_path.exists():
            print(f"Error: Playbook file not found: {args.playbook}")
            sys.exit(1)
        playbook_text = playbook_path.read_text()

    # Set up Opik tracing (registers OpikLogger on litellm.callbacks)
    opik_integration = setup_opik_tracing(
        args.domain, args.model, getattr(args, "opik_project", None)
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    experiment_name = f"tau_{args.domain}_{args.model}_{timestamp}"

    if not args.quiet:
        config_label = getattr(args, "_config_name", "default")
        config_file = getattr(args, "_config_file", "")
        print("🚀 TAU-bench Evaluation")
        print(f"   Config: {config_label} ({config_file})")
        print(f"   Domain: {args.domain}")
        print(f"   Model: {args.model}")
        reflector_model = getattr(args, "reflector_model", None) or args.model
        if reflector_model != args.model:
            print(f"   Reflector Model: {reflector_model}")
        print(f"   User LLM: {args.user_llm}")
        print(f"   K: {args.k}, Max steps: {args.max_steps}, Seed: {args.seed}")
        opik_project = getattr(args, "opik_project", None) or "tau-bench"
        opik_status = (
            f"enabled (project: {opik_project})" if opik_integration else "disabled"
        )
        print(f"   Opik tracing: {opik_status}")
        if args.skillbook:
            print(f"   Skillbook: {args.skillbook}")
        if playbook_text:
            print(f"   Playbook: {args.playbook} ({len(playbook_text)} chars)")
        if not args.skip_ace and not args.skillbook and not playbook_text:
            print(f"   ACE epochs: {args.epochs}")
            if args.batch_reflect:
                print("   Learning mode: BATCH (deferred reflection)")

    def _filter_task_ids(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter tasks to only specified IDs if --task-ids is set."""
        if not getattr(args, "task_ids", None):
            return tasks
        ids = set(args.task_ids.split(","))
        filtered = [t for t in tasks if str(t["task_id"]) in ids]
        if not args.quiet:
            print(f"  Filtered to {len(filtered)} tasks by --task-ids: {args.task_ids}")
        return filtered

    if args.skillbook or playbook_text:
        # Pre-trained skillbook or playbook: only need test tasks
        test_tasks = _filter_task_ids(load_tau_tasks(args, split=args.task_split))
        train_tasks = []

        if not test_tasks:
            print("Error: No tasks loaded")
            sys.exit(1)

        if not args.quiet:
            print(
                f"\n📊 Loaded {len(test_tasks)} test tasks (split: {args.task_split})"
            )

    elif args.compare or not args.skip_ace:
        # Load train/test splits from tau2's official splits
        train_tasks = load_tau_tasks(args, split="train")
        test_tasks = _filter_task_ids(load_tau_tasks(args, split="test"))

        if not train_tasks or not test_tasks:
            print("Error: No tasks loaded")
            sys.exit(1)

        if not args.quiet:
            print(
                f"\n📊 Loaded {len(train_tasks)} train + {len(test_tasks)} test tasks"
            )
    else:
        # Baseline only: load tasks from specified split (default: test for official benchmark)
        test_tasks = _filter_task_ids(load_tau_tasks(args, split=args.task_split))
        train_tasks = []

        if not test_tasks:
            print("Error: No tasks loaded")
            sys.exit(1)

        if not args.quiet:
            print(f"\n📊 Loaded {len(test_tasks)} tasks (split: {args.task_split})")

    if playbook_text and args.compare:
        # Compare baseline vs playbook (no training)

        # Run baseline (no injection)
        print("\n" + "=" * 60)
        print("  1️⃣  BASELINE (no playbook)")
        print("=" * 60)
        ACELLMAgent.set_playbook_text(None)
        baseline_skillbook = Skillbook()
        baseline_results = run_evaluation(
            args,
            test_tasks,
            baseline_skillbook,
            "Baseline",
            experiment_name=experiment_name,
        )
        print_results(baseline_results, "BASELINE Results", args)

        # Run enhanced (with playbook injected)
        print("\n" + "=" * 60)
        print("  2️⃣  ENHANCED (with playbook)")
        print("=" * 60)
        ACELLMAgent.set_playbook_text(playbook_text)
        enhanced_results = run_evaluation(
            args,
            test_tasks,
            baseline_skillbook,
            "Enhanced",
            experiment_name=experiment_name,
        )
        ACELLMAgent.set_playbook_text(None)
        print_results(enhanced_results, "ENHANCED Results", args)

        # Compare
        print("\n" + "=" * 60)
        print("  📊 COMPARISON")
        print("=" * 60)
        for j in range(1, args.k + 1):
            baseline_metric = baseline_results["metrics"][f"pass_{j}"]
            enhanced_metric = enhanced_results["metrics"][f"pass_{j}"]
            diff = enhanced_metric - baseline_metric
            indicator = "✅" if diff > 0 else ("⚠️" if diff < 0 else "➖")
            print(
                f"  pass^{j}: {baseline_metric:.2%} → {enhanced_metric:.2%} ({diff:+.2%}) {indicator}"
            )
        print("=" * 60)

        # Save both results
        save_results(args, baseline_results, baseline_skillbook, "baseline")
        save_results(args, enhanced_results, baseline_skillbook, "ace")

    elif playbook_text:
        # Evaluate with playbook only (no comparison)
        ACELLMAgent.set_playbook_text(playbook_text)
        enhanced_skillbook = Skillbook()
        results = run_evaluation(
            args,
            test_tasks,
            enhanced_skillbook,
            "Enhanced",
            experiment_name=experiment_name,
        )
        ACELLMAgent.set_playbook_text(None)
        print_results(results, "ENHANCED Results (playbook)", args)
        save_results(args, results, enhanced_skillbook, "ace")

    elif args.skillbook and args.compare:
        # Compare baseline vs pre-trained skillbook (no training)
        loaded_skillbook = Skillbook.load_from_file(args.skillbook)
        if not args.quiet:
            print(f"  Loaded skillbook: {len(loaded_skillbook.skills())} skills")

        # Run baseline (empty skillbook)
        print("\n" + "=" * 60)
        print("  1️⃣  BASELINE (no skillbook)")
        print("=" * 60)
        baseline_skillbook = Skillbook()
        baseline_results = run_evaluation(
            args,
            test_tasks,
            baseline_skillbook,
            "Baseline",
            experiment_name=experiment_name,
        )
        print_results(baseline_results, "BASELINE Results", args)

        # Run enhanced (loaded skillbook, no training)
        print("\n" + "=" * 60)
        print("  2️⃣  ENHANCED (pre-trained skillbook)")
        print("=" * 60)
        enhanced_results = run_evaluation(
            args,
            test_tasks,
            loaded_skillbook,
            "Enhanced",
            experiment_name=experiment_name,
        )
        print_results(enhanced_results, "ENHANCED Results", args)

        # Compare
        print("\n" + "=" * 60)
        print("  📊 COMPARISON")
        print("=" * 60)
        for j in range(1, args.k + 1):
            baseline_metric = baseline_results["metrics"][f"pass_{j}"]
            enhanced_metric = enhanced_results["metrics"][f"pass_{j}"]
            diff = enhanced_metric - baseline_metric
            indicator = "✅" if diff > 0 else ("⚠️" if diff < 0 else "➖")
            print(
                f"  pass^{j}: {baseline_metric:.2%} → {enhanced_metric:.2%} ({diff:+.2%}) {indicator}"
            )

        print(f"\n  Skills in skillbook: {len(loaded_skillbook.skills())}")
        print("=" * 60)

        # Save both results
        save_results(args, baseline_results, baseline_skillbook, "baseline")
        save_results(args, enhanced_results, loaded_skillbook, "ace")

    elif args.skillbook:
        # Evaluate with pre-trained skillbook only (no baseline, no training)
        loaded_skillbook = Skillbook.load_from_file(args.skillbook)
        if not args.quiet:
            print(f"  Loaded skillbook: {len(loaded_skillbook.skills())} skills")

        results = run_evaluation(
            args,
            test_tasks,
            loaded_skillbook,
            "Enhanced",
            experiment_name=experiment_name,
        )
        print_results(results, "ENHANCED Results (pre-trained skillbook)", args)
        save_results(args, results, loaded_skillbook, "ace")

    elif args.compare:

        # Run baseline (empty skillbook)
        print("\n" + "=" * 60)
        print("  1️⃣  BASELINE (no ACE)")
        print("=" * 60)
        baseline_skillbook = Skillbook()
        baseline_results = run_evaluation(
            args,
            test_tasks,
            baseline_skillbook,
            "Baseline",
            experiment_name=experiment_name,
        )
        print_results(baseline_results, "BASELINE Results", args)

        # Run ACE training + evaluation
        print("\n" + "=" * 60)
        print("  2️⃣  ACE (with training)")
        print("=" * 60)
        if args.batch_reflect:
            ace_skillbook = run_ace_batch_training(
                train_tasks,
                args,
                args.quiet,
                experiment_name=experiment_name,
            )
        else:
            ace_skillbook = run_ace_training(
                train_tasks,
                args,
                args.quiet,
                experiment_name=experiment_name,
            )
        ace_results = run_evaluation(
            args,
            test_tasks,
            ace_skillbook,
            "ACE Test",
            experiment_name=experiment_name,
        )
        print_results(ace_results, "ACE Results", args)

        # Compare
        print("\n" + "=" * 60)
        print("  📊 COMPARISON")
        print("=" * 60)
        for j in range(1, args.k + 1):
            baseline_metric = baseline_results["metrics"][f"pass_{j}"]
            ace_metric = ace_results["metrics"][f"pass_{j}"]
            diff = ace_metric - baseline_metric
            indicator = "✅" if diff > 0 else ("⚠️" if diff < 0 else "➖")
            print(
                f"  pass^{j}: {baseline_metric:.2%} → {ace_metric:.2%} ({diff:+.2%}) {indicator}"
            )

        print(f"\n  Skills learned: {len(ace_skillbook.skills())}")
        print("=" * 60)

        # Save both results
        save_results(args, baseline_results, baseline_skillbook, "baseline")
        save_results(args, ace_results, ace_skillbook, "ace")

    elif args.skip_ace:
        # Baseline only
        baseline_skillbook = Skillbook()
        results = run_evaluation(
            args,
            test_tasks,
            baseline_skillbook,
            "Baseline",
            experiment_name=experiment_name,
        )
        print_results(results, "BASELINE Results", args)
        save_results(args, results, baseline_skillbook, "baseline")

    else:
        # ACE training + evaluation (train/test already loaded above)
        if args.batch_reflect:
            skillbook = run_ace_batch_training(
                train_tasks,
                args,
                args.quiet,
                experiment_name=experiment_name,
            )
        else:
            skillbook = run_ace_training(
                train_tasks,
                args,
                args.quiet,
                experiment_name=experiment_name,
            )

        # Evaluate on test set (frozen skillbook)
        results = run_evaluation(
            args,
            test_tasks,
            skillbook,
            "Test",
            experiment_name=experiment_name,
        )
        print_results(results, "TAU-bench Results (ACE)", args)
        save_results(args, results, skillbook, "ace")

    if not args.quiet:
        print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
