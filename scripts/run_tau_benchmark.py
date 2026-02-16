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
from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import SimulationRun
from tau2.metrics.agent_metrics import pass_hat_k
from tau2.registry import registry
from tau2.run import run_task


class ACELLMAgent(LLMAgent):
    """LLMAgent with ACE skillbook injection into the system prompt."""

    # Class-level skillbook (set before each run)
    _skillbook: Optional[Skillbook] = None
    _playbook_text: Optional[str] = None
    _last_system_prompt: Optional[str] = None

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
            prompt = (
                base_prompt
                + f"\n\n<learned_strategies>\n{self._playbook_text}\n</learned_strategies>\n"
            )
        elif self._skillbook and len(self._skillbook.skills()) > 0:
            from ace.prompts_v3 import wrap_skillbook_for_external_agent

            wrapped = wrap_skillbook_for_external_agent(self._skillbook)
            prompt = (
                base_prompt
                + f"\n\n<learned_strategies>\n{wrapped}\n</learned_strategies>\n"
            )
        else:
            prompt = base_prompt

        ACELLMAgent._last_system_prompt = prompt
        return prompt


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
    if not path.exists():
        # Try as a config name in the config directory
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
        "--label",
        type=str,
        default=None,
        help="Custom label for output filenames (overrides config name in file paths)",
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
    parser.add_argument(
        "--reflector-prompts",
        choices=["base", "v1", "v2", "v3", "v4", "v5"],
        default=None,
        help="Recursive reflector prompt version (default: base)",
    )
    parser.add_argument(
        "--sweep-prompts",
        action="store_true",
        help="Sweep all prompt versions: train + evaluate each, compare against shared baseline",
    )
    parser.add_argument(
        "--capture-reflector-inputs",
        type=str,
        default=None,
        metavar="DIR",
        help="Run train tasks and save reflector inputs per task to DIR (no reflection/learning)",
    )
    parser.add_argument(
        "--replay-reflector-inputs",
        type=str,
        default=None,
        metavar="DIR",
        help="Replay captured reflector inputs from DIR through all prompt versions to train skillbooks",
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
        "reflector_prompts": "reflector_prompts",
    }

    for cfg_key, value in flat.items():
        if cfg_key.startswith("_"):
            continue
        attr = key_map.get(cfg_key, cfg_key)
        # Only apply config value if CLI didn't set it (CLI value is None)
        if hasattr(args, attr) and getattr(args, attr) is None:
            setattr(args, attr, value)

    # Store config metadata for banner
    args._config_name = args.label or args.config
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


def _enrich_trace_with_feedback(
    trace_str: str, simulation: SimulationRun, feedback_level: str
) -> str:
    """Append feedback-level enrichments (assertions, action checks) to a trace string.

    This is separated from trace construction so that TraceContext.to_markdown()
    produces the base trace and enrichment is applied on top.
    """
    if feedback_level != "full":
        return trace_str

    lines = [trace_str]

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

    if (
        simulation.termination_reason
        and simulation.termination_reason.value != "success"
    ):
        lines.append(f"\n## Termination: {simulation.termination_reason.value}")

    return "\n".join(lines)


def _extract_last_agent_message(simulation: SimulationRun) -> str:
    """Extract the last substantive agent message from a simulation.

    Returns the last text message the agent sent to the user (not a tool call),
    or a summary of the last tool call if no text message exists.
    """
    for msg in reversed(simulation.messages):
        if isinstance(msg, AssistantMessage):
            if msg.content:
                return msg.content[:500] if len(msg.content) > 500 else msg.content
            if msg.tool_calls:
                tc = msg.tool_calls[-1]
                return f"[TOOL] {tc.name}(...)"
    return "No agent response"


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

    # Resolve recursive reflector prompt template
    reflector_kwargs: Dict[str, Any] = dict(mode=ReflectorMode.RECURSIVE)
    prompt_version = getattr(args, "reflector_prompts", None)
    if prompt_version:
        from ace.reflector.prompt_registry import get_prompt_template

        reflector_kwargs["recursive_prompt_template"] = get_prompt_template(
            prompt_version
        )

    reflector = Reflector(reflector_client, **reflector_kwargs)
    skill_manager = SkillManager(
        reflector_client, prompt_template=pm.get_skill_manager_prompt(version="3.0")
    )
    skillbook = Skillbook()

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
                    trace_ctx = TraceContext.from_tau_simulation(
                        simulation.messages,
                        system_prompt=ACELLMAgent._last_system_prompt or "",
                    )
                    trace_str = trace_ctx.to_markdown()
                    # Append feedback-level enrichments
                    trace_str = _enrich_trace_with_feedback(
                        trace_str, simulation, feedback_level
                    )
                else:
                    trace_str = "No trace available"
                    trace_ctx = None

                if feedback_level == "trace":
                    feedback = "Task completed"
                else:
                    outcome = "SUCCEEDED" if result["success"] else "FAILED"
                    feedback = f"Task {outcome}. Reward: {result['reward']:.2f}, Steps: {result['steps']}"

                # Trace goes in reasoning (Model Reasoning), outcome+policy in feedback (Environment Feedback)
                # Use last agent message as final_answer so reflector sees a real prediction
                last_msg = (
                    _extract_last_agent_message(simulation)
                    if simulation
                    else "No agent response"
                )
                agent_output = AgentOutput(
                    final_answer=last_msg,
                    reasoning=trace_str,
                    skill_ids=[],
                    trace_context=trace_ctx,
                )

                # Learn from result with full conversation context
                reflection = reflector.reflect(
                    question="customer service task",
                    agent_output=agent_output,
                    skillbook=skillbook,
                    ground_truth=None,
                    feedback=feedback,
                )

                skill_manager_output = skill_manager.update_skills(
                    reflection=reflection,
                    skillbook=skillbook,
                    question_context="customer service task",
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

    # Resolve recursive reflector prompt template
    reflector_kwargs: Dict[str, Any] = dict(mode=ReflectorMode.RECURSIVE)
    prompt_version = getattr(args, "reflector_prompts", None)
    if prompt_version:
        from ace.reflector.prompt_registry import get_prompt_template

        reflector_kwargs["recursive_prompt_template"] = get_prompt_template(
            prompt_version
        )

    reflector = Reflector(reflector_client, **reflector_kwargs)
    skill_manager = SkillManager(
        reflector_client, prompt_template=pm.get_skill_manager_prompt(version="3.0")
    )
    skillbook = Skillbook()

    # Phase 1: Execute all tasks and collect traces
    # Each entry: (task, result, trace_str, feedback, trace_ctx)
    traces: List[
        Tuple[Dict[str, Any], Dict[str, Any], str, str, Optional[TraceContext]]
    ] = []
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
                trace_ctx = TraceContext.from_tau_simulation(
                    simulation.messages,
                    system_prompt=ACELLMAgent._last_system_prompt or "",
                )
                trace = trace_ctx.to_markdown()
                trace = _enrich_trace_with_feedback(trace, simulation, feedback_level)
                # Truncate trace if needed (configurable via --trace-limit)
                trace_lines = trace.split("\n")
                if len(trace_lines) > args.trace_limit // 10:  # ~10 chars per line
                    half = args.trace_limit // 20
                    trace = "\n".join(
                        trace_lines[:half] + ["..."] + trace_lines[-half:]
                    )
            else:
                trace = f"Error: {result.get('error', 'Unknown')}"
                trace_ctx = None

            if feedback_level == "trace":
                feedback = "Task completed"
            else:
                outcome = "SUCCEEDED" if result["success"] else "FAILED"
                feedback = f"Task {outcome}. Reward: {result['reward']:.2f}, Steps: {result['steps']}"

            traces.append((task, result, trace, feedback, trace_ctx))

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
                    None,
                )
            )

    if not quiet:
        print(f"\n  Phase 1 complete: {success_count}/{len(traces)} tasks succeeded")
        print("  Phase 2: Batch reflection on all traces...")

    # Phase 2: Combine traces into mega-context
    combined_reasoning = []
    combined_feedback = []
    trace_contexts: List[TraceContext] = []

    for i, (task, result, trace, feedback, trace_ctx) in enumerate(traces):
        task_header = f"### Task {i + 1}: {task['task_id']}"
        status = "✓ SUCCESS" if result.get("success") else "✗ FAILED"
        combined_reasoning.append(f"{task_header} ({status})\n{trace}")
        combined_feedback.append(f"Task {i + 1} ({task['task_id']}): {feedback}")
        if trace_ctx is not None:
            trace_contexts.append(trace_ctx)

    mega_trace = "\n\n---\n\n".join(combined_reasoning)
    mega_feedback = "\n".join(combined_feedback)

    # Build combined TraceContext so the recursive reflector gets structured steps
    combined_trace_ctx = (
        TraceContext.combine(trace_contexts) if trace_contexts else None
    )

    # Phase 3: Single reflection on all traces
    agent_output = AgentOutput(
        final_answer=f"{success_count}/{len(traces)} tasks succeeded",
        reasoning=mega_trace,
        skill_ids=[],
        trace_context=combined_trace_ctx,
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


def capture_reflector_inputs(args: argparse.Namespace) -> None:
    """Run train tasks and save reflector inputs per task to disk.

    Executes each task with an empty skillbook (no ACE injection), then
    builds the exact same AgentOutput/feedback/question that run_ace_training()
    would build — but serializes them to JSON instead of calling reflector.reflect().
    """
    output_dir = Path(args.capture_reflector_inputs)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default to train split unless explicitly overridden
    split = args.task_split if args.task_split != "test" else "train"
    tasks = load_tau_tasks(args, split=split)

    # Filter by --task-ids if set
    if getattr(args, "task_ids", None):
        ids = set(args.task_ids.split(","))
        tasks = [t for t in tasks if str(t["task_id"]) in ids]

    if not tasks:
        print("Error: No tasks loaded")
        sys.exit(1)

    if not args.quiet:
        print(f"📸 Capturing reflector inputs for {len(tasks)} tasks (split: {split})")

    skillbook = Skillbook()  # Empty — no ACE injection
    feedback_level = getattr(args, "feedback_level", "outcome")
    saved = 0

    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        if not args.quiet:
            print(f"  [{i + 1}/{len(tasks)}] Task {task_id}", end=" ", flush=True)

        try:
            result, simulation = run_single_task_traced(
                task, skillbook, args, phase="capture", trial=0,
            )

            # Build trace — identical to run_ace_training() lines 822–857
            if simulation:
                trace_ctx = TraceContext.from_tau_simulation(
                    simulation.messages,
                    system_prompt=ACELLMAgent._last_system_prompt or "",
                )
                trace_str = trace_ctx.to_markdown()
                trace_str = _enrich_trace_with_feedback(
                    trace_str, simulation, feedback_level
                )
            else:
                trace_str = "No trace available"
                trace_ctx = None

            if feedback_level == "trace":
                feedback = "Task completed"
            else:
                outcome = "SUCCEEDED" if result["success"] else "FAILED"
                feedback = f"Task {outcome}. Reward: {result['reward']:.2f}, Steps: {result['steps']}"

            last_msg = (
                _extract_last_agent_message(simulation)
                if simulation
                else "No agent response"
            )
            agent_output = AgentOutput(
                final_answer=last_msg,
                reasoning=trace_str,
                skill_ids=[],
                trace_context=trace_ctx,
            )

            # Serialize per-task JSON — exactly what reflector.reflect() receives
            record = {
                "question": "customer service task",
                "ground_truth": None,
                "feedback": feedback,
                "agent_output": {
                    "final_answer": agent_output.final_answer,
                    "reasoning": agent_output.reasoning,
                    "skill_ids": agent_output.skill_ids,
                },
                "skillbook": "(empty skillbook)",
            }

            task_file = output_dir / f"task_{task_id}.json"
            with open(task_file, "w") as f:
                json.dump(record, f, indent=2, default=str)

            saved += 1
            if not args.quiet:
                status = "✓" if result["success"] else "✗"
                print(f"{status} (reward={result['reward']:.2f})")

        except Exception as e:
            if not args.quiet:
                print(f"ERROR: {e}")

    if not args.quiet:
        print(f"\n✅ Saved {saved}/{len(tasks)} reflector input files to {output_dir}")


def replay_reflector_inputs(args: argparse.Namespace) -> None:
    """Replay captured reflector inputs through all prompt versions to train skillbooks.

    Loads task_*.json files from the given directory, then for each prompt version:
    creates a fresh Skillbook, Reflector (recursive mode), and SkillManager, iterates
    through all inputs (reflect -> update_skills -> apply_update), and saves the
    trained skillbook.
    """
    from ace.prompt_manager import PromptManager
    from ace.reflector.prompt_registry import ALL_PROMPT_VERSION_NAMES, get_prompt_template

    input_dir = Path(args.replay_reflector_inputs)
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        sys.exit(1)

    # Load all task files, sorted by task ID for deterministic order
    task_files = sorted(
        input_dir.glob("task_*.json"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    if not task_files:
        print(f"Error: No task_*.json files found in {input_dir}")
        sys.exit(1)

    # Load all inputs
    inputs = []
    for tf in task_files:
        with open(tf) as f:
            inputs.append(json.load(f))

    print(f"\n🔄 Replaying {len(inputs)} reflector inputs through {len(ALL_PROMPT_VERSION_NAMES)} prompt versions")

    reflector_model = getattr(args, "reflector_model", None) or args.model
    reflector_client = create_llm_client(args, model=reflector_model)
    pm = PromptManager()

    output_base = input_dir / "training_recursive_sequential"
    output_base.mkdir(parents=True, exist_ok=True)

    summary_table: Dict[str, int] = {}

    for version in ALL_PROMPT_VERSION_NAMES:
        print(f"\n{'=' * 60}")
        print(f"  Prompt version: {version}")
        print(f"{'=' * 60}")

        skillbook = Skillbook()
        reflector = Reflector(
            reflector_client,
            mode=ReflectorMode.RECURSIVE,
            recursive_prompt_template=get_prompt_template(version),
        )
        skill_manager = SkillManager(
            reflector_client,
            prompt_template=pm.get_skill_manager_prompt(version="3.0"),
        )

        for i, record in enumerate(inputs):
            task_file = task_files[i]
            task_id = task_file.stem  # e.g. "task_0"

            try:
                # Reconstruct AgentOutput from JSON (no trace_context — auto-built from reasoning)
                ao_data = record["agent_output"]
                agent_output = AgentOutput(
                    final_answer=ao_data["final_answer"],
                    reasoning=ao_data["reasoning"],
                    skill_ids=ao_data.get("skill_ids", []),
                )

                reflection = reflector.reflect(
                    question=record["question"],
                    agent_output=agent_output,
                    skillbook=skillbook,
                    ground_truth=record.get("ground_truth"),
                    feedback=record["feedback"],
                )

                skill_manager_output = skill_manager.update_skills(
                    reflection=reflection,
                    skillbook=skillbook,
                    question_context=record["question"],
                    progress=f"{version} · {i + 1}/{len(inputs)}",
                )

                skillbook.apply_update(skill_manager_output.update)

                print(f"    [{i + 1}/{len(inputs)}] {task_id} ✓ ({len(skillbook.skills())} skills)")

            except Exception as e:
                print(f"    [{i + 1}/{len(inputs)}] {task_id} ERROR: {e}")
                continue

        # Save trained skillbook
        version_dir = output_base / version
        version_dir.mkdir(parents=True, exist_ok=True)
        skillbook_path = version_dir / "skillbook.json"
        skillbook.save_to_file(str(skillbook_path))

        n_skills = len(skillbook.skills())
        summary_table[version] = n_skills
        print(f"  Saved {n_skills} skills to {skillbook_path}")

    # Print summary table
    print(f"\n{'=' * 40}")
    print("  REPLAY SUMMARY")
    print(f"{'=' * 40}")
    print(f"  {'Version':<10} {'Skills':>8}")
    print(f"  {'-' * 20}")
    for version in ALL_PROMPT_VERSION_NAMES:
        print(f"  {version:<10} {summary_table[version]:>8}")
    print(f"{'=' * 40}")
    print(f"\n✅ All skillbooks saved to {output_base}/")


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
    print(f"  Model:      {args.model}")
    print(f"  User LLM:   {args.user_llm}")
    print(f"  Domain:     {args.domain} ({results['tasks_evaluated']} tasks)")
    print(
        f"  Config:     k={results['k']}, seed={args.seed}, max_steps={args.max_steps}"
    )
    print()
    for j in range(1, results["k"] + 1):
        metric = results["metrics"][f"pass_{j}"]
        print(f"  pass^{j}:  {metric:.1%}")
    print("=" * 60)


def _print_comparison(
    baseline_results: Dict[str, Any],
    enhanced_results: Dict[str, Any],
    k: int,
) -> None:
    """Print side-by-side comparison of baseline vs enhanced metrics."""
    print("\n" + "=" * 60)
    print("  COMPARISON")
    print("=" * 60)
    for j in range(1, k + 1):
        b = baseline_results["metrics"][f"pass_{j}"]
        e = enhanced_results["metrics"][f"pass_{j}"]
        diff = e - b
        indicator = "+" if diff > 0 else ("-" if diff < 0 else "=")
        print(f"  pass^{j}:  {b:.1%} -> {e:.1%}  ({diff:+.1%}) {indicator}")
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
            "reflector_prompts": getattr(args, "reflector_prompts", None),
            "feedback_level": getattr(args, "feedback_level", "outcome"),
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


def _run_prompt_sweep(args: argparse.Namespace) -> None:
    """Sweep all prompt versions: shared baseline, then train+evaluate each.

    1. Run baseline evaluation (no ACE) once.
    2. For each prompt version (base, v2, v3, v4, v5):
       - Train skillbook using that prompt version's reflector
       - Evaluate on test split with the trained skillbook
    3. Print comparison table and save combined results.
    """
    from ace.reflector.prompt_registry import ALL_PROMPT_VERSION_NAMES
    import copy

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    experiment_name = f"tau_{args.domain}_{args.model}_sweep_{timestamp}"

    # Load tasks
    train_tasks = load_tau_tasks(args, split="train")
    test_tasks = load_tau_tasks(args, split="test")

    # Filter by --task-ids if set
    if getattr(args, "task_ids", None):
        ids = set(args.task_ids.split(","))
        test_tasks = [t for t in test_tasks if str(t["task_id"]) in ids]

    if not train_tasks or not test_tasks:
        print("Error: No tasks loaded")
        sys.exit(1)

    print(f"\n📊 Loaded {len(train_tasks)} train + {len(test_tasks)} test tasks")
    print(f"\n{'=' * 60}")
    print("  PROMPT SWEEP: base, v2, v3, v4, v5")
    print(f"{'=' * 60}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: Shared baseline ---
    print("\n" + "=" * 60)
    print("  BASELINE (no ACE)")
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
    save_results(args, baseline_results, baseline_skillbook, "baseline")

    # --- Phase 2: Train + evaluate each prompt version ---
    sweep_results: Dict[str, Dict[str, Any]] = {"baseline": baseline_results}
    sweep_skillbooks: Dict[str, Skillbook] = {}
    original_config_name = getattr(args, "_config_name", args.config)

    for version in ALL_PROMPT_VERSION_NAMES:
        print(f"\n{'=' * 60}")
        print(f"  PROMPT VERSION: {version}")
        print(f"{'=' * 60}")

        # Set prompt version for training
        args.reflector_prompts = version
        args._config_name = f"{original_config_name}_rr-{version}"

        # Train
        if getattr(args, "batch_reflect", False):
            ace_skillbook = run_ace_batch_training(
                train_tasks,
                args,
                args.quiet,
                experiment_name=f"{experiment_name}_{version}",
            )
        else:
            ace_skillbook = run_ace_training(
                train_tasks,
                args,
                args.quiet,
                experiment_name=f"{experiment_name}_{version}",
            )

        sweep_skillbooks[version] = ace_skillbook

        # Evaluate with the trained skillbook (frozen)
        version_results = run_evaluation(
            args,
            test_tasks,
            ace_skillbook,
            f"ACE ({version})",
            experiment_name=f"{experiment_name}_{version}",
        )
        sweep_results[version] = version_results
        print_results(version_results, f"ACE Results (rr-{version})", args)
        save_results(args, version_results, ace_skillbook, f"ace_rr-{version}")

    # --- Phase 3: Comparison table ---
    args._config_name = original_config_name
    k = args.k

    print(f"\n{'=' * 70}")
    print("  SWEEP COMPARISON")
    print(f"{'=' * 70}")

    # Header
    header = f"  {'Metric':<10}"
    header += f"{'baseline':>10}"
    for v in ALL_PROMPT_VERSION_NAMES:
        header += f"{'rr-' + v:>10}"
    print(header)
    print("  " + "-" * (10 + 10 * (1 + len(ALL_PROMPT_VERSION_NAMES))))

    # Rows for each pass^j
    for j in range(1, k + 1):
        row = f"  {'pass^' + str(j):<10}"
        b = baseline_results["metrics"][f"pass_{j}"]
        row += f"{b:>9.1%} "
        for v in ALL_PROMPT_VERSION_NAMES:
            val = sweep_results[v]["metrics"][f"pass_{j}"]
            diff = val - b
            sign = "+" if diff > 0 else ""
            row += f"{val:>6.1%}({sign}{diff:.1%})"
        print(row)

    # Skillbook sizes
    row = f"  {'skills':<10}"
    row += f"{'0':>10}"
    for v in ALL_PROMPT_VERSION_NAMES:
        n = len(sweep_skillbooks[v].skills())
        row += f"{n:>10}"
    print(row)
    print(f"{'=' * 70}")

    # Save combined sweep results
    sweep_summary = {
        "benchmark": "tau_bench",
        "sweep_type": "reflector_prompts",
        "domain": args.domain,
        "model": args.model,
        "reflector_model": getattr(args, "reflector_model", None) or args.model,
        "timestamp": timestamp,
        "config_profile": original_config_name,
        "k": k,
        "train_tasks": len(train_tasks),
        "test_tasks": len(test_tasks),
        "versions": {},
    }
    for label, results in sweep_results.items():
        sweep_summary["versions"][label] = {
            "metrics": results["metrics"],
            "tasks_evaluated": results["tasks_evaluated"],
        }
    sweep_file = output_dir / f"tau_{args.domain}_{original_config_name}_sweep_{timestamp}.json"
    with open(sweep_file, "w") as f:
        json.dump(sweep_summary, f, indent=2)
    print(f"\n💾 Sweep results saved to: {sweep_file}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Capture mode: save reflector inputs and exit
    if getattr(args, "capture_reflector_inputs", None):
        setup_opik_tracing(
            args.domain, args.model, getattr(args, "opik_project", None)
        )
        capture_reflector_inputs(args)
        return

    # Replay mode: replay captured inputs through all prompt versions and exit
    if getattr(args, "replay_reflector_inputs", None):
        replay_reflector_inputs(args)
        return

    # Sweep mode: run all prompt versions and compare
    if getattr(args, "sweep_prompts", False):
        # Set up Opik tracing before sweep
        opik_integration = setup_opik_tracing(
            args.domain, args.model, getattr(args, "opik_project", None)
        )
        if not args.quiet:
            config_label = getattr(args, "_config_name", "default")
            config_file = getattr(args, "_config_file", "")
            reflector_model = getattr(args, "reflector_model", None) or args.model
            print("🚀 TAU-bench Prompt Sweep")
            print(f"   Config: {config_label} ({config_file})")
            print(f"   Domain: {args.domain}")
            print(f"   Agent Model: {args.model}")
            if reflector_model != args.model:
                print(f"   Reflector Model: {reflector_model}")
            print(f"   User LLM: {args.user_llm}")
            print(f"   K: {args.k}, Max steps: {args.max_steps}, Seed: {args.seed}")
            print(f"   ACE epochs: {args.epochs}")
        _run_prompt_sweep(args)
        if not args.quiet:
            print("\n✅ Prompt sweep completed successfully!")
        return

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
            rp = getattr(args, "reflector_prompts", None)
            if rp:
                print(f"   Reflector prompts: {rp}")
            if getattr(args, "sweep_prompts", False):
                print("   Sweep mode: ALL prompt versions")

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

        _print_comparison(baseline_results, enhanced_results, args.k)

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

        _print_comparison(baseline_results, enhanced_results, args.k)

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

        _print_comparison(baseline_results, ace_results, args.k)

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
