"""
Benchmark-specific environment implementations.

This module provides specialized evaluation environments for different benchmarks,
each implementing the evaluation logic appropriate for their task type.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Set, Any

from ace import EnvironmentResult

from ace import Sample
from .base import BenchmarkConfig, BenchmarkEnvironment, BenchmarkSample


class GenericBenchmarkEnvironment(BenchmarkEnvironment):
    """
    Generic benchmark environment for basic evaluation tasks.

    Provides standard evaluation metrics like exact match, accuracy, and F1 score.
    Can be used for most text-based benchmarks with straightforward evaluation.
    """

    def evaluate(self, sample: Sample, agent_output) -> EnvironmentResult:
        """Evaluate agent output using configured metrics."""
        prediction = agent_output.final_answer or ""
        ground_truth = sample.ground_truth or ""

        # Compute metrics based on configuration
        metrics = self._compute_metrics(prediction, ground_truth)

        # Generate feedback based on primary metric
        primary_metric = (
            self.config.metrics[0]["name"] if self.config.metrics else "accuracy"
        )
        score = metrics.get(primary_metric, 0.0)

        if score >= 0.8:
            feedback = f"Good performance ({score:.1%}). Answer aligns well with expected output."
        elif score >= 0.5:
            feedback = f"Moderate performance ({score:.1%}). Consider refining approach for better accuracy."
        else:
            feedback = f"Low performance ({score:.1%}). Significant improvement needed in reasoning or format."

        return EnvironmentResult(
            feedback=feedback, ground_truth=ground_truth, metrics=metrics
        )


class FiNEREnvironment(BenchmarkEnvironment):
    """
    Environment for FiNER (Financial Named Entity Recognition) benchmark.

    Evaluates NER predictions against gold labels with support for both
    token-level and entity-level evaluation metrics.
    """

    def evaluate(self, sample: Sample, agent_output) -> EnvironmentResult:
        """Evaluate NER predictions with entity-level metrics."""
        prediction = agent_output.final_answer or ""

        # Extract entities from prediction and ground truth
        predicted_entities = self._extract_entities(prediction, sample)
        gold_entities = self._extract_gold_entities(sample)

        # Compute NER-specific metrics
        metrics = self._compute_ner_metrics(predicted_entities, gold_entities)

        # Generate detailed feedback
        feedback = self._generate_ner_feedback(
            predicted_entities, gold_entities, metrics
        )

        return EnvironmentResult(
            feedback=feedback, ground_truth=sample.ground_truth, metrics=metrics
        )

    def _extract_entities(self, prediction: str, sample: Sample) -> Set[tuple]:
        """Extract entities from model prediction."""
        entities = set()

        # Try to parse structured output (JSON or similar)
        try:
            if prediction.strip().startswith("{") or prediction.strip().startswith("["):
                parsed = json.loads(prediction)
                if isinstance(parsed, list):
                    for entity in parsed:
                        if (
                            isinstance(entity, dict)
                            and "text" in entity
                            and "label" in entity
                        ):
                            entities.add((entity["text"], entity["label"]))
                elif isinstance(parsed, dict) and "entities" in parsed:
                    for entity in parsed["entities"]:
                        entities.add((entity["text"], entity["label"]))
        except (json.JSONDecodeError, KeyError):
            pass

        # Fallback: extract from free text using patterns
        if not entities:
            entities = self._extract_entities_from_text(prediction)

        return entities

    def _extract_entities_from_text(self, text: str) -> Set[tuple]:
        """Extract entities from unstructured text using patterns."""
        entities = set()

        # Common patterns for entity mentions
        patterns = [
            r"(?:PERSON|PER):\s*([^,\n]+)",
            r"(?:ORGANIZATION|ORG):\s*([^,\n]+)",
            r"(?:LOCATION|LOC):\s*([^,\n]+)",
            r"(?:FINANCIAL|FIN):\s*([^,\n]+)",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_text = match.group(1).strip()
                entity_type = match.group(0).split(":")[0].strip().upper()
                entities.add((entity_text, entity_type))

        return entities

    def _extract_gold_entities(self, sample: Sample) -> Set[tuple]:
        """Extract gold entities from sample metadata."""
        entities = set()

        # Check if entities are already extracted by processor
        if hasattr(sample, "metadata") and sample.metadata:
            extracted_entities = sample.metadata.get("entities", [])

            if extracted_entities:
                # Use pre-extracted entities from processor
                for entity in extracted_entities:
                    entities.add((entity["text"], entity["label"]))
                return entities

            # Fallback: parse from BIO labels if available
            tokens = sample.metadata.get("tokens", [])
            bio_labels = sample.metadata.get("bio_labels", [])

            if tokens and bio_labels and len(tokens) == len(bio_labels):
                current_entity = []
                current_label = None

                for token, label in zip(tokens, bio_labels):
                    if label.startswith("B-"):  # Beginning of entity
                        if current_entity:
                            entities.add((" ".join(current_entity), current_label))
                        current_entity = [token]
                        current_label = label[2:]  # Remove B- prefix
                    elif label.startswith("I-") and current_label:  # Inside entity
                        current_entity.append(token)
                    else:  # O or end of entity
                        if current_entity:
                            entities.add((" ".join(current_entity), current_label))
                        current_entity = []
                        current_label = None

                # Handle last entity
                if current_entity:
                    entities.add((" ".join(current_entity), current_label))

        return entities

    def _compute_ner_metrics(
        self, predicted: Set[tuple], gold: Set[tuple]
    ) -> Dict[str, float]:
        """Compute NER evaluation metrics."""
        if not gold:
            return {
                "precision": 1.0 if not predicted else 0.0,
                "recall": 1.0,
                "f1": 1.0,
            }

        true_positives = len(predicted & gold)
        predicted_count = len(predicted)
        gold_count = len(gold)

        precision = true_positives / predicted_count if predicted_count > 0 else 0.0
        recall = true_positives / gold_count if gold_count > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "exact_match": float(predicted == gold),
        }

    def _generate_ner_feedback(
        self, predicted: Set[tuple], gold: Set[tuple], metrics: Dict[str, float]
    ) -> str:
        """Generate detailed feedback for NER evaluation."""
        f1_score = metrics["f1"]
        precision = metrics["precision"]
        recall = metrics["recall"]

        feedback_parts = [
            f"F1: {f1_score:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}"
        ]

        if f1_score >= 0.8:
            feedback_parts.append("Excellent entity recognition performance.")
        elif f1_score >= 0.6:
            feedback_parts.append("Good entity recognition with room for improvement.")
        else:
            feedback_parts.append("Entity recognition needs significant improvement.")

        # Specific guidance based on precision/recall balance
        if precision < recall:
            feedback_parts.append(
                "Focus on reducing false positives - be more selective in entity identification."
            )
        elif recall < precision:
            feedback_parts.append(
                "Focus on improving recall - ensure all relevant entities are identified."
            )

        # Identify missed and incorrect entities
        missed = gold - predicted
        incorrect = predicted - gold

        if missed:
            feedback_parts.append(
                f"Missed {len(missed)} entities: {list(missed)[:3]}..."
            )
        if incorrect:
            feedback_parts.append(
                f"Incorrectly identified {len(incorrect)} entities: {list(incorrect)[:3]}..."
            )

        return " ".join(feedback_parts)


class XBRLMathEnvironment(BenchmarkEnvironment):
    """
    Environment for XBRL-Math benchmark (financial reasoning with numerical computation).

    Evaluates numerical reasoning capabilities with XBRL financial data,
    focusing on accuracy of calculations and understanding of financial relationships.
    """

    def evaluate(self, sample: Sample, agent_output) -> EnvironmentResult:
        """Evaluate numerical reasoning for financial calculations."""
        prediction = agent_output.final_answer or ""
        ground_truth = sample.ground_truth or ""

        # Extract numerical answer from prediction
        predicted_number = self._extract_number(prediction)
        ground_truth_number = self._extract_number(ground_truth)

        # Compute numerical accuracy metrics
        metrics = self._compute_numerical_metrics(predicted_number, ground_truth_number)

        # Generate feedback focused on numerical reasoning
        feedback = self._generate_numerical_feedback(
            predicted_number, ground_truth_number, metrics, prediction
        )

        return EnvironmentResult(
            feedback=feedback, ground_truth=ground_truth, metrics=metrics
        )

    def _extract_number(self, text: str) -> float:
        """Extract numerical value from text response."""
        if not text:
            return float("nan")

        # Remove common currency symbols and formatting
        cleaned = re.sub(r"[\$,\s%]", "", text)

        # Look for numerical patterns
        patterns = [
            r"(?:answer|result|equals?|is)[\s:]*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)",
            r"([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)(?:\s*(?:dollars?|USD|\$))?",
            r"(?:^|\s)([+-]?\d+\.?\d*)(?:\s|$)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, cleaned, re.IGNORECASE)
            if matches:
                try:
                    return float(matches[-1])  # Take the last match as likely answer
                except ValueError:
                    continue

        return float("nan")

    def _compute_numerical_metrics(
        self, predicted: float, ground_truth: float
    ) -> Dict[str, float]:
        """Compute numerical accuracy metrics with tolerance."""
        import math

        if math.isnan(predicted) or math.isnan(ground_truth):
            return {
                "exact_match": 0.0,
                "relative_error": float("inf"),
                "within_1_percent": 0.0,
                "within_5_percent": 0.0,
            }

        exact_match = float(abs(predicted - ground_truth) < 1e-6)

        if ground_truth != 0:
            relative_error = abs(predicted - ground_truth) / abs(ground_truth)
        else:
            relative_error = float("inf") if predicted != 0 else 0.0

        within_1_percent = float(relative_error <= 0.01)
        within_5_percent = float(relative_error <= 0.05)

        return {
            "exact_match": exact_match,
            "relative_error": relative_error,
            "within_1_percent": within_1_percent,
            "within_5_percent": within_5_percent,
        }

    def _generate_numerical_feedback(
        self,
        predicted: float,
        ground_truth: float,
        metrics: Dict[str, float],
        full_prediction: str,
    ) -> str:
        """Generate feedback for numerical reasoning performance."""
        import math

        if math.isnan(predicted):
            return (
                "Could not extract numerical answer from response. "
                "Ensure final answer is clearly stated with numerical value."
            )

        if math.isnan(ground_truth):
            return "No ground truth available for comparison."

        rel_error = metrics["relative_error"]

        if metrics["exact_match"]:
            return f"Perfect! Exact numerical match: {predicted}"
        elif metrics["within_1_percent"]:
            return f"Excellent accuracy (within 1%): predicted {predicted}, expected {ground_truth}"
        elif metrics["within_5_percent"]:
            return (
                f"Good accuracy (within 5%): predicted {predicted}, expected {ground_truth}. "
                f"Relative error: {rel_error:.2%}"
            )
        else:
            error_mag = "large" if rel_error > 0.5 else "moderate"
            return (
                f"Numerical error ({error_mag}): predicted {predicted}, expected {ground_truth}. "
                f"Relative error: {rel_error:.2%}. Review calculation steps and XBRL relationships."
            )


class TauBenchEnvironment(BenchmarkEnvironment):
    """
    Environment for TAU-bench (τ²-bench) tool-calling agent evaluation.

    TAU-bench evaluates agents in customer service domains through
    multi-turn conversations with simulated users, measuring task
    completion through database state assertions.

    Key features:
    - Multi-turn conversation handling via tau2 gym interface
    - User simulation with configurable LLM
    - Database state assertions for task evaluation
    - Pass^k consistency metrics
    """

    def __init__(self, config: BenchmarkConfig, user_llm: str = "gpt-4o-mini"):
        """
        Initialize TAU-bench environment.

        Args:
            config: Benchmark configuration
            user_llm: LLM model for user simulation
        """
        super().__init__(config)
        self.user_llm = user_llm
        self.max_steps = config.metadata.get("max_steps", 30) if config.metadata else 30

    def evaluate(self, sample: Sample, agent_output) -> EnvironmentResult:
        """
        Evaluate agent output for TAU-bench task.

        For TAU-bench, the actual evaluation happens during the conversation loop
        in the runner. This method is called with the final conversation result
        stored in the sample metadata.

        Args:
            sample: Sample containing task info and conversation result in metadata
            agent_output: Agent's final output (contains conversation history)

        Returns:
            EnvironmentResult with task_success and assertion metrics
        """
        # Get evaluation results from sample metadata (set by conversation loop)
        metadata: Dict[str, Any] = dict(sample.metadata) if sample.metadata else {}
        task_result: Dict[str, Any] = metadata.get("task_result", {})

        # Extract metrics from tau2 evaluation
        task_success = task_result.get("reward", 0.0)
        db_check_passed = task_result.get("db_check_passed", False)
        nl_assertion_passed = task_result.get("nl_assertion_passed", False)
        steps_taken = task_result.get("steps_taken", 0)

        metrics = {
            "task_success": float(task_success),
            "db_check": 1.0 if db_check_passed else 0.0,
            "nl_assertion": 1.0 if nl_assertion_passed else 0.0,
            "steps_taken": float(steps_taken),
        }

        # Generate feedback for the Reflector
        feedback = self._generate_feedback(task_result, metrics)

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=None,  # TAU-bench uses assertions, not ground truth
            metrics=metrics,
        )

    def _generate_feedback(
        self, task_result: Dict[str, Any], metrics: Dict[str, float]
    ) -> str:
        """Generate detailed feedback for Reflector analysis."""
        parts = []

        if metrics["task_success"] >= 1.0:
            parts.append("Task completed successfully!")
        else:
            parts.append(f"Task incomplete (reward: {metrics['task_success']:.2f}).")

        if not metrics["db_check"]:
            error = task_result.get("db_error", "Unknown database assertion failure")
            parts.append(f"Database check failed: {error}")

        if not metrics["nl_assertion"]:
            error = task_result.get("nl_error", "Natural language assertion failed")
            parts.append(f"NL assertion failed: {error}")

        steps = int(metrics["steps_taken"])
        max_steps = task_result.get("max_steps", self.max_steps)
        if steps >= max_steps:
            parts.append(f"Reached maximum steps ({max_steps}).")
        else:
            parts.append(f"Completed in {steps} steps.")

        # Add conversation summary if available
        conversation_summary = task_result.get("conversation_summary", "")
        if conversation_summary:
            parts.append(f"Conversation: {conversation_summary}")

        return " ".join(parts)

    def run_conversation(
        self,
        task_data: Dict[str, Any],
        agent_generate_fn,
        skillbook,
    ) -> Dict[str, Any]:
        """
        Run a multi-turn conversation for a TAU-bench task.

        This method handles the conversation loop between the agent
        and the simulated user via tau2's gym interface.

        Args:
            task_data: Task data from Tau2Loader
            agent_generate_fn: Function to generate agent responses
            skillbook: Current skillbook for context injection

        Returns:
            Dict with conversation result and evaluation metrics
        """
        try:
            from tau2.gym import AgentGymEnv
        except ImportError:
            raise ImportError(
                "tau2-bench is required for TAU-bench environment. "
                "Install with: pip install ace-framework[tau-bench]"
            )

        domain = task_data.get("domain", "airline")
        task_id = task_data["task_id"]
        user_llm = task_data.get("user_llm", self.user_llm)

        # Create gym environment for this task
        env = AgentGymEnv(
            domain=domain,
            task_id=task_id,
            user_llm=user_llm,
            max_steps=self.max_steps,
        )

        # Reset environment to get initial observation
        # Returns: (observation: str, info: dict)
        observation, info = env.reset()
        conversation_history = []
        steps = 0
        done = False
        reward = 0.0

        while not done and steps < self.max_steps:
            # Build context from observation and history
            context = self._build_context(observation, conversation_history, task_data)

            # Generate agent response
            agent_output = agent_generate_fn(
                question=observation,
                context=context,
                skillbook=skillbook,
            )

            # Extract action from agent output (agent's response text)
            action = agent_output.final_answer

            # Step the environment
            # Returns: (observation: str, reward: float, terminated: bool, truncated: bool, info: dict)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record conversation turn
            conversation_history.append(
                {
                    "step": steps,
                    "user_message": observation,
                    "agent_response": action,
                }
            )

            steps += 1

        # Extract evaluation info
        return {
            "reward": reward,
            "db_check_passed": info.get("db_check_passed", reward >= 1.0),
            "nl_assertion_passed": info.get("nl_assertion_passed", True),
            "db_error": info.get("db_error", ""),
            "nl_error": info.get("nl_error", ""),
            "steps_taken": steps,
            "max_steps": self.max_steps,
            "done": done,
            "terminated": terminated if "terminated" in dir() else done,
            "truncated": truncated if "truncated" in dir() else False,
            "conversation_history": conversation_history,
            "conversation_summary": self._summarize_conversation(conversation_history),
        }

    def _build_context(
        self,
        observation: str,
        conversation_history: list,
        task_data: Dict[str, Any],
    ) -> str:
        """Build context string for agent generation."""
        parts = []

        # Add domain and task info
        domain = task_data.get("domain", "unknown")
        parts.append(f"Domain: {domain}")
        parts.append(f"Task: {task_data.get('instruction', '')}")

        # Add available tools
        tools = task_data.get("tools", [])
        if tools:
            tool_names = [
                t.get("name", str(t)) if isinstance(t, dict) else str(t)
                for t in tools[:10]
            ]
            parts.append(f"Available tools: {', '.join(tool_names)}")

        # Add conversation history (last 5 turns)
        if conversation_history:
            recent = conversation_history[-5:]
            history_lines = []
            for t in recent:
                user_msg = str(t.get("user_message", ""))[:100]
                agent_msg = str(t.get("agent_response", ""))[:100]
                history_lines.append(
                    f"Turn {t['step']}: User: {user_msg}... Agent: {agent_msg}..."
                )
            history_str = "\n".join(history_lines)
            parts.append(f"Recent conversation:\n{history_str}")

        return "\n\n".join(parts)

    def _summarize_conversation(self, history: list) -> str:
        """Create a brief summary of the conversation."""
        if not history:
            return "No conversation occurred."

        n_turns = len(history)
        if n_turns == 1:
            return f"Single turn conversation."

        return f"{n_turns} turn conversation."


class AppWorldEnvironment(BenchmarkEnvironment):
    """
    Environment for AppWorld benchmark (autonomous agent execution).

    Evaluates agent performance in realistic application environments with
    API interactions, task completion, and execution success metrics.
    """

    def evaluate(self, sample: Sample, agent_output) -> EnvironmentResult:
        """Evaluate agent execution in AppWorld environment."""
        # AppWorld evaluation is typically done through the world.execute() method
        # This environment focuses on analyzing the execution results

        prediction = agent_output.final_answer or ""

        # Extract execution results from sample metadata if available
        execution_results = self._extract_execution_results(sample)

        # Compute execution metrics
        metrics = self._compute_execution_metrics(execution_results, prediction)

        # Generate feedback based on execution success
        feedback = self._generate_execution_feedback(execution_results, metrics)

        return EnvironmentResult(
            feedback=feedback, ground_truth=sample.ground_truth, metrics=metrics
        )

    def _extract_execution_results(self, sample: Sample) -> Dict[str, Any]:
        """Extract execution results from sample metadata."""
        if not sample.metadata:
            return {"success": False, "error": "No execution results available"}

        return sample.metadata.get(
            "execution_results",
            {"success": False, "error": "No execution results in metadata"},
        )

    def _compute_execution_metrics(
        self, execution_results: Dict[str, Any], prediction: str
    ) -> Dict[str, float]:
        """Compute execution success metrics."""
        success = execution_results.get("success", False)

        metrics = {
            "task_success": float(success),
            "execution_error": float(not success),
        }

        # Add API usage metrics if available
        if "api_calls" in execution_results:
            api_calls = execution_results["api_calls"]
            metrics["api_calls_count"] = float(len(api_calls))
            metrics["api_success_rate"] = float(
                sum(1 for call in api_calls if call.get("success", False))
                / len(api_calls)
                if api_calls
                else 0.0
            )

        return metrics

    def _generate_execution_feedback(
        self, execution_results: Dict[str, Any], metrics: Dict[str, float]
    ) -> str:
        """Generate feedback for agent execution performance."""
        if metrics["task_success"]:
            feedback = "Task completed successfully! "

            api_success_rate = metrics.get("api_success_rate", 0.0)
            if api_success_rate >= 0.9:
                feedback += "Excellent API usage with minimal errors."
            elif api_success_rate >= 0.7:
                feedback += "Good API usage with some recoverable errors."
            else:
                feedback += "API usage had issues but task still completed."

        else:
            error = execution_results.get("error", "Unknown error")
            feedback = f"Task failed: {error}. "

            if "timeout" in error.lower():
                feedback += "Consider optimizing execution time and reducing unnecessary API calls."
            elif "api" in error.lower():
                feedback += (
                    "Review API documentation and ensure correct parameter usage."
                )
            else:
                feedback += "Analyze task requirements and improve reasoning approach."

        return feedback
