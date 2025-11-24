"""
Opik Integration for ACE Framework

Provides enterprise-grade observability and tracing for ACE components.
Replaces custom explainability with production-ready Opik platform.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from dataclasses import asdict

OpikLogger: Optional[type]

try:
    import opik
    from opik import track, opik_context

    OPIK_AVAILABLE = True

    # Try to import LiteLLM Opik integration
    try:
        from litellm.integrations.opik.opik import OpikLogger as OpikLoggerClass

        LITELLM_OPIK_AVAILABLE = True
        OpikLogger = OpikLoggerClass
    except ImportError:
        LITELLM_OPIK_AVAILABLE = False
        OpikLogger = None  # type: ignore[assignment]

except ImportError:
    OPIK_AVAILABLE = False
    LITELLM_OPIK_AVAILABLE = False
    OpikLogger = None  # type: ignore[assignment]

    # Create mock decorators for graceful degradation
    def track(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


logger = logging.getLogger(__name__)


def _should_skip_opik() -> bool:
    """Check if Opik should be disabled via environment variable."""
    return os.environ.get("OPIK_DISABLED", "").lower() in ("true", "1", "yes")


class OpikIntegration:
    """
    Main integration class for ACE + Opik observability.

    Provides enterprise-grade tracing, evaluation, and monitoring
    capabilities for ACE framework components.
    """

    def __init__(
        self,
        project_name: str = "ace-framework",
        enable_auto_config: bool = True,
        tags: Optional[List[str]] = None,
    ):
        """
        Initialize Opik integration.

        Args:
            project_name: Opik project name for organizing traces
            enable_auto_config: Auto-configure Opik if available
            tags: Default tags to apply to all traces
        """
        self.project_name = project_name
        self.tags = tags or ["ace-framework"]
        self.enabled = OPIK_AVAILABLE

        if self.enabled and enable_auto_config:
            try:
                # Configure Opik for local use
                opik.configure(use_local=True)
                logger.info(f"Opik configured locally for project: {project_name}")
            except Exception as e:
                logger.debug(f"Opik configuration skipped: {e}")
                self.enabled = False
        elif not OPIK_AVAILABLE:
            logger.debug(
                "Opik not available. Install with: pip install ace-framework[observability]"
            )

    def log_bullet_evolution(
        self,
        bullet_id: str,
        bullet_content: str,
        helpful_count: int,
        harmful_count: int,
        neutral_count: int,
        section: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log bullet evolution metrics to Opik."""
        if not self.enabled:
            return

        try:
            # Calculate effectiveness score
            total_votes = helpful_count + harmful_count + neutral_count
            effectiveness = helpful_count / total_votes if total_votes > 0 else 0.0

            # Update current trace with bullet metrics
            opik_context.update_current_trace(
                feedback_scores=[
                    {
                        "name": "bullet_effectiveness",
                        "value": effectiveness,
                        "reason": f"Bullet {bullet_id}: {helpful_count}H/{harmful_count}H/{neutral_count}N",
                    }
                ],
                metadata={
                    "bullet_id": bullet_id,
                    "bullet_content": bullet_content,
                    "section": section,
                    "helpful_count": helpful_count,
                    "harmful_count": harmful_count,
                    "neutral_count": neutral_count,
                    "total_votes": total_votes,
                    **(metadata or {}),
                },
                tags=self.tags + ["bullet-evolution"],
            )
        except Exception as e:
            logger.error(f"Failed to log bullet evolution: {e}")

    def log_playbook_update(
        self,
        operation_type: str,
        bullets_added: int = 0,
        bullets_updated: int = 0,
        bullets_removed: int = 0,
        total_bullets: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log playbook update metrics to Opik."""
        if not self.enabled:
            return

        try:
            opik_context.update_current_trace(
                feedback_scores=[
                    {
                        "name": "playbook_size",
                        "value": float(total_bullets),
                        "reason": f"Playbook contains {total_bullets} bullets after {operation_type}",
                    }
                ],
                metadata={
                    "operation_type": operation_type,
                    "bullets_added": bullets_added,
                    "bullets_updated": bullets_updated,
                    "bullets_removed": bullets_removed,
                    "total_bullets": total_bullets,
                    **(metadata or {}),
                },
                tags=self.tags + ["playbook-update"],
            )
        except Exception as e:
            logger.error(f"Failed to log playbook update: {e}")

    def log_role_performance(
        self,
        role_name: str,
        execution_time: float,
        success: bool,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log ACE role performance metrics."""
        if not self.enabled:
            return

        try:
            opik_context.update_current_trace(
                feedback_scores=[
                    {
                        "name": "role_success",
                        "value": 1.0 if success else 0.0,
                        "reason": f"{role_name} {'succeeded' if success else 'failed'} in {execution_time:.2f}s",
                    },
                    {
                        "name": "execution_time",
                        "value": execution_time,
                        "reason": f"{role_name} execution time in seconds",
                    },
                ],
                metadata={
                    "role_name": role_name,
                    "execution_time": execution_time,
                    "success": success,
                    "input_data": input_data,
                    "output_data": output_data,
                    **(metadata or {}),
                },
                tags=self.tags + [f"role-{role_name.lower()}"],
            )
        except Exception as e:
            logger.error(f"Failed to log role performance: {e}")

    def log_adaptation_metrics(
        self,
        epoch: int,
        step: int,
        performance_score: float,
        bullet_count: int,
        successful_predictions: int,
        total_predictions: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log adaptation training metrics."""
        if not self.enabled:
            return

        try:
            accuracy = (
                successful_predictions / total_predictions
                if total_predictions > 0
                else 0.0
            )

            opik_context.update_current_trace(
                feedback_scores=[
                    {
                        "name": "performance_score",
                        "value": performance_score,
                        "reason": f"Epoch {epoch}, Step {step} performance",
                    },
                    {
                        "name": "accuracy",
                        "value": accuracy,
                        "reason": f"Accuracy: {successful_predictions}/{total_predictions}",
                    },
                ],
                metadata={
                    "epoch": epoch,
                    "step": step,
                    "performance_score": performance_score,
                    "bullet_count": bullet_count,
                    "successful_predictions": successful_predictions,
                    "total_predictions": total_predictions,
                    "accuracy": accuracy,
                    **(metadata or {}),
                },
                tags=self.tags + ["adaptation-training"],
            )
        except Exception as e:
            logger.error(f"Failed to log adaptation metrics: {e}")

    def create_experiment(
        self,
        name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create an Opik experiment for evaluation."""
        if not self.enabled:
            return

        try:
            # Opik experiments are automatically created when logging
            # We'll use trace metadata to organize experiments
            opik_context.update_current_trace(
                metadata={
                    "experiment_name": name,
                    "experiment_description": description,
                    "experiment_timestamp": datetime.now().isoformat(),
                    **(metadata or {}),
                },
                tags=self.tags + ["experiment", f"exp-{name}"],
            )
            logger.info(f"Opik experiment created: {name}")
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")

    def setup_litellm_callback(self) -> bool:
        """
        Set up LiteLLM callback for automatic token and cost tracking.

        Returns:
            bool: True if callback was successfully configured, False otherwise
        """
        if not self.enabled or not LITELLM_OPIK_AVAILABLE or OpikLogger is None:
            return False

        try:
            import litellm

            # Initialize OpikLogger
            opik_logger = OpikLogger()  # type: ignore[misc]

            # Add to LiteLLM callbacks if not already present
            if not hasattr(litellm, "callbacks") or litellm.callbacks is None:
                litellm.callbacks = []

            # Check if OpikLogger is already in callbacks
            opik_logger_present = any(
                isinstance(callback, type(opik_logger))
                for callback in litellm.callbacks
            )

            if not opik_logger_present:
                litellm.callbacks.append(opik_logger)
                logger.info(
                    "OpikLogger added to LiteLLM callbacks for automatic token tracking"
                )
                return True
            else:
                logger.debug("OpikLogger already present in LiteLLM callbacks")
                return True

        except Exception as e:
            logger.error(f"Failed to setup LiteLLM callback: {e}")
            return False

    def is_available(self) -> bool:
        """Check if Opik integration is available and configured."""
        return self.enabled

    def is_litellm_integration_available(self) -> bool:
        """Check if LiteLLM Opik integration is available."""
        return LITELLM_OPIK_AVAILABLE


# Global integration instance
_global_integration: Optional[OpikIntegration] = None


def get_integration() -> OpikIntegration:
    """Get or create global Opik integration instance."""
    global _global_integration
    if _global_integration is None:
        if _should_skip_opik():
            # Return disabled integration
            _global_integration = OpikIntegration(enable_auto_config=False)
            _global_integration.enabled = False
        else:
            _global_integration = OpikIntegration()
    return _global_integration


def configure_opik(
    project_name: str = "ace-framework", tags: Optional[List[str]] = None
) -> OpikIntegration:
    """Configure global Opik integration."""
    global _global_integration
    _global_integration = OpikIntegration(project_name=project_name, tags=tags)
    return _global_integration
