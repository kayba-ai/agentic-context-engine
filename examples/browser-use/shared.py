#!/usr/bin/env python3
"""
Shared utilities for browser-use examples.

This module contains utilities shared across all browser-use demos.
Functions are marked as either:
- ✅ USED: Actively used in current examples
- 📝 TEMPLATE: Reference/template for your own code

Generic utilities go here. Example-specific utilities should live in
their respective example folders (e.g., domain_utils.py, form_utils.py).
"""

from typing import Dict, Any, Optional
import json
from pathlib import Path


# ============================================================================
# ✅ USED FUNCTIONS - Actively used in examples
# ============================================================================


def calculate_timeout_steps(timeout_seconds: float) -> int:
    """
    Calculate additional steps for timeout based on 1 step per 12 seconds.

    ✅ USED: domain-checker examples use this for timeout handling.

    Args:
        timeout_seconds: The timeout in seconds

    Returns:
        Number of additional steps to allow

    Example:
        >>> calculate_timeout_steps(180.0)
        15  # 180 seconds / 12 seconds per step
    """
    return int(timeout_seconds // 12)


# ============================================================================
# 📝 TEMPLATE FUNCTIONS - Useful reference for your own code
# ============================================================================


def format_result_output(
    task_name: str,
    success: bool,
    steps: int,
    error: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format a consistent output message for task results.

    📝 TEMPLATE: Not currently used, but useful for formatting results.

    Args:
        task_name: Name of the task completed
        success: Whether the task succeeded
        steps: Number of steps taken
        error: Error message if failed
        additional_info: Any additional information to include

    Returns:
        Formatted output string

    Example:
        >>> output = format_result_output(
        ...     task_name="Check domain availability",
        ...     success=True,
        ...     steps=12,
        ...     additional_info={"domain": "example.com", "status": "AVAILABLE"}
        ... )
        >>> print(output)

        ✅ SUCCESS: Check domain availability
        Steps taken: 12
        domain: example.com
        status: AVAILABLE
    """
    status = "✅ SUCCESS" if success else "❌ FAILED"
    output = f"\n{status}: {task_name}\n"
    output += f"Steps taken: {steps}\n"

    if error:
        output += f"Error: {error}\n"

    if additional_info:
        for key, value in additional_info.items():
            output += f"{key}: {value}\n"

    return output


def save_results_to_file(
    results: Dict[str, Any], filename: str, directory: str = "results"
) -> Path:
    """
    Save task results to a JSON file.

    📝 TEMPLATE: Not currently used, but useful for saving benchmark results.

    Args:
        results: Dictionary of results to save
        filename: Name of the file to save
        directory: Directory to save in (created if doesn't exist)

    Returns:
        Path to the saved file

    Example:
        >>> results = {
        ...     "domains_checked": 10,
        ...     "success_rate": 0.9,
        ...     "avg_steps": 8.5
        ... }
        >>> path = save_results_to_file(results, "domain_check_results.json")
        >>> print(f"Saved to {path}")
        Saved to results/domain_check_results.json
    """
    # Create directory if it doesn't exist
    results_dir = Path(directory)
    results_dir.mkdir(exist_ok=True)

    # Save results
    filepath = results_dir / filename
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return filepath


def get_browser_config(headless: bool = True) -> Dict[str, Any]:
    """
    Get common browser configuration settings.

    📝 TEMPLATE: Reference configuration for browser-use Browser objects.
    Not currently used (examples configure browsers directly), but shows
    recommended settings for browser automation.

    Args:
        headless: Whether to run browser in headless mode

    Returns:
        Dictionary of browser configuration options

    Example:
        >>> config = get_browser_config(headless=True)
        >>> browser = Browser(**config)  # If browser-use supported this
        >>> # Currently, examples do: browser = Browser(headless=True)
    """
    return {
        "headless": headless,
        "viewport": {"width": 1920, "height": 1080},
        "timeout": 30000,  # 30 seconds default timeout
        "wait_for_network_idle": True,
    }


# ============================================================================
# CONSTANTS - Used across examples
# ============================================================================

# Maximum number of retry attempts for browser tasks
MAX_RETRIES = 3

# Default timeout in seconds for browser operations
DEFAULT_TIMEOUT_SECONDS = 180.0

# Browser-use typically takes ~12 seconds per step
STEPS_PER_SECOND = 1 / 12
