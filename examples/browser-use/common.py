#!/usr/bin/env python3
"""
Common utilities for browser-use examples.

Shared functions and constants used by both baseline and ACE-enhanced demos.
This reduces code duplication and makes maintenance easier.
"""

from typing import List, Dict, Any, Optional
import json
from pathlib import Path


def calculate_timeout_steps(timeout_seconds: float) -> int:
    """
    Calculate additional steps for timeout based on 1 step per 12 seconds.

    Args:
        timeout_seconds: The timeout in seconds

    Returns:
        Number of additional steps to allow
    """
    return int(timeout_seconds // 12)


def get_test_domains() -> List[str]:
    """
    Get list of test domains to check for availability.

    Returns:
        List of domain names to test
    """
    return [
        "testdomain123456.com",
        "myuniquedomain789.net",
        "brandnewstartup2024.io",
        "innovativetech555.org",
        "creativesolutions999.co",
        "digitalagency2024.biz",
        "techstartup123.app",
        "newcompany456.info",
        "uniquebusiness789.online",
        "moderntech2024.dev"
    ]


def get_test_form_data() -> Dict[str, Any]:
    """
    Get test data for form filling examples.

    Returns:
        Dictionary containing form field values
    """
    return {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@example.com",
        "phone": "555-123-4567",
        "company": "Test Company Inc.",
        "address": "123 Test Street",
        "city": "Test City",
        "state": "CA",
        "zip": "12345",
        "country": "United States",
        "comments": "This is a test submission using browser automation.",
    }


def format_result_output(
    task_name: str,
    success: bool,
    steps: int,
    error: Optional[str] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format a consistent output message for task results.

    Args:
        task_name: Name of the task completed
        success: Whether the task succeeded
        steps: Number of steps taken
        error: Error message if failed
        additional_info: Any additional information to include

    Returns:
        Formatted output string
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
    results: Dict[str, Any],
    filename: str,
    directory: str = "results"
) -> Path:
    """
    Save task results to a JSON file.

    Args:
        results: Dictionary of results to save
        filename: Name of the file to save
        directory: Directory to save in (created if doesn't exist)

    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    results_dir = Path(directory)
    results_dir.mkdir(exist_ok=True)

    # Save results
    filepath = results_dir / filename
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return filepath


def parse_domain_checker_output(output: str) -> str:
    """
    Parse the output from domain checker to determine status.

    Args:
        output: Raw output from the domain checker

    Returns:
        Status string: "AVAILABLE", "TAKEN", or "ERROR"
    """
    output_upper = output.upper()

    if "AVAILABLE:" in output_upper:
        return "AVAILABLE"
    elif "TAKEN:" in output_upper:
        return "TAKEN"
    else:
        return "ERROR"


def parse_form_filler_output(output: str) -> bool:
    """
    Parse the output from form filler to determine success.

    Args:
        output: Raw output from the form filler

    Returns:
        True if form was successfully submitted, False otherwise
    """
    output_upper = output.upper()

    success_indicators = [
        "SUCCESSFULLY",
        "SUBMITTED",
        "COMPLETE",
        "SUCCESS",
        "DONE"
    ]

    return any(indicator in output_upper for indicator in success_indicators)


def get_browser_config(headless: bool = True) -> Dict[str, Any]:
    """
    Get common browser configuration settings.

    Args:
        headless: Whether to run browser in headless mode

    Returns:
        Dictionary of browser configuration options
    """
    return {
        "headless": headless,
        "viewport": {"width": 1920, "height": 1080},
        "timeout": 30000,  # 30 seconds default timeout
        "wait_for_network_idle": True,
    }


def get_llm_config(model: str = "gpt-4o", temperature: float = 0.0) -> Dict[str, Any]:
    """
    Get common LLM configuration settings.

    Args:
        model: The model to use
        temperature: Temperature setting for the model

    Returns:
        Dictionary of LLM configuration options
    """
    return {
        "model": model,
        "temperature": temperature,
        "max_tokens": 4096,
        "top_p": 1.0,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }


# Constants for retry logic
MAX_RETRIES = 3
DEFAULT_TIMEOUT_SECONDS = 180.0
STEPS_PER_SECOND = 1 / 12  # 1 step per 12 seconds


# Common prompts and templates
DOMAIN_CHECKER_TEMPLATE = """
You are a browser agent. For every step, first think, then act.
Use exactly this format:
Thought: describe what you want to do next
Action: <browser-use-tool with JSON args>
I will reply with Observation: … after each action.
Repeat Thought → Action → Observation until you can answer.
When you are done, write Final: with the result.

Task: Check if the domain "{domain}" is available.

  IMPORTANT: Do NOT navigate to {domain} directly. Instead:
  1. Go to a domain checking website
  2. In the search bar type "{domain}" on that website
  3. Read the availability status from the results

Output format (exactly one of these):
AVAILABLE: {domain}
TAKEN: {domain}
ERROR: <reason>
"""

FORM_FILLER_TEMPLATE = """
You are a form filling agent. Your task is to fill out and submit a web form.

Form Data:
{form_data}

Instructions:
1. Navigate to the form page
2. Fill in all available fields with the provided data
3. Submit the form
4. Confirm successful submission

Output format:
SUCCESS: Form submitted successfully
ERROR: <reason for failure>
"""