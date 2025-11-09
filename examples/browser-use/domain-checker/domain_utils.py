#!/usr/bin/env python3
"""
Domain-checker specific utilities.

Functions specific to domain availability checking examples.
"""

from typing import List


def get_test_domains() -> List[str]:
    """
    Get list of test domains to check for availability.

    Returns:
        List of domain names to test
    """
    return [
        "testdomain123456.com",
        "myuniquedomain789.net",
        # "brandnewstartup2024.io",
        # "innovativetech555.org",
        # "creativesolutions999.co",
        # "digitalagency2024.biz",
        # "techstartup123.app",
        # "newcompany456.info",
        # "uniquebusiness789.online",
        # "moderntech2024.dev",
    ]


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


# Domain checker prompt template
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
