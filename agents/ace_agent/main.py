"""
HAL agent wrapper for ACE framework with AppWorld HTTP backend.

This module provides the HAL-compatible interface for running ACE agents
on AppWorld benchmarks. Uses HTTP API to communicate with AppWorld servers
running in a separate environment (to avoid pydantic v1/v2 conflicts).

AppWorld servers must be running:
    - Environment server (default port 8000): appworld serve environment
    - API server (default port 9000): appworld serve apis

Usage with HAL:
    hal-eval --benchmark appworld_test_normal \
        --agent_dir agents/ace_agent \
        --agent_function main.run \
        -A model=gpt-4o \
        -A skillbook_path=skillbook.json
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

# AppWorld server endpoints
APPWORLD_ENV_URL = os.environ.get("APPWORLD_ENV_URL", "http://localhost:8000")
APPWORLD_API_URL = os.environ.get("APPWORLD_API_URL", "http://localhost:9000")


def run(input: Dict[str, Dict[str, Any]], **kwargs) -> Dict[str, str]:
    """
    HAL agent interface for ACE with AppWorld HTTP backend.

    This function is called by HAL harness to run the agent on benchmark tasks.

    Args:
        input: Dictionary mapping task_id to task data
            {"task_id_1": {"prompt": "...", "metadata": {...}}, ...}
        **kwargs: CLI arguments passed via -A flags
            - model: LLM model to use (default: gpt-4o-mini)
            - skillbook_path: Path to skillbook JSON file
            - learning_enabled: Enable online learning (default: False)
            - max_interactions: Maximum execution steps (default: 30)

    Returns:
        Dictionary mapping task_id to solution string
        {"task_id_1": "solution code or answer", ...}
    """
    # Import ACE components (delayed to allow proper environment setup)
    from ace import Agent, Skillbook
    from ace.llm_providers.litellm_client import LiteLLMClient

    # Extract configuration from kwargs
    model = kwargs.get("model", os.environ.get("ACE_MODEL", "gpt-4o-mini"))
    skillbook_path = kwargs.get("skillbook_path")
    learning_enabled = kwargs.get("learning_enabled", "false").lower() == "true"
    max_interactions = int(kwargs.get("max_interactions", "30"))

    logger.info(
        f"ACE Agent: model={model}, skillbook={skillbook_path}, learning={learning_enabled}"
    )

    # Initialize LLM client
    llm = LiteLLMClient(model=model)

    # Initialize or load skillbook
    if skillbook_path and os.path.exists(skillbook_path):
        skillbook = Skillbook.load_from_file(skillbook_path)
        logger.info(f"Loaded skillbook with {len(skillbook.skills())} skills")
    else:
        skillbook = Skillbook()

    # Initialize agent
    agent = Agent(llm)

    # Process each task
    results = {}

    for task_id, task_data in input.items():
        try:
            result = process_task(
                task_id=task_id,
                task_data=task_data,
                agent=agent,
                skillbook=skillbook,
                max_interactions=max_interactions,
            )
            results[task_id] = result
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            results[task_id] = f"Error: {e}"

    # Save updated skillbook if learning was enabled
    if learning_enabled and skillbook_path:
        skillbook.save_to_file(skillbook_path)
        logger.info(f"Saved updated skillbook to {skillbook_path}")

    return results


def process_task(
    task_id: str,
    task_data: Dict[str, Any],
    agent: "Agent",
    skillbook: "Skillbook",
    max_interactions: int = 30,
) -> str:
    """
    Process a single AppWorld task via HTTP API.

    Args:
        task_id: Unique task identifier
        task_data: Task data from HAL benchmark
        agent: ACE Agent instance
        skillbook: Current skillbook
        max_interactions: Maximum interaction steps

    Returns:
        Final solution code
    """
    client = httpx.Client(timeout=120.0)
    last_code = ""

    try:
        # Initialize AppWorld environment via HTTP
        init_response = client.post(
            f"{APPWORLD_ENV_URL}/initialize",
            json={
                "task_id": task_id,
                "experiment_name": "ace_eval",
                "max_interactions": max_interactions,
            },
        )
        init_response.raise_for_status()
        init_result = init_response.json()
        logger.info(f"Initialized AppWorld environment for task {task_id}")

        # Extract task instruction from initialization response
        output_data = init_result.get("output", {})
        instruction = (
            output_data.get("instruction")
            or task_data.get("instruction")
            or task_data.get("prompt")
            or task_data.get("question", "")
        )
        api_docs = task_data.get("api_docs", "")

        # Build execution history for context
        execution_history = []

        # Agent interaction loop
        for step in range(max_interactions):
            # Build prompt with execution history
            history_text = ""
            if execution_history:
                history_text = "\n\n## Previous Attempts:\n"
                for i, (code, output) in enumerate(
                    execution_history[-3:], 1
                ):  # Last 3 attempts
                    history_text += f"\n### Attempt {i}:\n```python\n{code}\n```\nOutput:\n```\n{output}\n```\n"

            prompt = f"""Task: {instruction}

Generate Python code to accomplish this task. The code will be executed in an AppWorld environment.
{history_text}
Generate only the code, no explanation needed."""

            context = ""
            if api_docs:
                context += f"# Available APIs\n{api_docs}\n\n"
            if skillbook.skills():
                context += f"# Learned Strategies\n{skillbook.as_prompt()}"

            # Generate code with ACE agent
            output = agent.generate(
                question=prompt,
                context=context,
                skillbook=skillbook,
            )

            # Extract code from response
            code = extract_code(output.final_answer)
            if not code:
                code = output.final_answer.strip()

            last_code = code
            logger.debug(f"Step {step + 1}: Generated code: {code[:100]}...")

            # Execute code via HTTP
            try:
                exec_response = client.post(
                    f"{APPWORLD_ENV_URL}/execute",
                    json={"task_id": task_id, "code": code},
                )
                exec_response.raise_for_status()
                exec_result = exec_response.json()
                exec_output = exec_result.get("output", "")
            except httpx.HTTPError as e:
                logger.warning(f"Execution error: {e}")
                exec_output = f"Execution error: {e}"

            execution_history.append((code, exec_output))
            logger.debug(f"Step {step + 1}: Execution output: {exec_output[:200]}...")

            # Check if task is completed
            try:
                completed_response = client.post(
                    f"{APPWORLD_ENV_URL}/task_completed",
                    json={"task_id": task_id},
                )
                completed_response.raise_for_status()
                completed_result = completed_response.json()
                completed = completed_result.get("output", False)
                if completed:
                    logger.info(f"Task {task_id} completed at step {step + 1}")
                    break
            except httpx.HTTPError as e:
                logger.warning(f"Error checking completion: {e}")

        return last_code

    finally:
        # Clean up environment
        try:
            client.post(
                f"{APPWORLD_ENV_URL}/close",
                json={"task_id": task_id},
            )
        except httpx.HTTPError:
            pass
        client.close()


def extract_code(response: str) -> Optional[str]:
    """Extract Python code from a response."""
    # Try to find ```python blocks first
    python_match = re.search(r"```python\s*(.*?)\s*```", response, re.DOTALL)
    if python_match:
        return python_match.group(1).strip()

    # Try generic code blocks
    code_match = re.search(r"```\s*(.*?)\s*```", response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    return None


# For direct testing
if __name__ == "__main__":
    import sys

    # Test HTTP connectivity to AppWorld servers
    print(f"Testing connection to AppWorld servers...")
    print(f"  Environment URL: {APPWORLD_ENV_URL}")
    print(f"  API URL: {APPWORLD_API_URL}")

    try:
        client = httpx.Client(timeout=5.0)

        # Test environment server
        try:
            resp = client.get(f"{APPWORLD_ENV_URL}/docs")
            print(f"  Environment server: OK (status {resp.status_code})")
        except httpx.HTTPError as e:
            print(f"  Environment server: FAILED ({e})")
            print("  Start with: appworld serve environment")

        # Test API server
        try:
            resp = client.get(f"{APPWORLD_API_URL}/docs")
            print(f"  API server: OK (status {resp.status_code})")
        except httpx.HTTPError as e:
            print(f"  API server: FAILED ({e})")
            print("  Start with: appworld serve apis")

        client.close()

    except Exception as e:
        print(f"Connection test failed: {e}")
        sys.exit(1)

    print("\nTo run with HAL:")
    print("  hal-eval --benchmark appworld_test_normal \\")
    print("      --agent_dir agents/ace_agent \\")
    print("      --agent_function main.run \\")
    print("      -A model=gpt-4o-mini \\")
    print("      --limit 5")
