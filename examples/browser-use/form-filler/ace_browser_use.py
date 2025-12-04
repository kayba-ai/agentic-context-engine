#!/usr/bin/env python3
"""
ACE Browser Agent (WITH ACE)

Browser automation agent with learning capabilities using ACE framework.
Compare this with baseline_browser_use.py to see ACE's value.
"""

import asyncio
import json
from typing import List, Dict
from pathlib import Path
from dotenv import load_dotenv
import argparse

from browser_use import Agent, Browser, ChatOpenAI

from ace import (
    LiteLLMClient,
    Agent as ACEAgent,
    Reflector,
    SkillManager,
    OnlineACE,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    Skillbook,
)
from ace.observability import configure_opik

load_dotenv()

import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler


import os

os.environ["BROWSER_USE_LOGGING_LEVEL"] = "critical"
os.environ["ANONYMIZED_TELEMETRY"] = "false"


def _start_http_server(port: int = 8765) -> str:
    """Start HTTP server in background thread, serving from script's directory."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()

    class QuietHandler(SimpleHTTPRequestHandler):
        def translate_path(self, path):
            """Translate URL path to file system path, relative to script directory."""
            # Remove leading slash and query string
            path = path.split("?", 1)[0].split("#", 1)[0]
            path = path.lstrip("/")
            # Return path relative to script directory
            return str(script_dir / path)

        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Started HTTP server on http://127.0.0.1:{port} (serving from {script_dir})")
    return f"http://127.0.0.1:{port}/form.html"


class BrowserUseEnvironment(TaskEnvironment):
    """Environment that evaluates browser automation performance."""

    def __init__(
        self,
        headless: bool = True,
        model: str = "gpt-4o-mini",
        local_port: int = None,
        run_start_time=None,
    ):
        self.headless = headless
        self.model = model
        self.run_start_time = run_start_time
        if local_port:
            self.form_uri = _start_http_server(local_port)

    def evaluate(self, sample: Sample, agent_output):
        """Run browser automation and evaluate the result."""

        task = sample.context

        print("AGENT OUTPUT: ", agent_output)

        # Extract action plan - handle both dict and string formats
        action_plan = {}
        action_plan_source = agent_output.final_answer

        # If final_answer is already a dict, use it directly
        if isinstance(action_plan_source, dict):
            action_plan = action_plan_source
        # If it's a string, try to parse it as JSON
        elif isinstance(action_plan_source, str):
            try:
                action_plan = json.loads(action_plan_source)
            except json.JSONDecodeError:
                # If parsing fails, try getting from raw
                action_plan_raw = (
                    agent_output.raw.get("final_answer", "{}")
                    if hasattr(agent_output, "raw")
                    else "{}"
                )
                if isinstance(action_plan_raw, dict):
                    action_plan = action_plan_raw
                elif isinstance(action_plan_raw, str):
                    try:
                        action_plan = json.loads(action_plan_raw)
                    except json.JSONDecodeError as e:
                        print("ERROR PARSING ACTION PLAN: ", e)
                        action_plan = {}
        # Fallback: try raw if available
        elif hasattr(agent_output, "raw"):
            action_plan_raw = agent_output.raw.get("final_answer", "{}")
            if isinstance(action_plan_raw, dict):
                action_plan = action_plan_raw
            elif isinstance(action_plan_raw, str):
                try:
                    action_plan = json.loads(action_plan_raw)
                except json.JSONDecodeError as e:
                    print("ERROR PARSING ACTION PLAN: ", e)
                    action_plan = {}

        num_steps = len(action_plan.keys())
        action_plan = "\n".join(
            [
                f"{step_number}: {step_description}"
                for step_number, step_description in action_plan.items()
            ]
        )

        browser_use_prompt = f"""
        {task}

        Follow these steps:
        {action_plan}
        """

        # Run browser automation
        result = asyncio.run(self._run_browser_task(browser_use_prompt))

        try:
            model_outputs = result.model_outputs()
            final_result = result.final_result()
            is_done = result.is_done()
            is_successful = result.is_successful()
            has_errors = result.has_errors()
            number_of_steps = result.number_of_steps()
        except:
            print("ERROR GETTING MODEL OUTPUTS: ", e)
            print("TYPE OF RESULT: ", type(result))
            print("RESULT: ", result)
            model_outputs = None
            final_result = ""
            is_done = False
            is_successful = False
            has_errors = True
            number_of_steps = 0

        model_outputs_text = ""
        for i, model_output in enumerate(model_outputs):
            thinking = (
                model_output.thinking if hasattr(model_output, "thinking") else ""
            )
            next_goal = (
                model_output.next_goal if hasattr(model_output, "next_goal") else ""
            )
            evaluation_previous_goal = (
                model_output.evaluation_previous_goal
                if hasattr(model_output, "evaluation_previous_goal")
                else ""
            )
            model_outputs_text += f"Thoughts about step {i+1}: {thinking}\nGoal for step {i+2}: {next_goal}\nEvaluation Previous Goal: {evaluation_previous_goal}\n\n"

        # Build steps text outside f-string to avoid backslash issue
        done_text = "" if is_done else "not "
        successful_text = "" if is_successful else "not "
        errors_text = "" if has_errors else "no "

        feedback = f"""
        The task was {done_text}finished.
        The task was {successful_text}successful.
        The browser use agent had {errors_text}errors.
        Browser use agent took {number_of_steps} steps.

        Following are the thoughts and goals of the agent while executing the task:
        {model_outputs_text}

        The final result was: {final_result}
        """

        status = "SUCCESS" if is_successful else "ERROR"
        success = is_successful
        efficient = number_of_steps <= 15  # Consider efficient if <= max_steps

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=None,  # No ground truth available for form filling
            metrics={
                "success": success,
                "efficient": efficient,
                "steps": number_of_steps,
                "status": status,
            },
        )

    async def _run_browser_task(self, browser_use_prompt: str):
        """Run browser task without any learning."""

        browser = None
        try:
            # Start browser
            browser = Browser(headless=self.headless)
            await browser.start()

            # Create agent with basic task (no learning, no strategy optimization)
            llm = ChatOpenAI(model=self.model, temperature=0.0)

            agent = Agent(
                task=browser_use_prompt,
                llm=llm,
                browser=browser,
                max_actions_per_step=5,
            )

            # Run with timeout
            history = await asyncio.wait_for(agent.run(max_steps=10), timeout=240.0)
            return history
        except asyncio.TimeoutError:
            # Try to get steps from history if it exists
            number_of_steps = 25  # default to max_steps
            try:
                if "history" in locals() and history is not None:
                    number_of_steps = (
                        history.number_of_steps()
                        if hasattr(history, "number_of_steps")
                        else 25
                    )
            except:
                pass
            return {
                "is_done": False,
                "is_successful": False,
                "has_errors": True,
                "number_of_steps": number_of_steps,
                "final_result": "Timeout",
            }
        except Exception as e:
            # Try to get steps from history if it exists
            number_of_steps = 0
            try:
                if "history" in locals() and history is not None:
                    number_of_steps = (
                        history.number_of_steps()
                        if hasattr(history, "number_of_steps")
                        else 0
                    )
            except:
                pass
            return {
                "is_done": False,
                "is_successful": False,
                "has_errors": True,
                "number_of_steps": number_of_steps,
                "final_result": str(e),
            }

        finally:
            if browser:
                try:
                    await browser.stop()
                except:
                    pass

    def _get_token_usage(self, trace_id: str = None) -> tuple[int, int, int, int]:
        """Query Opik for ACE token usage only.

        Returns:
            tuple: (ace_tokens, agent_tokens, reflector_tokens, skill_manager_tokens)
        """
        try:
            import opik
            import datetime

            # Create client and flush to ensure data is sent
            client = opik.Opik()
            client.flush()

            # Use run start time if available, otherwise fall back to last 10 minutes
            if self.run_start_time:
                recent_time = self.run_start_time.isoformat().replace("+00:00", "Z")
                print(f"   🕐 Searching for traces since run start: {recent_time}")
            else:
                now = datetime.datetime.now(datetime.timezone.utc)
                recent_time = (
                    (now - datetime.timedelta(minutes=10))
                    .isoformat()
                    .replace("+00:00", "Z")
                )
                print(
                    f"   🕐 Searching for traces since: {recent_time} (fallback: last 10 minutes)"
                )

            all_traces = []

            # Only search ACE project for role breakdown
            for project in ["ace-roles"]:
                try:
                    traces = client.search_traces(
                        project_name=project,
                        filter_string=f'start_time >= "{recent_time}"',
                        max_results=50,
                    )
                    print(
                        f"   📊 Found {len(traces)} recent traces in '{project}' project"
                    )
                    all_traces.extend(traces)
                except Exception as e:
                    print(f"   ⚠️ Failed to search '{project}' project: {e}")

            # Track individual ACE role tokens
            agent_tokens = 0
            reflector_tokens = 0
            skill_manager_tokens = 0

            print(f"   🔍 Processing {len(all_traces)} total traces...")

            # Process ACE role traces
            for trace in all_traces:
                trace_name = getattr(trace, "name", "unknown")
                trace_name_lower = trace_name.lower()

                if any(
                    role in trace_name_lower
                    for role in ["agent", "reflector", "skill_manager"]
                ):
                    # Get usage from trace or spans
                    total_tokens = 0

                    if trace.usage:
                        total_tokens = trace.usage.get("total_tokens", 0)
                    else:
                        # Check spans for this trace
                        try:
                            spans = client.search_spans(trace_id=trace.id)
                            for span in spans:
                                if hasattr(span, "usage") and span.usage:
                                    span_tokens = span.usage.get("total_tokens", 0)
                                    total_tokens += span_tokens
                        except Exception as e:
                            print(f"         ⚠️ Failed to get spans: {e}")

                    # Classify by role
                    if "agent" in trace_name_lower:
                        agent_tokens += total_tokens
                    elif "reflector" in trace_name_lower:
                        reflector_tokens += total_tokens
                    elif "skill_manager" in trace_name_lower:
                        skill_manager_tokens += total_tokens

            # Calculate total ACE tokens
            ace_tokens = agent_tokens + reflector_tokens + skill_manager_tokens

            print(f"   📊 Role breakdown:")
            print(f"      🎯 Agent: {agent_tokens} tokens")
            print(f"      🔍 Reflector: {reflector_tokens} tokens")
            print(f"      📝 SkillManager: {skill_manager_tokens} tokens")

            return (ace_tokens, agent_tokens, reflector_tokens, skill_manager_tokens)

        except Exception as e:
            print(f"   Warning: Could not retrieve token usage from Opik: {e}")
            return 0, 0, 0, 0


def main(task_file: str = "task2_form.txt"):
    """Main function - browser automation with ACE learning.

    Args:
        task_file: Path to the task file containing the browser task description.
                  Defaults to "task2_form.txt".
    """

    # Capture start time for trace filtering
    import datetime

    run_start_time = datetime.datetime.now(datetime.timezone.utc)

    # Configure Opik if available
    try:
        configure_opik(project_name="ace-browser-use")
        print("📊 Opik observability enabled")
    except:
        print("📊 Opik not available, continuing without observability")

    print("\n🤖 ACE Browser Agent (WITH ACE)")
    print("✨ Learning enabled - improves after each task")
    print("=" * 40)

    print("\n🔄 Starting browser task with learning...\n")

    # Read task from file
    # - Absolute paths: use as-is
    # - Simple filenames (no path separators): resolve relative to script directory
    # - Relative paths with directories: resolve relative to current working directory (original behavior)
    script_dir = Path(__file__).parent.absolute()
    task_file_path = Path(task_file)

    if task_file_path.is_absolute():
        # Absolute path - use as-is
        pass
    elif not any(sep in task_file for sep in ["/", "\\"]):
        # Simple filename (e.g., "task1_flight_search.txt") - resolve relative to script directory
        task_file_path = script_dir / task_file
    else:
        # Relative path with directories (e.g., "examples/browser-use/task2_form.txt")
        # Resolve relative to current working directory (original behavior)
        task_file_path = Path(task_file).resolve()

    with open(task_file_path, "r") as f:
        task_content = f.read()
    task_content = "Task:\n\n" + task_content

    results = []

    llm = LiteLLMClient(model="gpt-4o-mini", temperature=0.7)

    adapter = OnlineACE(
        skillbook=Skillbook(),
        agent=ACEAgent(llm),
        reflector=Reflector(llm),
        skill_manager=SkillManager(llm),
        max_refinement_rounds=2,
    )

    environment = BrowserUseEnvironment(
        headless=False,
        model="gpt-4o-mini",
        local_port=8765,
        run_start_time=run_start_time,
    )

    question = """
    If you were a browser use agent, what would you do to fullfil the following task?

    How would your step by step action plan look like for this browser use task?
    Single steps should be atomic and self-contained.
    The action plan should be a list of steps.
    Single steps could be things like:
    - Click on a button
    - Fill out a form
    - Navigate to a website
    - Read a value from the screen
    - etc.

    Provide the plan as an overall answer in the final_answer field as a dictionary of steps, where the key is the step number and the value is the step description.
    Example: "final_answer": {
        "1": "Click on the \\"next\\" button",
        "2": "Fill out the Email field",
        "3": "Fill out the Password field",
        "4": "Click on the login button"
    }
    """

    samples = []
    for i in range(5):
        samples.append(
            Sample(question=question, ground_truth="SUCCESS", context=task_content)
        )

    results = adapter.run(samples, environment)

    # Query ACE tokens after all roles have completed
    print(f"\n💰 Querying ACE token usage after all tasks processed...")
    import time

    time.sleep(5)  # Wait for Opik to index final traces
    (
        total_ace_tokens,
        total_agent_tokens,
        total_reflector_tokens,
        total_skill_manager_tokens,
    ) = environment._get_token_usage()

    # Show final results
    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)

    # Extract metrics from adapter results
    if results:
        successful = sum(
            1 for r in results if r.environment_result.metrics.get("success", False)
        )
        total_steps = sum(r.environment_result.metrics.get("steps", 0) for r in results)
        avg_steps = total_steps / len(results) if results else 0

        print(
            f"\n✅ Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)"
        )
        print(f"⚡ Average steps: {avg_steps:.1f}")
        print()
        print(
            f"{'📊 Steps:':<25} {total_steps:>6} total     {avg_steps:>6.1f} per task"
        )
        print(
            f"{'🧠 ACE Tokens:':<25} {total_ace_tokens:>6} total     {total_ace_tokens/len(results) if results else 0:>6.1f} per task"
        )
        print()
        print("🧠 ACE Role Breakdown (Think → Learn):")
        print(
            f"   🎯 Agent:          {total_agent_tokens:>6} tokens  (strategy planning)"
        )
        print(
            f"   🔍 Reflector:      {total_reflector_tokens:>6} tokens  (performance analysis)"
        )
        print(
            f"   📝 SkillManager:   {total_skill_manager_tokens:>6} tokens  (skillbook updates)"
        )
        print(f"   {'─' * 40}")
        print(f"   🧠 Total ACE:      {total_ace_tokens:>6} tokens")
    else:
        print("\n⚠️ No results to display")

    print(f"\n✨ Learning enabled - improves after each task")

    # Show learned strategies
    if adapter.skillbook.skills():
        print(f"\n🎯 Learned Strategies:")
        for i, skill in enumerate(adapter.skillbook.skills(), 1):
            print(f"  {i}. {skill.content}")

    print(f"\n💡 Compare with: python examples/browser-use/baseline_browser_use.py")
    print(f"   Baseline has no learning - same performance every time")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ACE Browser Use Agent")
    parser.add_argument(
        "--task-file", type=str, default="task2_form.txt", help="Path to the task file"
    )
    args = parser.parse_args()
    main(args.task_file)
