#!/usr/bin/env python3
"""
ACE + Browser-Use Domain Checker Demo

Simple demo showing ACE learning to improve at checking domain availability.
Uses OnlineAdapter for incremental learning after each domain check.
"""

import asyncio
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Import browser-use
from ace.prompts_v2_1 import PromptManager
from browser_use import Agent, Browser, ChatAnthropic

# Import common utilities
from common import (
    calculate_timeout_steps,
    parse_domain_checker_output,
    format_result_output,
    save_results_to_file,
    MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS
)


from ace import (
    LiteLLMClient,
    Generator,
    Reflector,
    Curator,
    OnlineAdapter,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    Playbook,
)
from ace.observability import configure_opik

import opik
client = opik.Opik()

class DomainCheckEnvironment(TaskEnvironment):
    """Environment that evaluates domain checking performance."""

    def __init__(self, headless: bool = True, model: str = "claude-sonnet-4-5-20250929", run_start_time = None):
        self.headless = headless
        self.model = model
        self.run_start_time = run_start_time

    def evaluate(self, sample: Sample, generator_output):
        """Run browser automation and evaluate the result."""

        # Extract domain from the sample question
        domain = sample.question.replace("Check if domain ", "").replace(" is available", "")

        # Get strategy from generator
        strategy = generator_output.final_answer

        print(f"🔍 Checking domain: {domain}")

        # Capture current trace ID for token tracking
        trace_id = None
        try:
            from opik import opik_context
            trace_data = opik_context.get_current_trace_data()
            trace_id = trace_data.id if trace_data else None
            print(f"   🆔 Captured trace ID: {trace_id[:8] if trace_id else 'None'}...")

            # If no current trace, we'll use time-based query in _get_token_usage
            if not trace_id:
                print(f"   📝 No current trace context, will use time-based query")
        except Exception as e:
            print(f"   ⚠️ Failed to get trace ID: {e}")
            pass  # Graceful fallback if Opik not available

        # Run browser automation
        result = asyncio.run(self._check_domain(domain, strategy))

        # Get browser-use tokens from result (if available)
        browseruse_tokens = result.get('browseruse_tokens', 0)
        print(f"   📊 Browser tokens: {browseruse_tokens}")

        # Evaluate correctness
        status_success = result['status'] != "ERROR"

        # For testing purposes, assume test domains should be AVAILABLE
        expected_status = "AVAILABLE"
        correct = (result['status'] == expected_status) if status_success else False

        feedback = f"Domain check {'succeeded' if status_success else 'failed'}. "

        if status_success:
            if result['status'] == expected_status:
                feedback += f"Correctly identified domain as {result['status']}. "
            else:
                feedback += f"Incorrectly identified domain as {result['status']} (expected: {expected_status}). "
        else:
            feedback += f"Error: {result.get('error', 'Unknown error')}. "

        # Add detailed execution logs for ACE Reflector to analyze
        execution_logs = result.get('execution_logs', [])
        if execution_logs:
            feedback += f"\n\n=== BROWSER EXECUTION DETAILS ===\n"
            feedback += "\n".join(execution_logs)
            feedback += f"\n=== END EXECUTION DETAILS ===\n"

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=None,  # No ground truth available for domain checking
            metrics={
                "correct": correct,
                "status_success": status_success,
                "steps": result['steps'],
                "total_steps": result.get('total_steps', result['steps']),
                "status": result['status'],
                "expected": expected_status,
                "attempt": result.get('attempt', 1),
                "attempt_details": result.get('attempt_details', []),
                "browseruse_tokens": browseruse_tokens
            }
        )

    def _get_token_usage(self, trace_id: str = None) -> tuple[int, int, int, int]:
        """Query Opik for ACE token usage only.

        Returns:
            tuple: (ace_tokens, generator_tokens, reflector_tokens, curator_tokens)
        """
        try:
            import opik
            import datetime

            # Create client and flush to ensure data is sent
            client = opik.Opik()
            client.flush()

            # Based on Claude research: Use search_traces() instead of search_spans()
            print(f"   📋 Using search_traces() method as recommended by Claude research...")

            # Use run start time if available, otherwise fall back to last 10 minutes
            if self.run_start_time:
                recent_time = self.run_start_time.isoformat().replace('+00:00', 'Z')
                print(f"   🕐 Searching for traces since run start: {recent_time}")
            else:
                now = datetime.datetime.now(datetime.timezone.utc)
                recent_time = (now - datetime.timedelta(minutes=10)).isoformat().replace('+00:00', 'Z')
                print(f"   🕐 Searching for traces since: {recent_time} (fallback: last 10 minutes)")

            all_traces = []

            # Only search ACE project for role breakdown
            for project in ["ace-roles"]:
                try:
                    traces = client.search_traces(
                        project_name=project,
                        filter_string=f'start_time >= "{recent_time}"',
                        max_results=50
                    )
                    print(f"   📊 Found {len(traces)} recent traces in '{project}' project")
                    all_traces.extend(traces)
                except Exception as e:
                    print(f"   ⚠️ Failed to search '{project}' project: {e}")

            # Debug: Show all trace names
            print(f"      🔍 All trace names: {[getattr(t, 'name', 'unknown') for t in all_traces]}")

            # Track individual ACE role tokens
            generator_tokens = 0
            reflector_tokens = 0
            curator_tokens = 0

            # Track processed traces to avoid double-counting
            ace_trace_ids = set()

            print(f"   🔍 Processing {len(all_traces)} total traces...")

            # First pass: identify and process ACE role traces
            for trace in all_traces:
                trace_name = getattr(trace, 'name', 'unknown')
                trace_name_lower = trace_name.lower()

                if any(role in trace_name_lower for role in ['generator', 'reflector', 'curator']):
                    print(f"      📋 ACE Trace: '{trace_name}'")

                    # Get usage from trace or spans
                    total_tokens = 0

                    # Debug: Check trace.usage structure
                    print(f"         🔍 trace.usage type: {type(getattr(trace, 'usage', None))}, value: {getattr(trace, 'usage', None)}")

                    if trace.usage:
                        total_tokens = trace.usage.get('total_tokens', 0)
                        print(f"         💰 Tokens: {total_tokens}")
                    else:
                        # Check spans for this trace
                        try:
                            spans = client.search_spans(trace_id=trace.id)
                            for span in spans:
                                if hasattr(span, 'usage') and span.usage:
                                    span_tokens = span.usage.get('total_tokens', 0)
                                    total_tokens += span_tokens
                                    # Track span trace IDs that belong to ACE roles
                                    if hasattr(span, 'trace_id'):
                                        ace_trace_ids.add(span.trace_id)

                            if total_tokens > 0:
                                print(f"         💰 Tokens (from spans): {total_tokens}")
                        except Exception as e:
                            print(f"         ⚠️ Failed to get spans: {e}")

                    # Classify by role
                    if 'generator' in trace_name_lower:
                        generator_tokens += total_tokens
                        print(f"         🎯 Added to Generator")
                    elif 'reflector' in trace_name_lower:
                        reflector_tokens += total_tokens
                        print(f"         🔍 Added to Reflector")
                    elif 'curator' in trace_name_lower:
                        curator_tokens += total_tokens
                        print(f"         📝 Added to Curator")

                    # Mark this trace as processed
                    ace_trace_ids.add(trace.id)

            # Browser-use tokens are now tracked directly, no need to search Opik

            # Calculate total ACE tokens
            ace_tokens = generator_tokens + reflector_tokens + curator_tokens

            print(f"   📊 Role breakdown:")
            print(f"      🎯 Generator: {generator_tokens} tokens")
            print(f"      🔍 Reflector: {reflector_tokens} tokens")
            print(f"      📝 Curator: {curator_tokens} tokens")

            return (ace_tokens, generator_tokens, reflector_tokens, curator_tokens)

        except Exception as e:
            print(f"   Warning: Could not retrieve token usage from Opik: {e}")
            return 0, 0, 0, 0

    async def _check_domain(self, domain: str, strategy: str):
        """Execute browser automation to check domain with retry logic."""
        max_retries = 3
        last_error = None
        total_steps = 0
        attempt_details = []
        total_browseruse_tokens = 0  # Track tokens across all attempts


        for attempt in range(max_retries):
            print(f"   ⏳ Attempt {attempt + 1}/{max_retries}...")
            browser = None
            try:
                # Start browser with debugging
                print(f"   🌐 Starting browser (headless={self.headless})...")
                browser = Browser(headless=self.headless)
                await browser.start()
                print(f"   ✅ Browser started successfully")

                # Create agent with ChatAnthropic (will log to browser-use project via env var)
                llm = ChatAnthropic(model=self.model, temperature=0.0)
                print(f"   🤖 Created ChatAnthropic for browser-use project: {self.model}")

                task = f"""
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

{strategy}
"""

                print(f"   🎯 Creating agent with task...")
                agent = Agent(
                    task=task,
                    llm=llm,
                    browser=browser,
                    max_actions_per_step=5,
                    max_steps=20,
                    calculate_cost=True  # Enable cost tracking
                )

                print(f"   🚀 Running agent (timeout: 180s)...")
                # Run with reasonable timeout to allow LLM calls to complete
                history = await asyncio.wait_for(agent.run(), timeout=180.0)
                print(f"   📋 Agent completed, processing results...")

                # Parse result
                output = history.final_result() if hasattr(history, "final_result") else ""
                steps = len(history.action_names()) if hasattr(history, "action_names") and history.action_names() else 0

                # Add steps to total and track attempt
                total_steps += steps
                attempt_details.append(f"attempt {attempt + 1}: {steps} steps")

                # Extract detailed execution logs for ACE Reflector
                execution_logs = []
                try:
                    # Get action history with results
                    if hasattr(history, 'action_history') and history.action_history():
                        for i, action in enumerate(history.action_history(), 1):
                            action_log = f"Step {i}: {action}"
                            execution_logs.append(action_log)

                    # Get action results (outcomes of each action)
                    if hasattr(history, 'action_results') and history.action_results():
                        execution_logs.append("\nAction Results:")
                        for i, result in enumerate(history.action_results(), 1):
                            result_log = f"Result {i}: {result}"
                            execution_logs.append(result_log)

                    # Get URLs visited
                    if hasattr(history, 'urls') and history.urls():
                        execution_logs.append(f"\nURLs visited: {history.urls()}")

                    # Get any errors
                    if hasattr(history, 'errors') and history.errors():
                        execution_logs.append(f"\nErrors encountered: {history.errors()}")

                    # Get model thoughts/reasoning
                    if hasattr(history, 'model_thoughts') and history.model_thoughts():
                        execution_logs.append("\nAgent reasoning:")
                        for i, thought in enumerate(history.model_thoughts(), 1):
                            execution_logs.append(f"Thought {i}: {thought}")

                except Exception as e:
                    execution_logs.append(f"Error extracting logs: {e}")

                # Determine status
                status = "ERROR"
                output_upper = output.upper()
                domain_upper = domain.upper()

                if f"AVAILABLE: {domain_upper}" in output_upper:
                    status = "AVAILABLE"
                elif f"TAKEN: {domain_upper}" in output_upper:
                    status = "TAKEN"

                # If successful, collect tokens before returning
                if status != "ERROR":
                    print(f"   ✅ Success! {status} ({steps} steps)")

                    # Collect tokens from this successful attempt
                    attempt_tokens = 0

                    # Method 1: Try to get tokens from history (works after successful completion)
                    if 'history' in locals() and history and hasattr(history, "usage"):
                        try:
                            usage = history.usage
                            if usage:
                                # Try different ways to extract total tokens
                                if hasattr(usage, 'total_tokens'):
                                    attempt_tokens = usage.total_tokens
                                elif isinstance(usage, dict) and 'total_tokens' in usage:
                                    attempt_tokens = usage['total_tokens']
                                elif hasattr(usage, 'input_tokens') and hasattr(usage, 'output_tokens'):
                                    attempt_tokens = usage.input_tokens + usage.output_tokens
                                elif isinstance(usage, dict) and 'input_tokens' in usage and 'output_tokens' in usage:
                                    attempt_tokens = usage['input_tokens'] + usage['output_tokens']
                        except Exception as e:
                            print(f"   ⚠️ Could not get tokens from history: {e}")

                    # Method 2: Try agent.token_cost_service (works even during partial execution)
                    if attempt_tokens == 0 and 'agent' in locals() and agent:
                        try:
                            if hasattr(agent, 'token_cost_service'):
                                usage_summary = await agent.token_cost_service.get_usage_summary()
                                if usage_summary:
                                    if isinstance(usage_summary, dict) and 'total_tokens' in usage_summary:
                                        attempt_tokens = usage_summary['total_tokens']
                                    elif hasattr(usage_summary, 'total_tokens'):
                                        attempt_tokens = usage_summary.total_tokens
                        except Exception as e:
                            print(f"   ⚠️ Could not get tokens from agent service: {e}")

                    total_browseruse_tokens += attempt_tokens
                    print(f"   🤖 Attempt {attempt + 1} tokens: {attempt_tokens} (total: {total_browseruse_tokens})")

                    return {
                        "status": status,
                        "steps": steps,  # Steps from final attempt
                        "total_steps": total_steps,  # Cumulative steps
                        "output": output,
                        "attempt": attempt + 1,
                        "attempt_details": attempt_details,
                        "browseruse_tokens": total_browseruse_tokens,
                        "execution_logs": execution_logs  # Detailed browser execution logs
                    }

                # Store error for potential retry
                print(f"   ❌ Failed ({steps} steps) - retrying...")
                last_error = f"Failed to get valid result: {output}"

            except asyncio.TimeoutError:
                # Calculate additional steps for timeout duration
                timeout_duration = 180.0  # The timeout value used in wait_for()
                timeout_steps = calculate_timeout_steps(timeout_duration)

                # Get actual steps even on timeout
                try:
                    actual_steps = history.number_of_steps() if 'history' in locals() and hasattr(history, "number_of_steps") else 0
                except:
                    actual_steps = 0

                # Add timeout steps to actual steps
                steps = actual_steps + timeout_steps
                total_steps += steps
                attempt_details.append(f"attempt {attempt + 1}: {steps} steps (timeout, +{timeout_steps} for duration)")
                print(f"   ⏱️ Timeout ({steps} steps, +{timeout_steps} for duration) - retrying...")
                last_error = f"Timeout on attempt {attempt + 1}"

            except Exception as e:
                # Get actual steps even on error
                try:
                    steps = history.number_of_steps() if 'history' in locals() and hasattr(history, "number_of_steps") else 0
                except:
                    steps = 0

                total_steps += steps
                attempt_details.append(f"attempt {attempt + 1}: {steps} steps (error)")
                print(f"   💥 Error ({steps} steps): {str(e)}")
                print(f"   📝 Error type: {type(e).__name__}")
                if hasattr(e, '__traceback__'):
                    import traceback
                    tb_lines = traceback.format_exc().split('\n')
                    if len(tb_lines) >= 3:
                        print(f"   🔍 Traceback preview: {tb_lines[-3]}")
                last_error = f"Error on attempt {attempt + 1}: {str(e)}"

            finally:
                # Capture tokens from this attempt using browser-use's cost tracking
                attempt_tokens = 0

                # Method 1: Try to get tokens from history (works after successful completion)
                if 'history' in locals() and history and hasattr(history, "usage"):
                    try:
                        usage = history.usage
                        if usage:
                            # Try different ways to extract total tokens
                            if hasattr(usage, 'total_tokens'):
                                attempt_tokens = usage.total_tokens
                            elif isinstance(usage, dict) and 'total_tokens' in usage:
                                attempt_tokens = usage['total_tokens']
                            elif hasattr(usage, 'input_tokens') and hasattr(usage, 'output_tokens'):
                                attempt_tokens = usage.input_tokens + usage.output_tokens
                            elif isinstance(usage, dict) and 'input_tokens' in usage and 'output_tokens' in usage:
                                attempt_tokens = usage['input_tokens'] + usage['output_tokens']
                    except Exception as e:
                        print(f"   ⚠️ Could not get tokens from history: {e}")

                # Method 2: Try agent.token_cost_service (works even during partial execution)
                if attempt_tokens == 0 and 'agent' in locals() and agent:
                    try:
                        if hasattr(agent, 'token_cost_service'):
                            usage_summary = await agent.token_cost_service.get_usage_summary()
                            if usage_summary:
                                if isinstance(usage_summary, dict) and 'total_tokens' in usage_summary:
                                    attempt_tokens = usage_summary['total_tokens']
                                elif hasattr(usage_summary, 'total_tokens'):
                                    attempt_tokens = usage_summary.total_tokens
                    except Exception as e:
                        print(f"   ⚠️ Could not get tokens from agent service: {e}")

                total_browseruse_tokens += attempt_tokens
                print(f"   🤖 Attempt {attempt + 1} tokens: {attempt_tokens} (total: {total_browseruse_tokens})")

                if browser:
                    try:
                        await browser.stop()
                    except:
                        pass

        # All retries failed - use accumulated tokens from all attempts
        return {
            "status": "ERROR",
            "steps": steps if 'steps' in locals() else 0,
            "total_steps": total_steps,
            "error": f"Failed after {max_retries} attempts. Last error: {last_error}",
            "attempt": max_retries,
            "attempt_details": attempt_details,
            "browseruse_tokens": total_browseruse_tokens,
            "execution_logs": execution_logs if 'execution_logs' in locals() else []
        }


def get_test_domains() -> List[str]:
    """Get list of test domains to check."""
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


def main():
    """Main function - ACE online learning for domain checking."""

    # Capture start time for trace filtering
    import datetime
    run_start_time = datetime.datetime.now(datetime.timezone.utc)

    # Configure Opik if available
    try:
        configure_opik(project_name="ace-browser-domain-checker")
        print("📊 Opik observability enabled")
    except:
        print("📊 Opik not available, continuing without observability")

    print("\n🚀 ACE + Browser-Use Domain Checker")
    print("🧠 Learns after each domain check!")
    print("=" * 50)

    # Get test domains
    domains = get_test_domains()
    print(f"📋 Testing {len(domains)} domains:")
    for i, domain in enumerate(domains, 1):
        print(f"  {i}. {domain}")

    # Create ACE components with OnlineAdapter (using LiteLLM for ACE roles)
    llm = LiteLLMClient(model="claude-sonnet-4-5-20250929", temperature=0.2, max_tokens=2048)

    # Create prompt manager
    manager = PromptManager()

    adapter = OnlineAdapter(
        playbook=Playbook(),
        generator=Generator(llm, prompt_template=manager.get_generator_prompt()),
        reflector=Reflector(llm, prompt_template=manager.get_reflector_prompt()),
        curator=Curator(llm, prompt_template=manager.get_curator_prompt()),
        max_refinement_rounds=2,
    )

    # Create environment
    environment = DomainCheckEnvironment(
        headless=False,  # Using headless mode false for better performance
        model="claude-sonnet-4-5-20250929",
        run_start_time=run_start_time  # Pass start time for trace filtering
    )

    print("\n🔄 Starting incremental ACE learning...\n")

    # Create all samples
    samples = []
    for i, domain in enumerate(domains, 1):
        samples.append(Sample(
            question=f"Check if domain {domain} is available",
            ground_truth="AVAILABLE or TAKEN",
            context="Achieve the best performance by optimising for accuracy and efficiency."
        ))

    # Run OnlineAdapter - it processes samples one by one and learns after each!
    print(f"\n📋 Processing {len(domains)} domains...")
    results = adapter.run(samples, environment)

    # Query ACE tokens after all roles have completed
    print(f"\n💰 Querying ACE token usage after all domains processed...")
    import time
    time.sleep(5)  # Wait for Opik to index final traces
    (total_ace_tokens, total_generator_tokens, total_reflector_tokens, total_curator_tokens) = environment._get_token_usage()

    # Show results
    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print(f"{'#':<3} {'Domain':<25} {'Status':<10} {'Acc':<4} {'Steps':<8} {'Browser-Tokens':<13} {'Details'}")
    print("-" * 85)

    for i, (domain, result) in enumerate(zip(domains, results), 1):
        metrics = result.environment_result.metrics
        status = metrics.get('status', 'UNKNOWN')
        steps = metrics.get('steps', 0)
        total_steps = metrics.get('total_steps', steps)
        correct = metrics.get('correct', False)
        attempt = metrics.get('attempt', 1)
        attempt_details = metrics.get('attempt_details', [])

        # Show detailed step breakdown for multiple attempts
        if attempt > 1:
            step_details = f"({', '.join(attempt_details)})"
        else:
            step_details = "(1 attempt)"

        accuracy_indicator = '✓' if correct else '✗'
        browseruse_tokens = metrics.get('browseruse_tokens', 0)

        print(f"{i:<3} {domain:<25} {status:<10} {accuracy_indicator:<4} {total_steps:<7} {browseruse_tokens:<12} {step_details}")

    # Enhanced Summary
    status_successful = sum(1 for r in results if r.environment_result.metrics.get('status_success', False))
    correct = sum(1 for r in results if r.environment_result.metrics.get('correct', False))
    total_steps = sum(r.environment_result.metrics.get('total_steps', r.environment_result.metrics.get('steps', 0)) for r in results)
    domains_with_retries = sum(1 for r in results if r.environment_result.metrics.get('attempt', 1) > 1)
    total_attempts = sum(r.environment_result.metrics.get('attempt', 1) for r in results)

    avg_steps_per_domain = total_steps / len(results) if results else 0

    # Calculate actual token usage from results
    total_browseruse_tokens = sum(r.environment_result.metrics.get('browseruse_tokens', 0) for r in results)
    # ACE tokens already queried above

    # Calculate averages
    avg_browseruse_tokens_per_domain = total_browseruse_tokens / len(results) if results else 0.0
    avg_ace_tokens_per_domain = total_ace_tokens / len(results) if results else 0.0

    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    print(f"✅ Success rate:          {status_successful:>2}/{len(results)} ({100*status_successful/len(results):>5.1f}%)")
    print(f"🎯 Accuracy rate:         {correct:>2}/{len(results)} ({100*correct/len(results):>5.1f}%)")
    print(f"🔄 Domains w/ retries:    {domains_with_retries:>2}/{len(results)}")
    print(f"🔢 Total attempts:        {total_attempts:>6}")
    print()
    print(f"{'📊 Steps:':<25} {total_steps:>6} total     {avg_steps_per_domain:>6.1f} per domain")
    print(f"{'🤖 Browser-Use Tokens:':<25} {total_browseruse_tokens:>6} total     {avg_browseruse_tokens_per_domain:>6.1f} per domain")
    print(f"{'🧠 ACE Tokens:':<25} {total_ace_tokens:>6} total     {avg_ace_tokens_per_domain:>6.1f} per domain")
    print()
    print("🧠 ACE Role Breakdown (Think → Learn):")
    print(f"   🎯 Generator:      {total_generator_tokens:>6} tokens  (strategy planning)")
    print(f"   🔍 Reflector:      {total_reflector_tokens:>6} tokens  (performance analysis)")
    print(f"   📝 Curator:        {total_curator_tokens:>6} tokens  (playbook updates)")
    print(f"   {'─' * 40}")
    print(f"   🧠 Total ACE:      {total_ace_tokens:>6} tokens")
    print("=" * 80)

    # Show learned strategies
    if adapter.playbook.bullets():
        print(f"\n🎯 Learned Strategies:")
        for i, bullet in enumerate(adapter.playbook.bullets(), 1):
            print(f"  {i}. {bullet.content}")

    # Save playbook
    playbook_path = Path("ace_domain_playbook.json")
    adapter.playbook.save_to_file(str(playbook_path))
    print(f"\n💾 Playbook saved to {playbook_path}")
    
    #traces = client.search_traces()
    #print("Collected trace IDs:", [trace.id for trace in traces])



if __name__ == "__main__":
    main()
