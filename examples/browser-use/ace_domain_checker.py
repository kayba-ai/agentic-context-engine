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

from browser_use import Agent, Browser, ChatOpenAI

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

load_dotenv()


class DomainCheckEnvironment(TaskEnvironment):
    """Environment that evaluates domain checking performance."""

    def __init__(self, headless: bool = True, model: str = "gpt-4o-mini"):
        self.headless = headless
        self.model = model

    def evaluate(self, sample: Sample, generator_output):
        """Run browser automation and evaluate the result."""

        # Extract domain from the sample question
        domain = sample.question.replace("Check if domain ", "").replace(" is available", "")

        # Get strategy from generator
        strategy = generator_output.final_answer

        print(f"🔍 Checking domain: {domain}")

        # Run browser automation
        result = asyncio.run(self._check_domain(domain, strategy))

        # Evaluate correctness and efficiency
        correct = result['status'] != "ERROR"
        efficient = result['steps'] <= 8

        feedback = f"Domain check {'succeeded' if correct else 'failed'}. "
        feedback += f"Took {result['steps']} steps. "
        if not efficient:
            feedback += "Should be more efficient (target: ≤8 steps). "
        if result['status'] == "ERROR":
            feedback += f"Error: {result.get('error', 'Unknown error')}. "

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=None,  # No ground truth available for domain checking
            metrics={
                "correct": correct,
                "efficient": efficient,
                "steps": result['steps'],
                "total_steps": result.get('total_steps', result['steps']),
                "status": result['status'],
                "attempt": result.get('attempt', 1),
                "attempt_details": result.get('attempt_details', [])
            }
        )

    async def _check_domain(self, domain: str, strategy: str):
        """Execute browser automation to check domain with retry logic."""
        max_retries = 3
        last_error = None
        total_steps = 0
        attempt_details = []

        for attempt in range(max_retries):
            print(f"   ⏳ Attempt {attempt + 1}/{max_retries}...")
            browser = None
            try:
                # Start browser
                browser = Browser(headless=self.headless)
                await browser.start()

                # Create agent with the strategy
                llm = ChatOpenAI(model=self.model, temperature=0.0)

                task = f"""{strategy}

Check if the domain "{domain}" is available for registration.

Use domain lookup websites. Avoid sites with CAPTCHAs.

Output format (exactly one of these):
AVAILABLE: {domain}
TAKEN: {domain}
ERROR: <reason>"""

                agent = Agent(
                    task=task,
                    llm=llm,
                    browser=browser,
                    max_actions_per_step=5,
                    max_steps=20,
                )

                # Run with timeout
                history = await asyncio.wait_for(agent.run(), timeout=180.0)

                # Parse result
                output = history.final_result() if hasattr(history, "final_result") else ""
                steps = len(history.action_names()) if hasattr(history, "action_names") and history.action_names() else 0

                # Add steps to total and track attempt
                total_steps += steps
                attempt_details.append(f"attempt {attempt + 1}: {steps} steps")

                # Determine status
                status = "ERROR"
                output_upper = output.upper()
                domain_upper = domain.upper()

                if f"AVAILABLE: {domain_upper}" in output_upper:
                    status = "AVAILABLE"
                elif f"TAKEN: {domain_upper}" in output_upper:
                    status = "TAKEN"

                # If successful, return immediately with cumulative data
                if status != "ERROR":
                    print(f"   ✅ Success! {status} ({steps} steps)")
                    return {
                        "status": status,
                        "steps": steps,  # Steps from final attempt
                        "total_steps": total_steps,  # Cumulative steps
                        "output": output,
                        "attempt": attempt + 1,
                        "attempt_details": attempt_details
                    }

                # Store error for potential retry
                print(f"   ❌ Failed ({steps} steps) - retrying...")
                last_error = f"Failed to get valid result: {output}"

            except asyncio.TimeoutError:
                # Get actual steps even on timeout
                try:
                    steps = history.number_of_steps() if 'history' in locals() and hasattr(history, "number_of_steps") else 0
                except:
                    steps = 20  # max_steps if we can't determine

                total_steps += steps
                attempt_details.append(f"attempt {attempt + 1}: {steps} steps (timeout)")
                print(f"   ⏱️ Timeout ({steps} steps) - retrying...")
                last_error = f"Timeout on attempt {attempt + 1}"

            except Exception as e:
                # Get actual steps even on error
                try:
                    steps = history.number_of_steps() if 'history' in locals() and hasattr(history, "number_of_steps") else 0
                except:
                    steps = 0

                total_steps += steps
                attempt_details.append(f"attempt {attempt + 1}: {steps} steps (error)")
                print(f"   💥 Error ({steps} steps) - retrying...")
                last_error = f"Error on attempt {attempt + 1}: {str(e)}"

            finally:
                if browser:
                    try:
                        await browser.stop()
                    except:
                        pass

        # All retries failed
        return {
            "status": "ERROR",
            "steps": steps if 'steps' in locals() else 0,
            "total_steps": total_steps,
            "error": f"Failed after {max_retries} attempts. Last error: {last_error}",
            "attempt": max_retries,
            "attempt_details": attempt_details
        }


def get_test_domains() -> List[str]:
    """Get list of test domains to check."""
    return [
        "test-domain-12345.com",
        "example-test-9999.org",
        "mytest-domain-xyz.net"
    ]


def main():
    """Main function - ACE online learning for domain checking."""

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

    # Create ACE components with OnlineAdapter
    llm = LiteLLMClient(model="gpt-4o-mini", temperature=0.7)

    adapter = OnlineAdapter(
        playbook=Playbook(),
        generator=Generator(llm),
        reflector=Reflector(llm),
        curator=Curator(llm),
        max_refinement_rounds=2,
    )

    # Create environment
    environment = DomainCheckEnvironment(
        headless=False,  # Change to True for headless mode
        model="gpt-4o-mini"
    )

    print("\n🔄 Starting incremental ACE learning...\n")

    # Create all samples
    samples = []
    for i, domain in enumerate(domains, 1):
        samples.append(Sample(
            question=f"Check if domain {domain} is available",
            ground_truth="AVAILABLE or TAKEN",
            context="Use domain lookup websites efficiently. Avoid CAPTCHAs."
        ))

    # Run OnlineAdapter - it processes samples one by one and learns after each!
    print(f"\n📋 Processing {len(domains)} domains...")
    results = adapter.run(samples, environment)

    # Show results
    print("\n" + "=" * 50)
    print("📊 Results:")

    for i, (domain, result) in enumerate(zip(domains, results), 1):
        metrics = result.environment_result.metrics
        status = metrics.get('status', 'UNKNOWN')
        steps = metrics.get('steps', 0)
        total_steps = metrics.get('total_steps', steps)
        correct = metrics.get('correct', False)
        attempt = metrics.get('attempt', 1)
        attempt_details = metrics.get('attempt_details', [])

        # Show detailed step breakdown for multiple attempts
        step_info = f"{total_steps} steps"
        if attempt > 1:
            step_info += f" total ({', '.join(attempt_details)})"
        else:
            step_info += f" (1 attempt)"

        print(f"[{i}] {domain}: {status} ({'✓' if correct else '✗'}) - {step_info}")

    # Enhanced Summary
    successful = sum(1 for r in results if r.environment_result.metrics.get('correct', False))
    total_steps = sum(r.environment_result.metrics.get('total_steps', r.environment_result.metrics.get('steps', 0)) for r in results)
    domains_with_retries = sum(1 for r in results if r.environment_result.metrics.get('attempt', 1) > 1)
    total_attempts = sum(r.environment_result.metrics.get('attempt', 1) for r in results)

    avg_steps_per_domain = total_steps / len(results) if results else 0
    avg_steps_per_success = total_steps / successful if successful > 0 else 0

    print(f"\n✅ Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
    print(f"⚡ Total steps: {total_steps} across all attempts")
    print(f"📈 Average steps per domain: {avg_steps_per_domain:.1f}")
    print(f"🎯 Average steps per success: {avg_steps_per_success:.1f}")
    print(f"🔄 Domains needing retries: {domains_with_retries}/{len(results)}")
    print(f"🔢 Total attempts made: {total_attempts}")
    print(f"🧠 Strategies learned: {len(adapter.playbook.bullets())}")

    # Show learned strategies
    if adapter.playbook.bullets():
        print(f"\n🎯 Learned Strategies:")
        for i, bullet in enumerate(adapter.playbook.bullets(), 1):
            print(f"  {i}. {bullet.content}")

    # Save playbook
    playbook_path = Path("ace_domain_playbook.json")
    adapter.playbook.to_file(str(playbook_path))
    print(f"\n💾 Playbook saved to {playbook_path}")


if __name__ == "__main__":
    main()
