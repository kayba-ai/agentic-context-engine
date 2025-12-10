#!/usr/bin/env python3
"""
ACE + Browser-Use Form Filler Demo

Shows ACE learning to improve at filling web forms.
Uses OnlineACE for incremental learning after each form.
"""

import asyncio
from typing import Dict
from dotenv import load_dotenv

from browser_use import Agent as BrowserAgent, Browser, ChatOpenAI

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

# Import form-specific utilities
from form_utils import get_test_forms

load_dotenv()


class FormFillEnvironment(TaskEnvironment):
    """Environment that evaluates form filling performance."""

    def __init__(self, headless: bool = True, model: str = "gpt-4o-mini"):
        self.headless = headless
        self.model = model

    def evaluate(self, sample: Sample, agent_output):
        """Run browser automation and evaluate the result."""

        # Extract form data from sample
        form_data = eval(
            sample.context
        )  # Simple eval for demo - in production use proper parsing

        # Get strategy from agent
        strategy = agent_output.final_answer

        # Run browser automation
        result = asyncio.run(self._fill_form(form_data, strategy))

        # Evaluate success and efficiency
        success = result["status"] == "SUCCESS"
        efficient = result["steps"] <= 12

        feedback = f"Form filling {'succeeded' if success else 'failed'}. "
        feedback += f"Took {result['steps']} steps. "
        if not efficient:
            feedback += "Should be more efficient (target: ≤12 steps). "
        if result["status"] == "ERROR":
            feedback += f"Error: {result.get('error', 'Unknown error')}. "

        return EnvironmentResult(
            feedback=feedback,
            ground_truth=None,  # No ground truth available for form filling
            metrics={
                "success": success,
                "efficient": efficient,
                "steps": result["steps"],
                "status": result["status"],
            },
        )

    async def _fill_form(self, form_data: Dict, strategy: str):
        """Execute browser automation to fill form."""
        browser = None
        try:
            # Start browser
            browser = Browser(headless=self.headless)
            await browser.start()

            # Create agent with the strategy
            llm = ChatOpenAI(model=self.model, temperature=0.0)

            # Format form data for task
            form_text = ""
            for field, value in form_data.items():
                form_text += f"- {field}: {value}\n"

            task = f"""{strategy}

Fill out a web form with this data:
{form_text}

Navigate to a form (like a contact form, signup form, etc.) and fill it out accurately.
You can use Google Forms, demo forms, or create a simple HTML form.

Output when done:
SUCCESS: Form filled successfully
ERROR: <reason>"""

            browser_agent = BrowserAgent(
                task=task,
                llm=llm,
                browser=browser,
                max_actions_per_step=5,
                max_steps=25,
            )

            # Run with timeout
            history = await asyncio.wait_for(browser_agent.run(), timeout=240.0)

            # Parse result
            output = history.final_result() if hasattr(history, "final_result") else ""
            steps = (
                len(history.action_names())
                if hasattr(history, "action_names") and history.action_names()
                else 0
            )

            # Determine status
            status = "ERROR"
            if "SUCCESS:" in output.upper():
                status = "SUCCESS"

            return {"status": status, "steps": steps, "output": output}

        except asyncio.TimeoutError:
            # Get actual steps even on timeout - history should exist
            try:
                steps = (
                    history.number_of_steps()
                    if "history" in locals() and hasattr(history, "number_of_steps")
                    else 0
                )
            except:
                steps = 25  # max_steps if we can't determine
            return {"status": "ERROR", "steps": steps, "error": "Timeout"}
        except Exception as e:
            # Get actual steps even on error - history might exist
            try:
                steps = (
                    history.number_of_steps()
                    if "history" in locals() and hasattr(history, "number_of_steps")
                    else 0
                )
            except:
                steps = 0
            return {"status": "ERROR", "steps": steps, "error": str(e)}
        finally:
            if browser:
                try:
                    await browser.stop()
                except:
                    pass


def main():
    """Main function - ACE online learning for form filling."""

    # Configure Opik if available
    try:
        configure_opik(project_name="ace-browser-form-filler")
        print("📊 Opik observability enabled")
    except:
        print("📊 Opik not available, continuing without observability")

    print("\n🚀 ACE + Browser-Use Form Filler")
    print("🧠 Learns after each form!")
    print("=" * 40)

    # Get test forms
    forms = get_test_forms()
    print(f"📋 Testing {len(forms)} forms:")
    for i, form in enumerate(forms, 1):
        print(f"  {i}. {form['name']}")

    # Create ACE components with OnlineACE
    llm = LiteLLMClient(model="gpt-4o-mini", temperature=0.7)

    adapter = OnlineACE(
        skillbook=Skillbook(),
        agent=ACEAgent(llm),
        reflector=Reflector(llm),
        skill_manager=SkillManager(llm),
        max_refinement_rounds=2,
    )

    # Create environment
    environment = FormFillEnvironment(
        headless=False, model="gpt-4o-mini"  # Change to True for headless mode
    )

    print("\n🔄 Starting incremental ACE learning...\n")

    # Create all samples
    samples = []
    for form in forms:
        samples.append(
            Sample(
                question=f"Fill out {form['name']} form",
                ground_truth="SUCCESS",
                context=str(form["data"]),  # Simple string representation for demo
            )
        )

    # Run OnlineACE - it processes samples one by one and learns after each!
    results = adapter.run(samples, environment)

    # Show results
    print("\n" + "=" * 40)
    print("📊 Results:")

    for i, (form, result) in enumerate(zip(forms, results), 1):
        metrics = result.environment_result.metrics
        status = metrics.get("status", "UNKNOWN")
        steps = metrics.get("steps", 0)
        success = metrics.get("success", False)

        print(
            f"[{i}] {form['name']}: {status} ({'✓' if success else '✗'}) - {steps} steps"
        )

    # Summary
    successful = sum(
        1 for r in results if r.environment_result.metrics.get("success", False)
    )
    total_steps = sum(r.environment_result.metrics.get("steps", 0) for r in results)
    avg_steps = total_steps / len(results) if results else 0

    print(
        f"\n✅ Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)"
    )
    print(f"⚡ Average steps: {avg_steps:.1f}")
    print(f"🧠 Strategies learned: {len(adapter.skillbook.skills())}")

    # Show learned strategies
    if adapter.skillbook.skills():
        print(f"\n🎯 Learned Strategies:")
        for i, skill in enumerate(adapter.skillbook.skills(), 1):
            print(f"  {i}. {skill.content}")

    # Save skillbook
    from pathlib import Path

    skillbook_path = Path("ace_form_skillbook.json")
    adapter.skillbook.save_to_file(str(skillbook_path))
    print(f"\n💾 Skillbook saved to {skillbook_path}")


if __name__ == "__main__":
    main()
