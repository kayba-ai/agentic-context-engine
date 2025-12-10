#!/usr/bin/env python3
"""
The Kayba Test - Self-Learning Demo

Demonstrates ACE's ability to self-reflect and learn strategies
without external feedback. Tests the famous "seahorse emoji problem"
that confuses most LLMs.

The simple pattern:
1. ASK the question -> LLM likely gets it wrong
2. LEARN from the response (self-reflection)
3. ASK again -> should show improvement

Named after Kayba AI (kaiba = seahorse in Japanese).
"""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from rich.console import Console
from rich.panel import Panel

from ace.integrations import ACELiteLLM
from ace import Sample, SimpleEnvironment
from ace.observability import configure_opik

# Suppress LiteLLM debug messages
import litellm

litellm.suppress_debug_info = True

console = Console()


def main():
    # Display header
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]The Kayba Test - ACE Self-Learning Demo[/bold cyan]")
    console.print("[dim]Simple 'ask twice' pattern[/dim]")
    console.print("=" * 60 + "\n")

    # Configure Opik observability
    integration = configure_opik(
        project_name="kayba-test", tags=["demo", "seahorse", "self-learning"]
    )
    status = "Enabled" if integration.is_available() else "Disabled"
    console.print(f"[cyan]Opik Observability: {status}[/cyan]\n")

    # Setup agent
    agent = ACELiteLLM(
        model="claude-sonnet-4-5-20250929",
        temperature=0.7,
        max_tokens=4000,
        is_learning=True,
    )

    question = "Give me the seahorse emoji?"
    environment = SimpleEnvironment()

    if len(agent.skillbook.skills()) > 0:
        console.print(
            f"[green]Skillbook updated with {len(agent.skillbook.skills())} learned strategies[/green]"
        )
        console.print("\n[cyan]📚 Current Skillbook:[/cyan]")
        console.print(Panel(agent.skillbook.as_prompt(), style="cyan"))

    # Now ask the actual question
    console.print(f"\n[yellow]━━━ Round 2: Asking About Seahorse ━━━[/yellow]")
    console.print(f"[bold]Question:[/bold] {question}")
    console.print(
        f"[dim]Skillbook: {len(agent.skillbook.skills())} learned strategies[/dim]\n"
    )
    answer1 = agent.ask(question=question, context="")
    console.print(f"[bold]Answer:[/bold] {answer1}")

    # Learn from this interaction (self-learning, no ground truth)
    console.print("\n[cyan]--- Learning Phase ---[/cyan]")
    console.print("[dim]ACE reflects on its response...[/dim]")

    sample = Sample(question=question, ground_truth=None)
    agent.learn(samples=[sample], environment=environment, epochs=1)

    console.print(
        f"[green]Skillbook now contains {len(agent.skillbook.skills())} total learned insights[/green]"
    )

    # Final ask to show further evolution
    console.print(f"\n[yellow]━━━ Round 3: Enhanced Knowledge ━━━[/yellow]")
    console.print(f"[bold]Question:[/bold] {question}")
    console.print(
        f"[dim]Skillbook: {len(agent.skillbook.skills())} learned strategies[/dim]\n"
    )
    answer2 = agent.ask(question=question, context="")
    console.print(f"[bold]Answer:[/bold] {answer2}")

    # Results comparison
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Results Comparison[/bold cyan]")
    console.print("=" * 60)

    console.print("\n[yellow]Round 1 (before learning):[/yellow]")
    console.print(Panel(answer1, style="yellow"))

    console.print("[green]Round 2 (after learning):[/green]")
    console.print(Panel(answer2, style="green"))

    console.print("\n[bold red]Fact Check:[/bold red]")
    console.print("[dim]There is NO seahorse emoji in Unicode.[/dim]")
    console.print("[dim]This demo shows ACE learning through self-reflection.[/dim]\n")


if __name__ == "__main__":
    main()
