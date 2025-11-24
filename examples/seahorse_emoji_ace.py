#!/usr/bin/env python3
"""
🌊 The Kayba Test - Self-Learning Demo 🌊
=========================================

Demonstrates ACE's ability to self-reflect and learn strategies
without external feedback. Tests the famous "seahorse emoji problem"
that confuses most LLMs.

Named after Kayba AI (海馬 kaiba = seahorse in Japanese).
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
    console.print(
        "[bold cyan]🌊 The Kayba Test - ACE Self-Learning Demo 🌊[/bold cyan]"
    )
    console.print("[dim]Using Claude Opus 4.1[/dim]")
    console.print("=" * 60 + "\n")

    # Configure Opik observability
    integration = configure_opik(
        project_name="kayba-test", tags=["demo", "seahorse", "self-learning"]
    )
    status = "✓ Enabled" if integration.is_available() else "✗ Disabled"
    console.print(f"[cyan]Opik Observability: {status}[/cyan]")
    if integration.is_available():
        console.print("[dim]View traces at: https://www.comet.com/[/dim]")
    console.print()

    # Setup - Claude Opus 4.1 via ACELiteLLM
    agent = ACELiteLLM(
        model="claude-opus-4-1-20250805",
        temperature=0.7,
        max_tokens=4000,
        is_learning=True,
    )

    question = "Give me the seahorse emoji?"

    # First ask - provide initial learning material
    console.print("[yellow]━━━ Round 1: Teaching ACE About Emojis ━━━[/yellow]")
    console.print(
        "[dim]First, let's teach ACE with a sample that will provide context...[/dim]\n"
    )

    # Create a learning sample about emojis (with some ground truth to establish knowledge)
    learning_sample = Sample(
        question="What emoji represents a horse?", ground_truth="🐴 (horse emoji)"
    )

    environment = SimpleEnvironment()
    console.print("Teaching ACE about emoji questions...")
    agent.learn(samples=[learning_sample], environment=environment, epochs=1)

    if len(agent.playbook.bullets()) > 0:
        console.print(
            f"[green]Playbook updated with {len(agent.playbook.bullets())} learned strategies[/green]"
        )
        console.print("\n[cyan]📚 Current Playbook:[/cyan]")
        console.print(Panel(agent.playbook.as_prompt(), style="cyan"))

    # Now ask the actual question
    console.print(f"\n[yellow]━━━ Round 2: Asking About Seahorse ━━━[/yellow]")
    console.print(f"[bold]Question:[/bold] {question}")
    console.print(
        f"[dim]Playbook: {len(agent.playbook.bullets())} learned strategies[/dim]\n"
    )
    answer1 = agent.ask(question=question, context="")
    console.print(f"[bold]Final Answer:[/bold] {answer1}")
    console.print(
        f"[dim]Note: ACE applies learned emoji strategies to new question[/dim]"
    )

    # Learn from this interaction too
    console.print("\n[cyan]━━━ Self-Learning Phase ━━━[/cyan]")
    console.print("[dim]ACE learns from its seahorse emoji response...[/dim]")

    seahorse_sample = Sample(question=question, ground_truth=None)  # Self-learning
    agent.learn(samples=[seahorse_sample], environment=environment, epochs=1)

    console.print(
        f"[green]Playbook now contains {len(agent.playbook.bullets())} total learned insights[/green]"
    )

    # Final ask to show further evolution
    console.print(f"\n[yellow]━━━ Round 3: Enhanced Knowledge ━━━[/yellow]")
    console.print(f"[bold]Question:[/bold] {question}")
    console.print(
        f"[dim]Playbook: {len(agent.playbook.bullets())} learned strategies[/dim]\n"
    )
    answer2 = agent.ask(question=question, context="")
    console.print(f"[bold]Final Answer:[/bold] {answer2}")
    console.print(f"[dim]Note: ACE continuously refines its approach[/dim]")

    # Just show the two answers for comparison
    import time

    time.sleep(2)  # Pause before showing results
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]📊 Results Comparison[/bold cyan]")
    console.print("=" * 60)

    console.print("\n[yellow]Round 2 Answer (initial with emoji knowledge):[/yellow]")
    console.print(Panel(answer1, style="yellow"))

    console.print("[green]Round 3 Answer (enhanced with seahorse learning):[/green]")
    console.print(Panel(answer2, style="green"))

    console.print("\n[bold red]⚠️  Fact Check:[/bold red]")
    console.print(
        "[dim]There is NO seahorse emoji in Unicode (despite what models often claim).[/dim]"
    )
    console.print(
        "[dim]This demo shows how ACE learns strategies through self-reflection.[/dim]\n"
    )


if __name__ == "__main__":
    main()
