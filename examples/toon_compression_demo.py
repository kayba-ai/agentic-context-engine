#!/usr/bin/env python3
"""
Real-world test comparing ACE with Toon compression vs without.
Measures actual token usage and cost savings.
"""

import os
from ace import (
    Playbook,
    Generator,
    Reflector,
    Curator,
    OfflineAdapter,
    Sample,
    SimpleEnvironment,
    LiteLLMClient,
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import time

console = Console()


def create_training_samples():
    """Create sample questions for ACE training."""
    return [
        Sample(
            question="What is 2 + 2?",
            context="Solve this arithmetic problem",
            ground_truth="4"
        ),
        Sample(
            question="What is the capital of France?",
            context="Answer with just the city name",
            ground_truth="Paris"
        ),
        Sample(
            question="What color is the sky on a clear day?",
            context="One word answer",
            ground_truth="blue"
        ),
    ]


def run_ace_experiment(use_toon: bool, samples, environment):
    """Run ACE training with or without Toon compression."""

    # Initialize LLM client - using Google AI Studio 
    llm = LiteLLMClient(
        model="gemini/gemini-2.5-flash", 
        temperature=0.7,
        max_tokens=2000,  
    )

    # Create roles with appropriate format
    format_str = "toon" if use_toon else "markdown"
    generator = Generator(llm, playbook_format=format_str)
    reflector = Reflector(llm)
    curator = Curator(llm, playbook_format=format_str)

    # Create adapter
    adapter = OfflineAdapter(
        playbook=Playbook(),
        generator=generator,
        reflector=reflector,
        curator=curator,
    )

    # Track metrics
    start_time = time.time()

    # Run training
    console.print(f"\n[cyan]Running ACE with {'Toon' if use_toon else 'Markdown'} format...[/cyan]")
    results = adapter.run(samples, environment, epochs=1)

    elapsed_time = time.time() - start_time

    # Get playbook stats
    playbook = adapter.playbook
    bullets_created = len(playbook.bullets())

    # Measure playbook size
    playbook_text = playbook.as_prompt(format=format_str)
    playbook_chars = len(playbook_text)

    # Estimate tokens (rough approximation: 1 token ≈ 4 chars for English)
    estimated_tokens = playbook_chars // 4

    return {
        "format": "Toon" if use_toon else "Markdown",
        "bullets_created": bullets_created,
        "playbook_chars": playbook_chars,
        "estimated_tokens": estimated_tokens,
        "elapsed_time": elapsed_time,
        "playbook_preview": playbook_text[:200] + "..." if len(playbook_text) > 200 else playbook_text,
    }


def main():
    """Run comparison test."""

    console.print(Panel.fit(
        "[bold cyan]ACE Framework: Toon Compression Real-World Test[/bold cyan]\n"
        "[dim]Comparing token usage with actual LLM (Gemini)[/dim]",
        border_style="cyan"
    ))

    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        console.print("[red] GOOGLE_API_KEY not set![/red]")
        console.print("[yellow]Run: export GOOGLE_API_KEY='your-key-here'[/yellow]")
        return

    console.print("[green] API key detected[/green]\n")

    # Create test data
    samples = create_training_samples()
    environment = SimpleEnvironment()

    console.print(f"[bold]Test Setup:[/bold]")
    console.print(f"  • Samples: {len(samples)}")
    console.print(f"  • Epochs: 1\n")

    # Run both experiments
    try:
        # Test 1: Without Toon (baseline)
        console.print("[bold yellow]═══ Test 1: Baseline (Markdown) ═══[/bold yellow]")
        results_markdown = run_ace_experiment(
            use_toon=False,
            samples=samples,
            environment=environment
        )

        # Test 2: With Toon compression
        console.print("\n[bold yellow]═══ Test 2: With Toon Compression ═══[/bold yellow]")
        results_toon = run_ace_experiment(
            use_toon=True,
            samples=samples,
            environment=environment
        )

        # Display results
        console.print("\n" + "=" * 70)
        console.print("[bold green] RESULTS COMPARISON[/bold green]")
        console.print("=" * 70 + "\n")

        # Create comparison table
        table = Table(title="Playbook Representation Comparison", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Markdown", justify="right", style="yellow")
        table.add_column("Toon", justify="right", style="green")
        table.add_column("Savings", justify="right", style="bold green")

        # Bullets created
        table.add_row(
            "Bullets Created",
            str(results_markdown["bullets_created"]),
            str(results_toon["bullets_created"]),
            "N/A"
        )

        # Character count
        char_savings = results_markdown["playbook_chars"] - results_toon["playbook_chars"]
        char_savings_pct = (char_savings / results_markdown["playbook_chars"]) * 100 if results_markdown["playbook_chars"] > 0 else 0
        table.add_row(
            "Playbook Size (chars)",
            f"{results_markdown['playbook_chars']:,}",
            f"{results_toon['playbook_chars']:,}",
            f"{char_savings:,} ({char_savings_pct:.1f}%)"
        )

        # Estimated tokens
        token_savings = results_markdown["estimated_tokens"] - results_toon["estimated_tokens"]
        token_savings_pct = (token_savings / results_markdown["estimated_tokens"]) * 100 if results_markdown["estimated_tokens"] > 0 else 0
        table.add_row(
            "Est. Tokens (÷4)",
            f"{results_markdown['estimated_tokens']:,}",
            f"{results_toon['estimated_tokens']:,}",
            f"{token_savings:,} ({token_savings_pct:.1f}%)"
        )

        # Execution time
        time_diff = results_markdown["elapsed_time"] - results_toon["elapsed_time"]
        table.add_row(
            "Execution Time (s)",
            f"{results_markdown['elapsed_time']:.2f}",
            f"{results_toon['elapsed_time']:.2f}",
            f"{time_diff:+.2f}s"
        )

        console.print(table)

        # Cost estimation
        console.print("\n[bold] Cost Estimation (Gemini 2.5 Flash @ $0.075/1M input tokens):[/bold]")

        # Assume 3 LLM calls per sample (Generator, Reflector, Curator)
        # Each call includes the playbook
        calls_per_sample = 2  # Generator and Curator use full playbook
        total_samples = len(samples) * 1  # 1 epoch

        markdown_total_tokens = results_markdown["estimated_tokens"] * calls_per_sample * total_samples
        toon_total_tokens = results_toon["estimated_tokens"] * calls_per_sample * total_samples

        markdown_cost = (markdown_total_tokens / 1_000_000) * 0.075
        toon_cost = (toon_total_tokens / 1_000_000) * 0.075

        console.print(f"  Markdown total:  ~{markdown_total_tokens:,} tokens → ${markdown_cost:.6f}")
        console.print(f"  Toon total:      ~{toon_total_tokens:,} tokens → ${toon_cost:.6f}")
        console.print(f"  [green]Savings:         ~{markdown_total_tokens - toon_total_tokens:,} tokens → ${markdown_cost - toon_cost:.6f}[/green]")

        # Projected savings at scale
        console.print("\n[bold] Projected Savings (1000 inferences):[/bold]")
        scaled_markdown = markdown_cost * (1000 / total_samples)
        scaled_toon = toon_cost * (1000 / total_samples)
        console.print(f"  Markdown:  ${scaled_markdown:.2f}")
        console.print(f"  Toon:      ${scaled_toon:.2f}")
        console.print(f"  [green bold]Savings:   ${scaled_markdown - scaled_toon:.2f} ({token_savings_pct:.1f}%)[/green bold]")

        # Show playbook previews
        console.print("\n[bold] Playbook Format Comparison:[/bold]\n")

        console.print(Panel(
            results_markdown["playbook_preview"],
            title="[yellow]Markdown Format[/yellow]",
            border_style="yellow"
        ))

        console.print(Panel(
            results_toon["playbook_preview"],
            title="[green]Toon Format[/green]",
            border_style="green"
        ))

        console.print("\n[bold green] Real-world test completed successfully![/bold green]\n")

    except Exception as e:
        console.print(f"\n[bold red] Error during test:[/bold red]")
        console.print(f"[red]{type(e).__name__}: {str(e)}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


if __name__ == "__main__":
    main()
