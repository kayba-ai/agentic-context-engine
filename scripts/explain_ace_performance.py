#!/usr/bin/env python3
"""
Enhanced ACE Performance Explanation Tool.

This script provides comprehensive analysis and explanation of ACE performance,
including strategy attribution, learning patterns, and interactive visualizations.

Usage:
    python scripts/explain_ace_performance.py --results benchmark_results/detailed.json
    python scripts/explain_ace_performance.py --live --benchmark finer_ord --model gpt-4o-mini
    python scripts/explain_ace_performance.py --compare baseline.json ace_adapted.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Note: Explainability module has been replaced by observability (Opik integration)
# The following imports are commented out as the module is deprecated:
# from ace.explainability import (
#     EvolutionTracker, AttributionAnalyzer, InteractionTracer, ExplainabilityVisualizer
# )
from ace.adaptation import AdapterStepResult, Sample
from ace.roles import AgentOutput, ReflectorOutput, SkillManagerOutput
from ace.skillbook import Skillbook


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--results", help="Path to detailed ACE results JSON file to analyze"
    )
    group.add_argument(
        "--live", action="store_true", help="Run live analysis during ACE adaptation"
    )
    group.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "ACE"),
        help="Compare baseline vs ACE results (two JSON files)",
    )

    # Live mode options
    parser.add_argument("--benchmark", help="Benchmark to run for live analysis")
    parser.add_argument(
        "--model", default="gpt-4o-mini", help="Model to use for live analysis"
    )
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of samples for live analysis"
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs for live analysis"
    )

    # Analysis options
    parser.add_argument(
        "--output-dir",
        default="explainability_output",
        help="Directory for analysis outputs",
    )
    parser.add_argument(
        "--generate-html",
        action="store_true",
        help="Generate HTML explainability report",
    )
    parser.add_argument(
        "--include-plots",
        action="store_true",
        help="Include matplotlib plots in output",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed analysis output"
    )

    # Focus areas
    parser.add_argument(
        "--focus",
        choices=["evolution", "attribution", "interactions", "all"],
        default="all",
        help="Focus analysis on specific area",
    )

    return parser.parse_args()


def load_ace_results(file_path: str) -> List[Dict[str, Any]]:
    """Load ACE results from JSON file."""
    with open(file_path, "r") as f:
        data = json.load(f)

    if "results" in data:
        return data["results"]
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Unable to find results in JSON file")


def reconstruct_step_results(
    results_data: List[Dict[str, Any]]
) -> List[AdapterStepResult]:
    """Reconstruct AdapterStepResult objects from JSON data."""
    step_results = []

    for result in results_data:
        # Reconstruct Sample
        sample = Sample(
            question=result.get("question", ""),
            ground_truth=result.get("ground_truth", ""),
            metadata={"sample_id": result.get("sample_id", "")},
        )

        # Reconstruct outputs (simplified)
        agent_output = AgentOutput(
            reasoning=result.get("reasoning", ""),
            final_answer=result.get("prediction", ""),
            skill_ids=result.get("skill_ids", []),
            raw=result.get("agent_raw", {}),
        )

        # Note: Full reconstruction would need the original objects
        # This is a simplified version for demonstration
        step_results.append(
            {
                "sample": sample,
                "agent_output": agent_output,
                "metrics": result.get("metrics", {}),
                "feedback": result.get("feedback", ""),
                "reflection_raw": result.get("reflection", {}),
                "skill_manager_raw": result.get("skill_manager_output", {}),
            }
        )

    return step_results


def analyze_evolution_patterns(results: List[Dict[str, Any]]) -> EvolutionTracker:
    """Analyze skillbook evolution patterns."""
    print("🔍 Analyzing skillbook evolution patterns...")

    tracker = EvolutionTracker()

    # Group by epochs
    epochs_data = {}
    for result in results:
        # Try to infer epoch from sample ordering
        sample_id = result.get("sample_id", "")
        if "epoch" in result:
            epoch = result["epoch"]
        else:
            # Estimate epoch from repetition pattern
            epoch = 1  # Simplified

        if epoch not in epochs_data:
            epochs_data[epoch] = []
        epochs_data[epoch].append(result)

    # Simulate skillbook snapshots
    for epoch, epoch_results in epochs_data.items():
        for step, result in enumerate(epoch_results):
            # Create mock skillbook snapshot
            metrics = result.get("metrics", {})
            tracker.take_snapshot(
                skillbook=Skillbook(),  # Would need actual skillbook state
                epoch=epoch,
                step=step,
                performance_metrics=metrics,
                context=f"Step {step} analysis",
            )

    return tracker


def analyze_strategy_attribution(results: List[Dict[str, Any]]) -> AttributionAnalyzer:
    """Analyze strategy attribution from results."""
    print("🎯 Analyzing strategy attribution...")

    analyzer = AttributionAnalyzer()

    for i, result in enumerate(results):
        # Extract skill usage if available
        skill_ids = result.get("skill_ids", [])

        # Try to extract skill info from reflection skill_tags if available
        if not skill_ids and "reflection" in result:
            reflection = result["reflection"]
            if isinstance(reflection, dict) and "skill_tags" in reflection:
                skill_tags = reflection["skill_tags"]
                if isinstance(skill_tags, list):
                    skill_ids = [
                        tag.get("id", f"tag_{i}_{j}")
                        for j, tag in enumerate(skill_tags)
                    ]

        # Create mock skill IDs if none available (for demonstration)
        if not skill_ids:
            skill_ids = [f"strategy_{i % 3}"]  # Create some variety for demo

        metrics = result.get("metrics", {})
        sample_id = result.get("sample_id", f"sample_{i}")

        # Determine success
        success = None
        if "f1" in metrics:
            success = metrics["f1"] > 0.5

        analyzer.record_skill_usage(
            skill_ids=skill_ids,
            performance_metrics=metrics,
            sample_id=sample_id,
            epoch=1,  # Simplified
            step=i,
            success=success,
        )

    return analyzer


def analyze_role_interactions(results: List[Dict[str, Any]]) -> InteractionTracer:
    """Analyze role interactions from results."""
    print("🔄 Analyzing role interactions...")

    tracer = InteractionTracer()

    for i, result in enumerate(results):
        # Extract available interaction data
        sample_id = result.get("sample_id", f"sample_{i}")
        question = result.get("question", "")
        context = result.get("context", "")

        # Mock interaction data (would need actual role outputs)
        agent_output = AgentOutput(
            reasoning=result.get("reasoning", ""),
            final_answer=result.get("prediction", ""),
            skill_ids=result.get("skill_ids", []),
            raw={},
        )

        reflector_output = ReflectorOutput(
            reasoning="",
            error_identification="",
            root_cause_analysis="",
            correct_approach="",
            key_insight="",
            skill_tags=[],
            raw=result.get("reflection", {}),
        )

        skill_manager_output = SkillManagerOutput(
            update=None,
            raw=result.get("skill_manager_output", {}),  # Would need actual update
        )

        # Record simplified interaction
        # tracer.record_interaction(...) would need actual objects

    return tracer


def perform_comparative_analysis(baseline_file: str, ace_file: str) -> Dict[str, Any]:
    """Perform comparative analysis between baseline and ACE results."""
    print("📊 Performing comparative analysis...")

    baseline_results = load_ace_results(baseline_file)
    ace_results = load_ace_results(ace_file)

    # Calculate metrics for both
    def calculate_avg_metrics(results):
        metrics_sum = {}
        count = 0
        for result in results:
            metrics = result.get("metrics", {})
            for metric, value in metrics.items():
                metrics_sum[metric] = metrics_sum.get(metric, 0) + value
            count += 1

        return {metric: total / count for metric, total in metrics_sum.items()}

    baseline_avg = calculate_avg_metrics(baseline_results)
    ace_avg = calculate_avg_metrics(ace_results)

    # Calculate improvements
    improvements = {}
    for metric in baseline_avg:
        if metric in ace_avg:
            improvement = ace_avg[metric] - baseline_avg[metric]
            relative_improvement = (
                improvement / baseline_avg[metric] if baseline_avg[metric] != 0 else 0
            )
            improvements[metric] = {
                "absolute": improvement,
                "relative": relative_improvement,
                "baseline": baseline_avg[metric],
                "ace": ace_avg[metric],
            }

    return {
        "baseline_avg": baseline_avg,
        "ace_avg": ace_avg,
        "improvements": improvements,
        "baseline_count": len(baseline_results),
        "ace_count": len(ace_results),
    }


def generate_insights(
    evolution_tracker: Optional[EvolutionTracker] = None,
    attribution_analyzer: Optional[AttributionAnalyzer] = None,
    interaction_tracer: Optional[InteractionTracer] = None,
    comparison: Optional[Dict] = None,
) -> List[str]:
    """Generate key insights from analysis."""
    insights = []

    if evolution_tracker:
        summary = evolution_tracker.get_evolution_summary()
        if summary.get("survival_rate", 0) > 0.8:
            insights.append(
                "🎯 High strategy survival rate indicates effective strategy curation"
            )
        if summary.get("skill_growth", 0) > 10:
            insights.append(
                "📈 Significant strategy expansion suggests active learning"
            )

        patterns = evolution_tracker.identify_learning_patterns()
        if patterns.get("performance_jumps"):
            insights.append(
                f"⚡ {len(patterns['performance_jumps'])} performance breakthroughs detected"
            )

    if attribution_analyzer:
        report = attribution_analyzer.generate_attribution_report()
        top_contributors = report["top_contributors"][:3]
        if top_contributors:
            best_strategy = top_contributors[0]
            insights.append(
                f"🏆 Best strategy: {best_strategy['skill_id']} (score: {best_strategy['attribution_score']:.3f})"
            )

        synergies = report["strategy_synergies"][:3]
        if synergies:
            insights.append(f"🤝 {len(synergies)} strong strategy synergies identified")

    if interaction_tracer:
        report = interaction_tracer.generate_interaction_report()
        if report["summary"]["feedback_loops_total"] > 0:
            insights.append(
                f"🔄 {report['summary']['feedback_loops_total']} feedback loops detected"
            )

    if comparison:
        for metric, data in comparison["improvements"].items():
            if data["relative"] > 0.1:  # 10% improvement
                insights.append(
                    f"📊 {metric.upper()}: {data['relative']:+.1%} improvement over baseline"
                )

    return insights


def print_summary_report(
    evolution_tracker: Optional[EvolutionTracker] = None,
    attribution_analyzer: Optional[AttributionAnalyzer] = None,
    interaction_tracer: Optional[InteractionTracer] = None,
    comparison: Optional[Dict] = None,
    insights: Optional[List[str]] = None,
):
    """Print a comprehensive summary report."""
    print("\n" + "=" * 60)
    print("🧠 ACE EXPLAINABILITY ANALYSIS REPORT")
    print("=" * 60)

    if evolution_tracker:
        summary = evolution_tracker.get_evolution_summary()
        print(f"\n📈 EVOLUTION ANALYSIS:")
        print(f"  Total Strategies: {summary.get('total_strategies', 0)}")
        print(f"  Survival Rate: {summary.get('survival_rate', 0):.1%}")
        print(f"  Strategy Growth: {summary.get('skill_growth', 0):+d}")

    if attribution_analyzer:
        report = attribution_analyzer.generate_attribution_report()
        print(f"\n🎯 ATTRIBUTION ANALYSIS:")
        print(f"  Active Strategies: {report['summary']['active_skills']}")
        print(
            f"  Avg Attribution Score: {report['summary']['avg_attribution_score']:.3f}"
        )
        print(f"  Strategy Synergies: {len(report['strategy_synergies'])}")

        top_contributors = report["top_contributors"][:5]
        if top_contributors:
            print(f"\n🏆 TOP CONTRIBUTORS:")
            for i, contributor in enumerate(top_contributors, 1):
                print(
                    f"  {i}. {contributor['skill_id'][:12]} (Score: {contributor['attribution_score']:.3f})"
                )

    if interaction_tracer:
        report = interaction_tracer.generate_interaction_report()
        print(f"\n🔄 INTERACTION ANALYSIS:")
        print(f"  Total Interactions: {report['summary']['total_interactions']}")
        print(f"  Decision Chains: {report['summary']['decision_chains_identified']}")
        print(f"  Feedback Loops: {report['summary']['feedback_loops_total']}")

    if comparison:
        print(f"\n📊 COMPARATIVE ANALYSIS:")
        for metric, data in comparison["improvements"].items():
            print(
                f"  {metric.upper()}: {data['baseline']:.3f} → {data['ace']:.3f} ({data['relative']:+.1%})"
            )

    if insights:
        print(f"\n🔍 KEY INSIGHTS:")
        for insight in insights:
            print(f"  {insight}")

    print("\n" + "=" * 60)


def run_live_analysis(args: argparse.Namespace) -> None:
    """Run live analysis during ACE adaptation."""
    print("🔴 LIVE ANALYSIS MODE")
    print("This would run real-time analysis during ACE adaptation")
    print("Implementation requires integration with the benchmark runner")
    # TODO: Implement live analysis integration


def main():
    """Main execution function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    evolution_tracker = None
    attribution_analyzer = None
    interaction_tracer = None
    comparison = None

    try:
        if args.live:
            run_live_analysis(args)
            return

        elif args.compare:
            # Comparative analysis mode
            comparison = perform_comparative_analysis(args.compare[0], args.compare[1])
            print(f"✅ Comparative analysis completed")

        elif args.results:
            # Analysis mode
            results = load_ace_results(args.results)
            print(f"📊 Loaded {len(results)} results from {args.results}")

            # Run requested analyses
            if args.focus in ["evolution", "all"]:
                evolution_tracker = analyze_evolution_patterns(results)
                evolution_tracker.export_timeline(
                    output_dir / "evolution_timeline.json"
                )

            if args.focus in ["attribution", "all"]:
                attribution_analyzer = analyze_strategy_attribution(results)
                attribution_analyzer.export_analysis(
                    output_dir / "attribution_analysis.json"
                )

            if args.focus in ["interactions", "all"]:
                interaction_tracer = analyze_role_interactions(results)
                if interaction_tracer.interactions:
                    interaction_tracer.export_traces(
                        output_dir / "interaction_traces.json"
                    )

            print(f"✅ Analysis completed")

        # Generate insights
        insights = generate_insights(
            evolution_tracker, attribution_analyzer, interaction_tracer, comparison
        )

        # Print summary report
        print_summary_report(
            evolution_tracker,
            attribution_analyzer,
            interaction_tracer,
            comparison,
            insights,
        )

        # Generate visualizations and HTML report if requested
        if args.generate_html:
            visualizer = ExplainabilityVisualizer()
            html_path = visualizer.generate_html_report(
                evolution_tracker=evolution_tracker,
                attribution_analyzer=attribution_analyzer,
                interaction_tracer=interaction_tracer,
                output_path=output_dir / "explainability_report.html",
                include_plots=args.include_plots,
            )
            print(f"📋 HTML report generated: {html_path}")

        # Generate individual plots if requested
        if args.include_plots:
            visualizer = ExplainabilityVisualizer()

            if evolution_tracker:
                plot_path = output_dir / "skillbook_evolution.png"
                visualizer.plot_skillbook_evolution(evolution_tracker, plot_path)
                print(f"📈 Evolution plot saved: {plot_path}")

            if attribution_analyzer:
                plot_path = output_dir / "attribution_analysis.png"
                visualizer.plot_skill_attribution(
                    attribution_analyzer, save_path=plot_path
                )
                print(f"🎯 Attribution plot saved: {plot_path}")

            if interaction_tracer:
                plot_path = output_dir / "interaction_heatmap.png"
                visualizer.create_interaction_heatmap(interaction_tracer, plot_path)
                print(f"🔄 Interaction heatmap saved: {plot_path}")

        print(f"\n📁 All outputs saved to: {output_dir}")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
