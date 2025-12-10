#!/usr/bin/env python3
"""
Analyze ACE results to understand sample counting and learning behavior.
"""

import json
import sys
from pathlib import Path


def analyze_ace_run(detailed_results_file):
    """Analyze a detailed ACE results file."""
    print("🔍 ACE RESULTS ANALYSIS")
    print("=" * 50)

    with open(detailed_results_file, "r") as f:
        data = json.load(f)

    print(f"📊 SUMMARY:")
    print(f"  Benchmark: {data['benchmark']}")
    print(f"  Model: {data['model']}")
    print(f"  Total samples reported: {data['samples_evaluated']}")
    print(f"  Actual results entries: {len(data['results'])}")

    # Analyze epochs by looking at repeated sample IDs
    sample_occurrences = {}
    for result in data["results"]:
        sample_id = result["sample_id"]
        if sample_id not in sample_occurrences:
            sample_occurrences[sample_id] = 0
        sample_occurrences[sample_id] += 1

    unique_samples = len(sample_occurrences)
    max_epochs = max(sample_occurrences.values())

    print(f"\n📈 EPOCH ANALYSIS:")
    print(f"  Unique samples: {unique_samples}")
    print(f"  Max epochs per sample: {max_epochs}")
    print(f"  Total evaluations: {sum(sample_occurrences.values())}")
    print(
        f"  Sample counting logic: {unique_samples} samples × {max_epochs} epochs = {unique_samples * max_epochs}"
    )

    # Show ACE learning components
    print(f"\n🧠 ACE LEARNING VERIFICATION:")
    first_result = data["results"][0]

    has_reflection = "reflection" in first_result
    has_skill_manager = "skill_manager_output" in first_result

    print(f"  ✅ Reflection component: {'Working' if has_reflection else 'Missing'}")
    print(
        f"  ✅ SkillManager component: {'Working' if has_skill_manager else 'Missing'}"
    )

    if has_reflection:
        reflection = first_result["reflection"]
        print(f"  ✅ Reflection fields: {list(reflection.keys())}")

    if has_skill_manager:
        skill_manager = first_result["skill_manager_output"]
        operations = skill_manager.get("operations", [])
        print(f"  ✅ SkillManager operations: {len(operations)} operations")
        if operations:
            op_types = [op["type"] for op in operations]
            print(f"      Operation types: {set(op_types)}")

    # Show learning progression
    print(f"\n📚 LEARNING PROGRESSION:")

    # Group by sample ID to show epochs
    epochs_by_sample = {}
    for result in data["results"]:
        sample_id = result["sample_id"]
        if sample_id not in epochs_by_sample:
            epochs_by_sample[sample_id] = []
        epochs_by_sample[sample_id].append(result)

    # Show first sample's progression
    first_sample_id = list(epochs_by_sample.keys())[0]
    first_sample_epochs = epochs_by_sample[first_sample_id]

    print(f"  Sample: {first_sample_id}")
    print(f"  Epochs: {len(first_sample_epochs)}")

    for i, epoch_result in enumerate(first_sample_epochs):
        f1 = epoch_result["metrics"]["f1"]
        print(f"    Epoch {i+1}: F1={f1:.3f}")

        # Show if any learning artifacts exist
        if "used_skills" in epoch_result:
            skills = epoch_result.get("used_skills", [])
            print(f"              Used {len(skills)} learned strategies")

    # Performance comparison across epochs
    print(f"\n📈 PERFORMANCE ACROSS EPOCHS:")

    epoch_1_f1s = []
    epoch_2_f1s = []

    for sample_id, epochs in epochs_by_sample.items():
        if len(epochs) >= 1:
            epoch_1_f1s.append(epochs[0]["metrics"]["f1"])
        if len(epochs) >= 2:
            epoch_2_f1s.append(epochs[1]["metrics"]["f1"])

    if epoch_1_f1s:
        avg_epoch_1 = sum(epoch_1_f1s) / len(epoch_1_f1s)
        print(f"  Epoch 1 average F1: {avg_epoch_1:.3f}")

    if epoch_2_f1s:
        avg_epoch_2 = sum(epoch_2_f1s) / len(epoch_2_f1s)
        improvement = avg_epoch_2 - avg_epoch_1
        print(f"  Epoch 2 average F1: {avg_epoch_2:.3f}")
        print(f"  Improvement: {improvement:+.3f}")

    return {
        "unique_samples": unique_samples,
        "max_epochs": max_epochs,
        "total_evaluations": len(data["results"]),
        "has_learning": has_reflection and has_skill_manager,
        "epoch_1_avg": avg_epoch_1 if epoch_1_f1s else 0,
        "epoch_2_avg": avg_epoch_2 if epoch_2_f1s else 0,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_ace_results.py <detailed_results.json>")

        # Find most recent detailed results
        results_dir = Path("benchmark_results")
        detailed_files = list(results_dir.glob("*detailed.json"))
        if detailed_files:
            latest_file = max(detailed_files, key=lambda f: f.stat().st_mtime)
            print(f"\nAnalyzing most recent file: {latest_file}")
            analyze_ace_run(latest_file)
        else:
            print("No detailed results files found")
        return

    results_file = sys.argv[1]
    analyze_ace_run(results_file)


if __name__ == "__main__":
    main()
