#!/usr/bin/env python3
"""
Analyze ACE benchmark logs to determine success/failure before API limit crash.

IMPORTANT CLARIFICATION:
- "Failed samples" in ACE = samples that threw EXCEPTIONS during processing
  (includes API limit errors, parse errors, etc.)
- "Successful samples" = samples fully processed (Agent + Reflector + SkillManager completed)
  This does NOT mean the model got the answer right!

In ACE training:
- If model gets answer RIGHT: Only Agent is called (1 LLM call), no learning triggered
- If model gets answer WRONG: Agent + Reflector + SkillManager are called (3 LLM calls)

When API limit is hit, all subsequent samples fail with InstructorRetryException.
"""

import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

LOG_DIR = Path("/local/home/wangni/agentic-context-engine/benchmark_logs")

def analyze_log(log_path: Path) -> dict:
    """Analyze a single log file."""
    content = log_path.read_text()
    lines = content.split('\n')

    stats = {
        'benchmark': log_path.stem.replace('_ace', '').replace('_full', ''),
        'log_file': log_path.name,
        'total_samples_loaded': 0,
        'train_samples': 0,
        'test_samples': 0,
        'split_ratio': 0.0,

        # Training phase
        'train_successful_calls': 0,  # LLM calls that succeeded
        'train_failed_samples': 0,  # Failed samples reported by ACE
        'train_api_limit_errors': 0,  # API limit errors during training

        # Testing phase
        'test_started': False,
        'test_api_limit_errors': 0,  # API limit errors during testing

        # Timing
        'first_timestamp': None,
        'last_success_timestamp': None,
        'first_api_limit_timestamp': None,

        # Completion status
        'training_completed': False,
        'crashed': False,
    }

    # Extract sample counts
    match = re.search(r'Loaded (\d+) samples', content)
    if match:
        stats['total_samples_loaded'] = int(match.group(1))

    match = re.search(r'Train/test split: (\d+) train, (\d+) test \(ratio: ([\d.]+)\)', content)
    if match:
        stats['train_samples'] = int(match.group(1))
        stats['test_samples'] = int(match.group(2))
        stats['split_ratio'] = float(match.group(3))

    # Check for training completed message
    match = re.search(r'Training completed with (\d+) failed samples out of (\d+) total attempts', content)
    if match:
        stats['train_failed_samples'] = int(match.group(1))
        total_attempts = int(match.group(2))
        stats['training_completed'] = True

    # Check if testing started
    if 'Testing on' in content or '🧪 Testing' in content:
        stats['test_started'] = True

    # Count successful LLM calls (completed calls before API limit)
    successful_calls = re.findall(r'Wrapper: Completed Call, calling success_handler', content)
    stats['train_successful_calls'] = len(successful_calls)

    # Count API limit errors
    api_limit_errors = re.findall(r'You have reached your specified API usage limits', content)
    stats['total_api_limit_errors'] = len(api_limit_errors)

    # Check if crashed
    if 'Traceback' in content and 'API usage limits' in content:
        stats['crashed'] = True

    # Extract timestamps
    timestamps = re.findall(r'\[92m(\d{2}:\d{2}:\d{2})', content)
    if timestamps:
        stats['first_timestamp'] = timestamps[0]

    # Find last successful call before API limit
    lines_with_success = []
    for i, line in enumerate(lines):
        if 'Completed Call, calling success_handler' in line:
            # Get timestamp from nearby lines
            for j in range(max(0, i-5), i+1):
                ts_match = re.search(r'\[92m(\d{2}:\d{2}:\d{2})', lines[j])
                if ts_match:
                    lines_with_success.append((i, ts_match.group(1)))
                    break

    if lines_with_success:
        stats['last_success_timestamp'] = lines_with_success[-1][1]
        stats['successful_api_calls_before_crash'] = len(lines_with_success)

    # Find first API limit error timestamp
    for i, line in enumerate(lines):
        if 'API usage limits' in line:
            for j in range(max(0, i-10), i+1):
                ts_match = re.search(r'\[92m(\d{2}:\d{2}:\d{2})', lines[j])
                if ts_match:
                    stats['first_api_limit_timestamp'] = ts_match.group(1)
                    break
            break

    # Count failed samples by type
    instructor_failures = re.findall(r'Failed to process sample (\d+)/(\d+).*InstructorRetryException', content)
    stats['instructor_retry_failures'] = len(instructor_failures)

    # Count non-API-limit warnings (actual task failures)
    all_warnings = re.findall(r'WARNING.*Failed to process sample', content)
    stats['total_warning_failures'] = len(all_warnings)

    return stats


def main():
    print("=" * 80)
    print("ACE BENCHMARK LOG ANALYSIS")
    print("=" * 80)

    # Find all ACE logs
    ace_logs = sorted(LOG_DIR.glob("*_ace.log"))

    if not ace_logs:
        print("No ACE logs found!")
        return

    for log_path in ace_logs:
        stats = analyze_log(log_path)

        print(f"\n{'=' * 80}")
        print(f"BENCHMARK: {stats['benchmark'].upper()}")
        print(f"{'=' * 80}")
        print(f"Log file: {stats['log_file']}")
        print(f"File size: {log_path.stat().st_size / 1024 / 1024:.2f} MB")
        print()

        print("📊 SAMPLE CONFIGURATION:")
        print(f"  Total samples loaded: {stats['total_samples_loaded']}")
        print(f"  Train samples: {stats['train_samples']}")
        print(f"  Test samples: {stats['test_samples']}")
        print(f"  Split ratio: {stats['split_ratio']}")
        print()

        print("🔄 TRAINING PHASE:")
        print(f"  Training completed: {'✓ Yes' if stats['training_completed'] else '✗ No'}")
        if stats['training_completed']:
            successful_train = stats['train_samples'] - stats['train_failed_samples']
            print(f"  Successful training samples: {successful_train}")
            print(f"  Failed training samples: {stats['train_failed_samples']}")
            print(f"  Success rate: {100 * successful_train / stats['train_samples']:.1f}%")
        print(f"  Total successful API calls: {stats.get('successful_api_calls_before_crash', stats['train_successful_calls'])}")
        print()

        print("🧪 TESTING PHASE:")
        print(f"  Testing started: {'✓ Yes' if stats['test_started'] else '✗ No'}")
        print()

        print("❌ CRASH INFORMATION:")
        print(f"  Crashed: {'✓ Yes' if stats['crashed'] else '✗ No'}")
        print(f"  Total API limit errors: {stats['total_api_limit_errors']}")
        print(f"  InstructorRetryException failures: {stats['instructor_retry_failures']}")
        print()

        print("⏱️  TIMESTAMPS:")
        print(f"  First timestamp: {stats['first_timestamp']}")
        print(f"  Last success: {stats['last_success_timestamp']}")
        print(f"  First API limit error: {stats['first_api_limit_timestamp']}")
        print()

        # Calculate estimates
        if stats['training_completed'] and stats['train_samples'] > 0:
            # ACE uses 3 LLM calls per sample (Agent + Reflector + SkillManager) for failed samples
            # and 1 LLM call for successful samples (Agent only)
            successful = stats['train_samples'] - stats['train_failed_samples']
            # Estimate: successful samples use ~1 call, failed samples trigger full pipeline
            estimated_calls_per_sample = stats.get('successful_api_calls_before_crash', 0) / stats['train_samples'] if stats['train_samples'] > 0 else 0
            print(f"📈 ESTIMATES:")
            print(f"  Estimated calls per sample: {estimated_calls_per_sample:.2f}")
            print(f"  Samples processed before API limit: ~{stats.get('successful_api_calls_before_crash', 0) // 3}")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Benchmark':<15} {'Train':<10} {'Test':<10} {'Processed':<12} {'API Errors':<12} {'Status':<15}")
    print("-" * 80)

    for log_path in ace_logs:
        stats = analyze_log(log_path)
        processed = stats['train_samples'] - stats['train_failed_samples'] if stats['training_completed'] else '?'
        status = "CRASHED" if stats['crashed'] else ("COMPLETE" if stats['training_completed'] else "UNKNOWN")
        print(f"{stats['benchmark']:<15} {stats['train_samples']:<10} {stats['test_samples']:<10} {str(processed):<12} {stats['total_api_limit_errors']:<12} {status:<15}")

    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print("""
WHAT HAPPENED:
- All 3 ACE benchmark runs were started in PARALLEL on Dec 15
- They ran successfully for ~1.5 hours (10:26 - 12:05)
- At ~12:05, the Anthropic API usage limit was hit
- All subsequent API calls failed with "API usage limits reached"
- The framework continued attempting samples, but they all failed immediately

CLARIFICATION ON "FAILED SAMPLES":
- In ACE, "failed samples" = samples that threw EXCEPTIONS (not wrong answers)
- The 689/890/11070 "failures" are mostly API limit errors, NOT model mistakes
- The 171/165/163 "successful" samples = samples FULLY PROCESSED before crash

WHAT WAS ACTUALLY PROCESSED:
- finer_ord: 171 training samples processed (19.9% of 860)
- gsm8k: 165 training samples processed (15.6% of 1055)
- mmlu: 163 training samples processed (1.5% of 11233)

TESTING PHASE:
- All 3 benchmarks started the testing phase
- Testing immediately crashed due to API limit (no test samples evaluated)

NO RESULTS SAVED:
- No ACE results were saved (runs crashed before completion)
- No skillbooks were checkpointed
- Only baseline results exist in benchmark_results/
""")


if __name__ == "__main__":
    main()
