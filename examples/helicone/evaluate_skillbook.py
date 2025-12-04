#!/usr/bin/env python3
"""
Skillbook Evaluation with Blinded Judging

Evaluates ACE skillbook effectiveness by comparing baseline (no skillbook) vs
ACE (with skillbook) responses on held-out test samples. Uses blinded judging
to eliminate bias.

Methodology:
1. Load last N samples from JSONL (test set, not used in training)
2. Generate responses with Sonnet 4.0:
   - Baseline: original system prompt only
   - ACE: original system prompt + skillbook
3. Judge each response independently with Sonnet 4.5 (blinded)
4. Compare success rates

Output:
- evaluation_results.json - detailed per-sample results
- evaluation_summary.json - aggregated metrics
"""

import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dotenv import load_dotenv


# Load environment variables
load_dotenv(override=True)

# Disable Opik logging for cleaner output (must be set before importing ace)
os.environ["OPIK_ENABLED"] = "false"

from ace import Skillbook, LiteLLMClient


def load_test_samples(jsonl_path: str, n_samples: int = 20) -> List[Dict[str, Any]]:
    """
    Load last N samples from JSONL file as test set.

    Args:
        jsonl_path: Path to ace_convex_samples.jsonl
        n_samples: Number of samples to load from end

    Returns:
        List of test samples
    """
    print(f"📂 Loading test set: last {n_samples} samples from {jsonl_path}")

    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))

    # Take last N samples (held-out test set)
    test_samples = samples[-n_samples:]

    print(f"✅ Loaded {len(test_samples)} test samples\n")
    return test_samples


def extract_ground_truth(response_text: str) -> Dict[str, str]:
    """
    Extract buggy code, error message, and fixed code from response text.

    The response format is:
    ## Initial Attempt
    ```typescript
    [buggy code]
    ```
    ## Error Encountered
    ```
    [error message]
    ```
    ## Fixed Version
    ```typescript
    [fixed code]
    ```
    """
    result = {"buggy_code": "", "error_msg": "", "fixed_code": ""}

    # Split into sections
    sections = response_text.split("##")

    for section in sections:
        section = section.strip()

        if section.startswith("Initial Attempt"):
            # Extract code between first ``` and next ```
            code_blocks = section.split("```")
            if len(code_blocks) >= 3:
                result["buggy_code"] = (
                    code_blocks[1].replace("typescript\n", "").strip()
                )

        elif section.startswith("Error Encountered") or section.startswith("Error"):
            code_blocks = section.split("```")
            if len(code_blocks) >= 3:
                result["error_msg"] = code_blocks[1].strip()

        elif section.startswith("Fixed Version") or section.startswith("Fixed"):
            code_blocks = section.split("```")
            if len(code_blocks) >= 3:
                result["fixed_code"] = (
                    code_blocks[1].replace("typescript\n", "").strip()
                )

    return result


def filter_skillbook_skills(skillbook: Skillbook, categories: List[str]) -> str:
    """
    Filter skillbook skills to only those matching sample categories.

    Args:
        skillbook: Full learned skillbook
        categories: Sample's error categories

    Returns:
        Formatted string of relevant skills
    """
    relevant_skills = []

    for skill in skillbook.skills():
        # Check if skill section matches any category
        skill_section = skill.section.lower()
        if any(
            cat.lower() in skill_section or skill_section in cat.lower()
            for cat in categories
        ):
            relevant_skills.append(skill)

    if not relevant_skills:
        # If no exact matches, include top 5 skills by score
        all_skills = sorted(
            skillbook.skills(), key=lambda s: s.helpful - s.harmful, reverse=True
        )
        relevant_skills = all_skills[:5]

    # Format skills
    formatted = "## Learned Patterns:\n\n"
    for skill in relevant_skills[:10]:  # Max 10 skills
        score = skill.helpful - skill.harmful
        formatted += f"- {skill.content} [score: {score:+d}]\n\n"

    return formatted


def generate_responses(
    question: str,
    system_prompt: str,
    skillbook_text: str,
    model: str = "claude-sonnet-4-20250514",
) -> Tuple[str, str]:
    """
    Generate baseline and ACE responses.

    Args:
        question: The question to answer
        system_prompt: Original system prompt from logs
        skillbook_text: Filtered skillbook skills
        model: Model to use (Sonnet 4.0)

    Returns:
        (baseline_response, ace_response)
    """
    llm = LiteLLMClient(model=model, temperature=0.7, max_tokens=4000)

    # Baseline: system prompt only
    baseline_response = llm.complete(prompt=question, system=system_prompt)

    # ACE: system prompt + skillbook
    ace_system_prompt = f"{system_prompt}\n\n{skillbook_text}"
    ace_response = llm.complete(prompt=question, system=ace_system_prompt)

    return baseline_response.text, ace_response.text


def judge_response(
    question: str,
    response: str,
    ground_truth: Dict[str, str],
    judge_model: str = "claude-sonnet-4-5-20250929",
) -> Optional[Dict[str, Any]]:
    """
    Judge whether a response would lead to error (blinded evaluation).

    Args:
        question: The question
        response: Model's response to evaluate
        ground_truth: Dict with buggy_code, error_msg, fixed_code
        judge_model: Judge model (Sonnet 4.5)

    Returns:
        Dict with verdict (SUCCESS/FAIL/UNCERTAIN) and reasoning, or None if judge fails
    """
    judge_llm = LiteLLMClient(model=judge_model, temperature=0, max_tokens=2000)

    judge_prompt = f"""You are evaluating a code suggestion against a known error pattern.

<question>
{question}
</question>

<model_response>
{response}
</model_response>

<known_error>
<buggy_code>
{ground_truth["buggy_code"]}
</buggy_code>
<error_message>
{ground_truth["error_msg"]}
</error_message>
<correct_fix>
{ground_truth["fixed_code"]}
</correct_fix>
</known_error>

Evaluate whether the model's response would lead to the same error:

1. Does the response contain or suggest the buggy code pattern?
2. Would following this response lead to the same error?
3. Does the response provide guidance that avoids this error?

Respond with ONLY a JSON object, no markdown formatting, no explanation:
{{"verdict": "SUCCESS/FAIL/UNCERTAIN", "reasoning": "one sentence explanation"}}

SUCCESS = Response avoids the error pattern
FAIL = Response contains or would lead to the error
UNCERTAIN = Cannot determine clearly"""

    try:
        judgment_text = judge_llm.complete(prompt=judge_prompt).text.strip()

        # Extract JSON from response (handle markdown code blocks and extra text)
        json_text = judgment_text

        # Remove markdown code blocks if present
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0].strip()

        # Try to find JSON object in the text
        import re

        json_match = re.search(r"\{[^}]+\}", json_text)
        if json_match:
            json_text = json_match.group(0)

        # Parse JSON
        judgment = json.loads(json_text)

        # Validate required fields
        if "verdict" not in judgment:
            raise ValueError("Missing 'verdict' field in JSON")

        # Normalize verdict to uppercase
        judgment["verdict"] = judgment["verdict"].upper()

        # Add default fields if missing
        judgment.setdefault("reasoning", "No reasoning provided")
        judgment.setdefault("would_lead_to_error", "uncertain")
        judgment.setdefault("contains_buggy_pattern", "no")
        judgment.setdefault("provides_correct_guidance", "uncertain")

        return judgment

    except Exception:
        # Judge failed - return None to skip this sample
        return None


def print_interim_summary(
    results: List[Dict[str, Any]], total_attempted: int, skipped: int
) -> None:
    """Print summary statistics for current results."""
    if not results:
        print("\n⚠️  No samples evaluated yet\n")
        return

    baseline_verdicts = [r["baseline_judgment"]["verdict"] for r in results]
    ace_verdicts = [r["ace_judgment"]["verdict"] for r in results]

    evaluated = len(results)
    baseline_success = baseline_verdicts.count("SUCCESS")
    ace_success = ace_verdicts.count("SUCCESS")
    baseline_rate = baseline_success / evaluated * 100 if evaluated > 0 else 0
    ace_rate = ace_success / evaluated * 100 if evaluated > 0 else 0
    delta = ace_rate - baseline_rate

    print("\n" + "=" * 70)
    print(
        f"📊 Summary (after {total_attempted} samples: {evaluated} evaluated, {skipped} skipped)"
    )
    print("=" * 70)
    print(f"  Baseline: {baseline_success}/{evaluated} SUCCESS ({baseline_rate:.1f}%)")
    print(f"  ACE:      {ace_success}/{evaluated} SUCCESS ({ace_rate:.1f}%)")
    print(f"  Improvement: {delta:+.1f}% ({ace_success - baseline_success:+d} samples)")
    print("=" * 70 + "\n")


def evaluate_skillbook(
    test_samples: List[Dict[str, Any]], skillbook: Skillbook, output_dir: Path
) -> Dict[str, Any]:
    """
    Run full evaluation: generate responses and judge them blindly.

    Args:
        test_samples: List of test samples
        skillbook: Learned skillbook
        output_dir: Directory to save results

    Returns:
        Evaluation results with metrics
    """
    print("🎯 Starting Skillbook Evaluation")
    print("=" * 70)
    print(f"Test samples: {len(test_samples)}")
    print(f"Skillbook skills: {len(skillbook.skills())}\n")

    results = []
    skipped_samples = 0

    for i, sample in enumerate(test_samples, 1):
        print(f"[{i}/{len(test_samples)}] Evaluating sample...", end=" ")

        question = sample["question"]
        categories = sample.get("metadata", {}).get("categories", [])
        ground_truth = extract_ground_truth(sample["response"])

        # Use a default system prompt (since we may not have original)
        system_prompt = "You are an expert Convex backend developer helping debug and fix code issues."

        # Filter skillbook for relevant skills
        skillbook_text = filter_skillbook_skills(skillbook, categories)

        # Generate responses
        baseline_resp, ace_resp = generate_responses(
            question, system_prompt, skillbook_text
        )

        # Judge responses (blinded - random order)
        responses_to_judge = [
            {"condition": "baseline", "response": baseline_resp},
            {"condition": "ace", "response": ace_resp},
        ]
        random.shuffle(responses_to_judge)

        judgments = []
        judge_failed = False
        for resp_item in responses_to_judge:
            judgment = judge_response(question, resp_item["response"], ground_truth)
            if judgment is None:
                judge_failed = True
                break
            judgment["condition"] = resp_item["condition"]
            judgments.append(judgment)

        # Skip this sample if judge failed
        if judge_failed:
            print("⊗ Skipped (judge failed)")
            skipped_samples += 1
            continue

        # Store results
        result = {
            "sample_id": i,
            "question": question[:200],  # Truncate for readability
            "categories": categories,
            "ground_truth": ground_truth,
            "baseline_response": baseline_resp[:500],
            "ace_response": ace_resp[:500],
            "baseline_judgment": next(
                j for j in judgments if j["condition"] == "baseline"
            ),
            "ace_judgment": next(j for j in judgments if j["condition"] == "ace"),
        }
        results.append(result)

        # Print results immediately
        print(f"✓ Done")
        print(f"\n  Question: {question[:150]}...")
        print(f"\n  📝 Baseline Response:")
        print(f"     {baseline_resp[:300]}...")
        print(f"     Verdict: {result['baseline_judgment']['verdict']}")
        print(f"\n  🎯 ACE Response (with skillbook):")
        print(f"     {ace_resp[:300]}...")
        print(f"     Verdict: {result['ace_judgment']['verdict']}")
        print(f"\n" + "-" * 70 + "\n")

        # Print interim summary every 10 samples
        if i % 10 == 0:
            print_interim_summary(results, i, skipped_samples)

    # Aggregate metrics
    baseline_verdicts = [r["baseline_judgment"]["verdict"] for r in results]
    ace_verdicts = [r["ace_judgment"]["verdict"] for r in results]

    # Calculate success rates (handle empty results)
    evaluated_count = len(results)
    baseline_success_rate = (
        baseline_verdicts.count("SUCCESS") / evaluated_count * 100
        if evaluated_count > 0
        else 0
    )
    ace_success_rate = (
        ace_verdicts.count("SUCCESS") / evaluated_count * 100
        if evaluated_count > 0
        else 0
    )
    success_rate_delta = (
        (ace_verdicts.count("SUCCESS") - baseline_verdicts.count("SUCCESS"))
        / evaluated_count
        * 100
        if evaluated_count > 0
        else 0
    )

    metrics = {
        "total_samples_attempted": len(test_samples),
        "samples_evaluated": evaluated_count,
        "samples_skipped": skipped_samples,
        "baseline": {
            "success": baseline_verdicts.count("SUCCESS"),
            "fail": baseline_verdicts.count("FAIL"),
            "uncertain": baseline_verdicts.count("UNCERTAIN"),
            "success_rate": baseline_success_rate,
        },
        "ace": {
            "success": ace_verdicts.count("SUCCESS"),
            "fail": ace_verdicts.count("FAIL"),
            "uncertain": ace_verdicts.count("UNCERTAIN"),
            "success_rate": ace_success_rate,
        },
        "improvement": {
            "success_gain": ace_verdicts.count("SUCCESS")
            - baseline_verdicts.count("SUCCESS"),
            "fail_reduction": baseline_verdicts.count("FAIL")
            - ace_verdicts.count("FAIL"),
            "success_rate_delta": success_rate_delta,
        },
    }

    # Save detailed results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save summary
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {"metrics": metrics, "results": results}


def main():
    """Main evaluation script."""

    print("\n" + "=" * 70)
    print("ACE Skillbook Evaluation - Blinded Judging")
    print("=" * 70 + "\n")

    # Configuration
    DATA_PATH = "../../.private/helicone/ace_convex_training/ace_convex_samples.jsonl"
    SKILLBOOK_PATH = (
        "checkpoints/convex_checkpoint_20.json"  # Use checkpoint after 20 samples
    )
    N_TEST_SAMPLES = 200
    OUTPUT_DIR = "evaluation_results"

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Please set ANTHROPIC_API_KEY in your .env file")
        return

    script_dir = Path(__file__).parent
    data_path = script_dir / DATA_PATH
    skillbook_path = script_dir / SKILLBOOK_PATH
    output_dir = script_dir / OUTPUT_DIR
    output_dir.mkdir(exist_ok=True)

    # Check files exist
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}\n")
        return

    if not skillbook_path.exists():
        print(f"❌ Skillbook not found: {skillbook_path}")
        print(f"   Run convex_training.py first to train skillbook\n")
        return

    # Load test samples
    test_samples = load_test_samples(str(data_path), n_samples=N_TEST_SAMPLES)

    # Load skillbook
    print(f"📚 Loading skillbook from: {skillbook_path}")
    skillbook = Skillbook.load_from_file(str(skillbook_path))
    print(f"✅ Loaded skillbook with {len(skillbook.skills())} skills\n")

    # Run evaluation
    eval_results = evaluate_skillbook(test_samples, skillbook, output_dir)

    # Print final summary
    print("\n" + "=" * 70)
    print("📊 FINAL EVALUATION RESULTS")
    print("=" * 70)

    metrics = eval_results["metrics"]

    print(f"\n**Overview:**")
    print(f"  Total samples attempted: {metrics['total_samples_attempted']}")
    print(f"  Successfully evaluated:  {metrics['samples_evaluated']}")
    if metrics["samples_skipped"] > 0:
        print(f"  Skipped (judge errors):  {metrics['samples_skipped']}")

    print(f"\n**Performance Comparison:**")
    print(
        f"  Baseline (no skillbook): {metrics['baseline']['success']}/{metrics['samples_evaluated']} SUCCESS ({metrics['baseline']['success_rate']:.1f}%)"
    )
    print(
        f"  ACE (with skillbook):    {metrics['ace']['success']}/{metrics['samples_evaluated']} SUCCESS ({metrics['ace']['success_rate']:.1f}%)"
    )

    print(f"\n**Improvement:**")
    print(f"  Success rate delta: {metrics['improvement']['success_rate_delta']:+.1f}%")
    print(
        f"  Additional successes: {metrics['improvement']['success_gain']:+d} samples"
    )
    print(f"  Failures reduced: {metrics['improvement']['fail_reduction']:+d} samples")

    print(f"\n**Detailed Breakdown:**")
    print(
        f"  Baseline - Success: {metrics['baseline']['success']}, Fail: {metrics['baseline']['fail']}, Uncertain: {metrics['baseline']['uncertain']}"
    )
    print(
        f"  ACE      - Success: {metrics['ace']['success']}, Fail: {metrics['ace']['fail']}, Uncertain: {metrics['ace']['uncertain']}"
    )

    print(f"\n💾 Detailed results saved to:")
    print(f"   {output_dir}/evaluation_results.json")
    print(f"   {output_dir}/evaluation_summary.json")

    print("\n" + "=" * 70)
    print("✨ Evaluation complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
