#!/usr/bin/env python3
"""
Agentic System Prompting Example

Demonstrates using ACE's OfflineACE adapter to analyze past agent
conversations and generate system prompt improvements.

Usage:
    python agentic_system_prompting.py /path/to/traces
    python agentic_system_prompting.py /path/to/traces --model gpt-4o --epochs 2
    python agentic_system_prompting.py /path/to/traces --input-skillbook existing.json

Options:
    traces_dir              Path to directory containing .md or .toon trace files
    --model, -m             LLM model for analysis (default: claude-haiku-4-5-20251001)
    --epochs, -e            Number of training epochs (default: 1)
    --threshold, -t         Deduplication similarity threshold 0.0-1.0 (default: 0.7)
    --input-skillbook, -i   Path to existing skillbook to continue from
    --output-dir, -o        Output directory for results (default: script directory)

Requirements:
    - LLM API key for analysis (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY)
    - OPENAI_API_KEY for deduplication (uses OpenAI embeddings to detect similar skills)
"""

import argparse
import os
from pathlib import Path
from datetime import datetime
from itertools import groupby
from typing import List, Dict, Any

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from ace import (
    Skillbook,
    Sample,
    OfflineACE,
    Reflector,
    SkillManager,
    ReplayAgent,
    SimpleEnvironment,
    DeduplicationConfig,
)
from ace.llm_providers.litellm_client import LiteLLMClient, LiteLLMConfig
from ace.prompt_manager import PromptManager, wrap_skillbook_for_external_agent


def load_conversations(conversations_dir: Path) -> List[Dict[str, Any]]:
    """Load all .md and .toon conversation files from directory."""
    if not conversations_dir.exists():
        print(f"Directory not found: {conversations_dir}")
        return []

    conversations = []

    # Load markdown files
    for file_path in sorted(conversations_dir.glob("*.md")):
        try:
            content = file_path.read_text(encoding="utf-8")
            conversations.append({"filename": file_path.name, "content": content})
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    # Load TOON files (fed directly to LLM as raw text)
    for file_path in sorted(conversations_dir.glob("*.toon")):
        try:
            content = file_path.read_text(encoding="utf-8")
            conversations.append({"filename": file_path.name, "content": content})
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    print(f"Loaded {len(conversations)} conversations")
    return conversations


def create_samples(conversations: List[Dict[str, Any]]) -> List[Sample]:
    """Convert conversations to ACE samples."""
    samples = []

    for conv in conversations:
        sample = Sample(
            question="-",
            ground_truth="",
            metadata={"response": conv["content"]},
        )
        samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Analyze agent conversations and generate system prompt improvements"
    )
    parser.add_argument(
        "traces_dir",
        type=Path,
        help="Path to directory containing .md or .toon trace files",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="claude-haiku-4-5-20251001",
        help="LLM model for analysis",
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.7,
        help="Deduplication similarity threshold (0.0-1.0)",
    )
    parser.add_argument(
        "-i",
        "--input-skillbook",
        type=Path,
        default=None,
        help="Path to existing skillbook to continue from",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results",
    )
    args = parser.parse_args()

    CONVERSATIONS_DIR = args.traces_dir
    LLM_MODEL = args.model
    EPOCHS = args.epochs
    DEDUPLICATOR_SIMILARITY_THRESHOLD = args.threshold
    INPUT_SKILLBOOK = args.input_skillbook

    SCRIPT_DIR = args.output_dir or Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_SKILLBOOK = SCRIPT_DIR / f"skillbook_{timestamp}.json"

    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("WARNING: OPENAI_API_KEY required for deduplication embeddings!")
        return

    # Load conversations
    conversations = load_conversations(CONVERSATIONS_DIR)
    if not conversations:
        print("\nTo use this example:")
        print(f"  1. Create directory: {CONVERSATIONS_DIR}/")
        print(
            f"  2. Add .md or .toon trace files to that directory (Use the convert.py script to convert JSON to TOON)"
        )
        return

    samples = create_samples(conversations)
    print(f"Created {len(samples)} samples")

    # Initialize ACE components - load existing or create new skillbook
    if INPUT_SKILLBOOK and INPUT_SKILLBOOK.exists():
        skillbook = Skillbook.load_from_file(str(INPUT_SKILLBOOK))
        print(
            f"Loaded existing skillbook: {len(skillbook.skills())} skills from {INPUT_SKILLBOOK}"
        )
    else:
        skillbook = Skillbook()

    config = LiteLLMConfig(model=LLM_MODEL, max_tokens=8192, temperature=1)
    llm = LiteLLMClient(config=config)
    prompt_mgr = PromptManager()

    agent = ReplayAgent()
    reflector = Reflector(llm=llm, prompt_template=prompt_mgr.get_reflector_prompt())
    skill_manager = SkillManager(
        llm=llm, prompt_template=prompt_mgr.get_skill_manager_prompt(version="3.0")
    )

    # Deduplication uses OpenAI embeddings to detect and merge similar skills
    dedup_config = DeduplicationConfig(
        enabled=True,
        similarity_threshold=DEDUPLICATOR_SIMILARITY_THRESHOLD,
        embedding_model="text-embedding-3-small",
    )

    adapter = OfflineACE(
        skillbook=skillbook,
        agent=agent,
        reflector=reflector,
        skill_manager=skill_manager,
        dedup_config=dedup_config,
    )

    print(
        f"\nStarting analysis: {len(samples)} conversations, {EPOCHS} epoch(s), model={LLM_MODEL}"
    )

    start_time = datetime.now()
    results = adapter.run(
        samples=samples, environment=SimpleEnvironment(), epochs=EPOCHS
    )
    duration = (datetime.now() - start_time).total_seconds()

    # Save and display results
    adapter.skillbook.save_to_file(str(OUTPUT_SKILLBOOK))

    skills = adapter.skillbook.skills()
    print(f"\nCompleted in {duration:.1f}s")
    print(f"Analyzed: {len(results)} conversations")
    print(f"Generated: {len(skills)} skills")
    print(f"Saved to: {OUTPUT_SKILLBOOK}")

    # Save skills grouped by section in markdown format
    OUTPUT_SKILLS_MD = SCRIPT_DIR / f"skills_{timestamp}.md"
    with open(OUTPUT_SKILLS_MD, "w") as f:
        for section, section_skills in groupby(
            sorted(skills, key=lambda s: s.section), key=lambda s: s.section
        ):
            f.write(f"## {section}\n\n")
            for skill in section_skills:
                f.write(f"- {skill.content}\n")
                if skill.justification:
                    f.write(f"  Justification: {skill.justification}\n")
                if skill.evidence:
                    f.write(f"  Evidence: {skill.evidence}\n")
            f.write("\n")
    print(f"Skills: {OUTPUT_SKILLS_MD}")

    if skills:
        print("\nTop skills:")
        for i, skill in enumerate(
            sorted(skills, key=lambda s: s.helpful, reverse=True)[:5], 1
        ):
            print(f"  {i}. [{skill.section}] {skill.content[:80]}...")

    # Generate external agent injection file
    OUTPUT_INJECTION = SCRIPT_DIR / f"external_agent_injection_{timestamp}.txt"
    injection_text = wrap_skillbook_for_external_agent(adapter.skillbook)
    if injection_text:
        with open(OUTPUT_INJECTION, "w") as f:
            f.write(injection_text)
        print(f"External agent injection: {OUTPUT_INJECTION}")
    else:
        print("No skills generated - skipping external agent injection file")


if __name__ == "__main__":
    main()
