#!/usr/bin/env python3
"""
Agentic System Prompting Example

Demonstrates using ACE's OfflineACE adapter to analyze past agent
conversations and generate system prompt improvements.

Usage:
    1. Export/convert your agent conversations to .md or .toon files
       - To convert JSON to TOON, use the toon library (included with ACE):
         import toon
         toon_str = toon.encode(your_json_data)
       - Or use the CLI: toon input.json -o output.toon
    2. Place them in a directory
    3. Update CONVERSATIONS_DIR path below
    4. Run: python agentic_system_prompting.py
    5. View the generated suggestions with reasoning and evidence in the skills_{timestamp}.md file
    6. Review the suggestions and implement them in your system prompt

Requirements:
    - LLM API key for analysis (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY, Alternative_api_key)
    - OPENAI_API_KEY for deduplication (uses OpenAI embeddings to detect similar skills)
"""

import os
from pathlib import Path
from datetime import datetime
from itertools import groupby
from typing import List, Dict, Any

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
from ace.prompts_v2_1 import PromptManager


def load_conversations(conversations_dir: Path) -> List[Dict[str, Any]]:
    """Load all .md and .toon conversation files from directory."""
    if not conversations_dir.exists():
        print(f"Directory not found: {conversations_dir}")
        return []

    conversations = []

    # Load markdown files
    for file_path in sorted(conversations_dir.glob("*.md")):
        try:
            content = file_path.read_text(encoding='utf-8')
            conversations.append({'filename': file_path.name, 'content': content})
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")

    # Load TOON files (fed directly to LLM as raw text)
    for file_path in sorted(conversations_dir.glob("*.toon")):
        try:
            content = file_path.read_text(encoding='utf-8')
            conversations.append({'filename': file_path.name, 'content': content})
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
            metadata={'response': conv['content']},
        )
        samples.append(sample)

    return samples


def main():
    # =========================================================================
    # USER CONFIGURATION - Update these values for your use case
    # =========================================================================
    CONVERSATIONS_DIR = Path("/path/to/your/conversations")  # Absolute path to .md files
    LLM_MODEL = "gpt-5-mini"                   # LLM model for analysis
    EPOCHS = 1                                 # Number of training epochs
    DEDUPLICATOR_SIMILARITY_THRESHOLD = 0.7    # Deduplication threshold (0.0-1.0)
    # =========================================================================

    SCRIPT_DIR = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_SKILLBOOK = SCRIPT_DIR / f'skillbook_{timestamp}.json'

    # Check for API keys
    if not os.getenv('OPENAI_API_KEY'):
        print("WARNING: OPENAI_API_KEY required for deduplication embeddings!")
        return

    # Load conversations
    conversations = load_conversations(CONVERSATIONS_DIR)
    if not conversations:
        print("\nTo use this example:")
        print(f"  1. Create directory: {CONVERSATIONS_DIR}/")
        print(f"  2. Add .md conversation files to that directory")
        return

    samples = create_samples(conversations)
    print(f"Created {len(samples)} samples")

    # Initialize ACE components
    skillbook = Skillbook()

    config = LiteLLMConfig(model=LLM_MODEL, max_tokens=8192, temperature=1)
    llm = LiteLLMClient(config=config)
    prompt_mgr = PromptManager()

    agent = ReplayAgent()
    reflector = Reflector(llm=llm, prompt_template=prompt_mgr.get_reflector_prompt())
    skill_manager = SkillManager(llm=llm, prompt_template=prompt_mgr.get_skill_manager_prompt())

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

    print(f"\nStarting analysis: {len(samples)} conversations, {EPOCHS} epoch(s), model={LLM_MODEL}")

    start_time = datetime.now()
    results = adapter.run(samples=samples, environment=SimpleEnvironment(), epochs=EPOCHS)
    duration = (datetime.now() - start_time).total_seconds()

    # Save and display results
    adapter.skillbook.save_to_file(str(OUTPUT_SKILLBOOK))

    skills = adapter.skillbook.skills()
    print(f"\nCompleted in {duration:.1f}s")
    print(f"Analyzed: {len(results)} conversations")
    print(f"Generated: {len(skills)} skills")
    print(f"Saved to: {OUTPUT_SKILLBOOK}")

    # Save skills grouped by section in markdown format
    OUTPUT_SKILLS_MD = SCRIPT_DIR / f'skills_{timestamp}.md'
    with open(OUTPUT_SKILLS_MD, 'w') as f:
        for section, section_skills in groupby(sorted(skills, key=lambda s: s.section), key=lambda s: s.section):
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
        for i, skill in enumerate(sorted(skills, key=lambda s: s.helpful, reverse=True)[:5], 1):
            print(f"  {i}. [{skill.section}] {skill.content[:80]}...")


if __name__ == '__main__':
    main()
