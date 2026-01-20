#!/usr/bin/env python3
"""
Agentic System Prompting Example

Demonstrates using ACE's OfflineACE adapter to analyze past agent
conversations and generate system prompt improvements.

Usage:
    1. Export/convert your agent conversations to .md files
    2. Place them in a directory
    3. Update CONVERSATIONS_DIR path below
    4. Run: python agentic_system_prompting.py

Requirements:
    - ANTHROPIC_API_KEY/OPENAI_API_KEY/Alternative_api_key environment variable
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from ace import (
    Skillbook,
    Sample,
    OfflineACE,
    Reflector,
    SkillManager,
    ReplayAgent,
    SimpleEnvironment,
)
from ace.llm_providers.litellm_client import LiteLLMClient, LiteLLMConfig
from ace.prompts_v2_1 import PromptManager


def load_conversations(conversations_dir: Path) -> List[Dict[str, Any]]:
    """Load all .md conversation files from directory."""
    if not conversations_dir.exists():
        print(f"Directory not found: {conversations_dir}")
        return []

    conversations = []
    for file_path in sorted(conversations_dir.glob("*.md")):
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
    LLM_MODEL = "claude-sonnet-4-5-20250929"   # LLM model to use
    EPOCHS = 1                                 # Number of training epochs
    # =========================================================================

    SCRIPT_DIR = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_SKILLBOOK = SCRIPT_DIR / f'skillbook_{timestamp}.json'

    # Check for API key
    if not os.getenv('OPENAI_API_KEY') and not os.getenv('ANTHROPIC_API_KEY'):
        print("WARNING: No OPENAI_API_KEY or ANTHROPIC_API_KEY found!")
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

    config = LiteLLMConfig(model=LLM_MODEL, max_tokens=8192, temperature=0.1)
    llm = LiteLLMClient(config=config)
    prompt_mgr = PromptManager()

    agent = ReplayAgent()
    reflector = Reflector(llm=llm, prompt_template=prompt_mgr.get_reflector_prompt())
    skill_manager = SkillManager(llm=llm, prompt_template=prompt_mgr.get_skill_manager_prompt())

    adapter = OfflineACE(
        skillbook=skillbook,
        agent=agent,
        reflector=reflector,
        skill_manager=skill_manager,
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

    # Save simplified version with just skill content
    OUTPUT_SKILLS_ONLY = SCRIPT_DIR / f'skills_{timestamp}.md'
    with open(OUTPUT_SKILLS_ONLY, 'w') as f:
        for skill in sorted(skills, key=lambda s: s.section):
            f.write(f"- {skill.content}\n")
    print(f"Skills only: {OUTPUT_SKILLS_ONLY}")

    if skills:
        print("\nTop skills:")
        for i, skill in enumerate(sorted(skills, key=lambda s: s.helpful, reverse=True)[:5], 1):
            print(f"  {i}. [{skill.section}] {skill.content[:80]}...")


if __name__ == '__main__':
    main()
