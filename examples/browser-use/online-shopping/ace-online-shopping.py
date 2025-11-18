#!/usr/bin/env python3
"""
ACE + Browser-Use Grocery Shopping Demo

Simple demo showing ACE learning to improve at grocery shopping automation.
Uses the new ACEAgent integration for clean, automatic learning.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from dotenv import load_dotenv
load_dotenv()

# Import ACE framework with new integration
from ace import ACEAgent
from ace.observability import configure_opik
from browser_use import ChatBrowserUse

# Shopping task definition
GROCERY_SHOPPING_TASK = """
### Migros Grocery Shopping - Essential 5 Items

**Objective:**
Shop for 5 essential items at Migros online store. Find the CHEAPEST available option for each item and add them to the basket.

**Essential Items:**
1. **1L full-fat milk**
2. **10 eggs (large)**
3. **1kg bananas**
4. **500g butter**
5. **1 loaf white bread (500g)**

**Instructions:**
- Visit https://www.migros.ch/en
- Search for each item and find the CHEAPEST option
- Add ONE of each to basket (don't checkout)
- Record item details: name, brand, price
- Provide final basket summary with total price

**Final Output Format:**
```
MIGROS BASKET:
- 1L milk: [brand] - CHF [price]
- 10 eggs: [brand] - CHF [price]
- 1kg bananas: [brand] - CHF [price]
- 500g butter: [brand] - CHF [price]
- White bread: [brand] - CHF [price]
TOTAL: CHF [total]
```

**Important:**
- Find CHEAPEST options only
- If exact match unavailable, choose closest alternative
- DO NOT complete purchase - basket only
"""


async def main():
    """Run grocery shopping with ACE learning."""

    # Configure observability
    try:
        configure_opik(project_name="ace-grocery-shopping")
        print("📊 Opik observability enabled")
    except:
        print("📊 Opik not available, continuing without observability")

    print("\n🛒 ACE + Browser-Use Grocery Shopping")
    print("🧠 Automated shopping with learning!")
    print("=" * 60)

    # Setup playbook persistence
    playbook_path = Path(__file__).parent / "ace_grocery_playbook.json"

    # Create ACE agent - handles everything automatically!
    agent = ACEAgent(
        llm=ChatBrowserUse(),                    # Browser automation LLM
        ace_model="claude-haiku-4-5-20251001",   # ACE learning LLM
        ace_max_tokens=4096,                     # Enough for shopping analysis
        playbook_path=str(playbook_path) if playbook_path.exists() else None,
        max_steps=30,                            # Browser automation steps
        calculate_cost=True                      # Track usage
    )

    # Show current knowledge
    if playbook_path.exists():
        print(f"📚 Loaded {len(agent.playbook.bullets())} learned strategies")
    else:
        print("🆕 Starting with empty playbook - learning from scratch")

    print(f"\n🎯 Task: Shop for 5 essential items at Migros")
    print("💡 ACE will learn from this shopping experience\n")

    try:
        # Run shopping with automatic ACE learning
        history = await agent.run(task=GROCERY_SHOPPING_TASK)

        print("\n" + "=" * 60)
        print("✅ SHOPPING COMPLETED")
        print("=" * 60)

        # Show results
        if hasattr(history, 'final_result'):
            print("📋 Shopping Results:")
            print(history.final_result())

        # Show performance metrics
        if hasattr(history, 'number_of_steps'):
            steps = history.number_of_steps()
            print(f"\n📊 Performance: {steps} steps taken")

    except Exception as e:
        print(f"\n❌ Shopping failed: {e}")
        print("🧠 ACE will still learn from this failure")

    # Save learned strategies
    agent.save_playbook(str(playbook_path))

    # Show what ACE learned
    strategies = agent.playbook.bullets()
    print(f"\n🎯 LEARNED STRATEGIES: {len(strategies)} total")
    print("-" * 60)

    if strategies:
        # Show recent strategies (last 3)
        recent_strategies = strategies[-3:] if len(strategies) > 3 else strategies

        for i, bullet in enumerate(recent_strategies, 1):
            helpful = bullet.helpful
            harmful = bullet.harmful
            effectiveness = "✅" if helpful > harmful else "⚠️" if helpful == harmful else "❌"
            print(f"{i}. {effectiveness} {bullet.content}")
            print(f"   (+{helpful}/-{harmful})")

        if len(strategies) > 3:
            print(f"   ... and {len(strategies) - 3} older strategies")

        print(f"\n💾 Strategies saved to: {playbook_path}")
        print("🔄 Next run will use these learned strategies automatically!")
    else:
        print("No new strategies learned (task may have failed)")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())