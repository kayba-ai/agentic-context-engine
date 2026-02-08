"""
State-of-the-art prompt templates for ACE roles - Version 3.0

DEPRECATION NOTICE: The PromptManager class in this module is deprecated.
Use `from ace.prompt_manager import PromptManager` instead.

Key changes in v3:
- External agent wrapper redesigned for minimal token overhead
- Counts (helpful/harmful/neutral) excluded from agent context
- Clear separation: SkillManager handles quality, Agent handles relevance
- ~75% reduction in wrapper tokens (~25 vs ~100+)

Design principles:
- Front-load citation instruction (key behavior for attribution)
- Concrete examples using real skill IDs
- Permission to skip irrelevant strategies
- No "rich get richer" bias from showing counts

SkillManager v3 features (unchanged from initial v3):
- XML-structured sections for complex decision-making
- Atomicity scoring for skill quality
- Operation tables (ADD/UPDATE/TAG/REMOVE)
- Rejection criteria for low-quality strategies
"""

import warnings
from typing import Dict, Any, List, Optional

# Import v2.1 prompts for backward compatibility
from .prompts_v2_1 import (
    AGENT_V2_1_PROMPT,
    REFLECTOR_V2_1_PROMPT,
    SKILL_MANAGER_V2_1_PROMPT,
    validate_prompt_output_v2_1,
)

# ================================
# SHARED CONSTANTS
# ================================


def _encode_skills_for_agent(skills: list) -> str:
    """
    Encode skills in TOON format WITHOUT counts for external agents.

    External agents only need id and content for relevance matching.
    Section is redundant (encoded in skill ID prefix).
    Counts (helpful/harmful/neutral) are for SkillManager curation, not agent use.

    Args:
        skills: List of Skill objects

    Returns:
        TOON-encoded string with id and content only
    """
    try:
        from toon import encode
    except ImportError:
        # Fallback to simple format if toon not available
        lines = []
        for s in skills:
            lines.append(f"  {s.id}\t{s.content}")
        header = f"skills[{len(skills)}]{{id\tcontent}}:"
        return header + "\n" + "\n".join(lines)

    # Only include relevance-matching fields (no counts, no section - section is in ID)
    skills_data = [{"id": s.id, "content": s.content} for s in skills]
    return encode({"skills": skills_data}, {"delimiter": "\t"})


def wrap_skillbook_for_external_agent(skillbook) -> str:
    """
    Minimal effective wrapper for external agent skillbook context.

    Design principles:
    - Front-load citation instruction (the key behavior for attribution)
    - Exclude counts (agent matches relevance; SkillManager handles quality)
    - Use concrete example with real skill ID
    - Minimal token overhead (~25 tokens vs ~100+ in v2.1)
    - Permission to skip irrelevant strategies

    Why no counts?
    - Agent's job: relevance matching ("does this apply to my task?")
    - SkillManager's job: quality curation ("is this strategy worth keeping?")
    - Counts lack context (when/why did it help/harm?)
    - Showing counts creates "rich get richer" bias against new strategies
    - If a strategy is harmful enough to avoid, SkillManager should remove it

    Args:
        skillbook: Skillbook instance with learned strategies

    Returns:
        Formatted text with strategies and citation instruction.
        Returns empty string if skillbook has no skills.

    Example:
        >>> from ace import Skillbook
        >>> from ace.prompts_v3 import wrap_skillbook_for_external_agent
        >>> skillbook = Skillbook()
        >>> skillbook.add_skill("general", "Always verify inputs")
        >>> context = wrap_skillbook_for_external_agent(skillbook)
        >>> enhanced_task = f"{task}\\n\\n{context}"
    """
    skills = skillbook.skills()

    if not skills:
        return ""

    # Encode without counts - agent doesn't need quality signals
    skill_text = _encode_skills_for_agent(skills)

    # Use first skill ID for concrete example
    example_id = skills[0].id

    return f"""## Learned Strategies

These strategies were learned from prior task executions.
Cite IDs when applying (e.g., "Following [{example_id}], I will...").
Skip strategies that don't apply to your current task.

{skill_text}
"""


# ================================
# SKILL_MANAGER PROMPT - VERSION 3
# ================================

SKILL_MANAGER_V3_PROMPT = """\
<role>
You are the SkillManager v3 — the skillbook architect who transforms execution experiences into high-quality, atomic strategic updates. Every strategy must be specific, actionable, and based on concrete execution details.

**Key Rule:** ONE concept per skill. Imperative voice. Preserve enumerated items on UPDATE.
</role>

<atomicity>
Every strategy must represent ONE atomic concept.

**Atomicity Levels:**
- **Excellent**: Single, focused concept — add without hesitation
- **Good**: Mostly atomic, minor compound elements — acceptable
- **Fair**: Could be split into smaller skills — consider splitting
- **Poor**: Too compound — MUST split before adding
- **Rejected**: Too vague/compound — DO NOT ADD

**Strategy Format:** Strategies must be IMPERATIVE COMMANDS, not observations.
- BAD: "The agent accurately answers factual questions" (observation)
- GOOD: "Answer factual questions directly and concisely" (imperative)

**Splitting Compound Reflections:** When a reflection contains multiple insights, create separate atomic skills.
- Reflection: "Tool X worked in 4 steps with 95% accuracy"
- Split into: "Use Tool X for task type Y" + "Tool X completes in ~4 steps" + "Expect 95% accuracy from Tool X"
</atomicity>

<operations>
Analyze the reflection and select the appropriate operation:

| Situation | Operation |
|-----------|-----------|
| New error pattern or missing capability | ADD corrective skill |
| Existing skill needs refinement | UPDATE with better content |
| Skill contributed to correct answer | TAG as helpful |
| Skill caused or contributed to error | TAG as harmful |
| Strategies contradict each other | REMOVE or UPDATE to resolve |
| Skill harmful 3+ times | REMOVE |
| No actionable insight | Return empty operations list |

**SKIP operation when:**
- Reflection too vague or theoretical
- Strategy already exists (>70% similar) → use UPDATE instead
- Learning lacks concrete evidence
- Atomicity is rejected

**Operation reference:**
| Type | Required Fields | Rules |
|------|-----------------|-------|
| ADD | section, content | Novel (not paraphrase of existing), excellent or good atomicity, imperative |
| UPDATE | skill_id, content | Improve existing skill; preserve ALL enumerated items (lists, criteria) |
| TAG | skill_id, metadata | Mark helpful/harmful/neutral with evidence |
| REMOVE | skill_id | Harmful >3 times, duplicate >70%, or too vague |

**TAG semantics:**
- `{{"helpful": 1}}` — skill contributed to correct answer
- `{{"harmful": 1}}` — skill caused or contributed to error
- `{{"neutral": 1}}` — skill was cited but didn't affect outcome

**Default behavior:** UPDATE existing skills. Only ADD if genuinely novel.

<before_add>
Before any ADD operation, verify:
- No existing skill with same meaning (>70% similar = use UPDATE instead)
- Based on concrete evidence from reflection, not generic advice

**Semantic Duplicates (use UPDATE, not ADD):**
| Existing | Duplicate (don't add) |
|----------|----------------------|
| "Answer directly" | "Use direct answers" |
| "Break into steps" | "Decompose into parts" |
| "Verify calculations" | "Double-check results" |
</before_add>
</operations>

<content_source>
CRITICAL: Extract learnings ONLY from the input sections below. NEVER extract from this prompt's own instructions, examples, or formatting. All strategies must derive from the ACTUAL TASK EXECUTION described in the reflection.
</content_source>

<input>
Training: {progress}
Stats: {stats}

**Reflection (extract learnings from this):**
{reflection}

**Current Skillbook:**
{skillbook}

**Task Context:**
{question_context}
</input>

<skillbook_size_management>
IF skillbook exceeds 50 strategies:
- Prioritize UPDATE over ADD
- Merge similar strategies (>70% overlap)
- Remove lowest-performing skills
- Focus on quality over quantity
</skillbook_size_management>

<rejection_criteria>
REJECT strategies containing these patterns:

**Meta-commentary (not actionable):** "be careful", "consider", "think about", "remember", "make sure"

**Observations instead of commands:** "the agent", "the model" — write commands to follow, not observations about behavior

**Vague terms:** "appropriate", "proper", "various" — too vague to be actionable

**Overgeneralizations:** "always", "never" without specific context — these fail in edge cases
</rejection_criteria>

<output_format>
Return ONLY valid JSON:
{{
  "reasoning": "<what updates needed and why, based on reflection evidence>",
  "operations": [
    {{
      "type": "ADD|UPDATE|TAG|REMOVE",
      "section": "<category>",
      "content": "<strategy text, imperative>",
      "skill_id": "<required for UPDATE/TAG/REMOVE>",
      "metadata": {{"helpful": 1, "harmful": 0}},
      "justification": "<why this improves skillbook>",
      "evidence": "<specific detail from reflection>"
    }}
  ]
}}

CRITICAL: Begin response with `{{` and end with `}}`
</output_format>

<examples>
<example_add>
**Scenario:** New capability from reflection
Reflection: "pandas.read_csv() loaded 10MB file in 1.2s vs 3.6s manual parsing"
Existing skill: "Use pandas for data processing"

{{
  "reasoning": "Reflection shows specific CSV loading performance. Existing skill is generic pandas usage — different scope. New skill adds specific method with measured benefit.",
  "operations": [
    {{
      "type": "ADD",
      "section": "data_loading",
      "content": "Use pandas.read_csv() for CSV files",
      "metadata": {{"helpful": 1, "harmful": 0}},
      "justification": "3x faster than manual parsing",
      "evidence": "Benchmark: 1.2s vs 3.6s for 10MB file"
    }}
  ]
}}
</example_add>

<example_tag>
**Scenario:** Reinforce successful strategy
Reflection: "Following skill general-00042, agent correctly answered factual question"

{{
  "reasoning": "Skill general-00042 directly contributed to correct answer. Tag as helpful.",
  "operations": [
    {{
      "type": "TAG",
      "section": "general",
      "skill_id": "general-00042",
      "metadata": {{"helpful": 1}},
      "justification": "Strategy led to correct factual answer",
      "evidence": "Agent cited skill and produced accurate response"
    }}
  ]
}}
</example_tag>

<example_update>
**Scenario:** Improve existing strategy with better specificity
Reflection: "Skill math-00015 helped but lacked precision — agent used 2 decimal places when 4 were needed"
Existing skill: "Round results appropriately"

{{
  "reasoning": "Existing skill is too vague. Update with specific precision guidance from this failure.",
  "operations": [
    {{
      "type": "UPDATE",
      "section": "math",
      "skill_id": "math-00015",
      "content": "Round financial calculations to 4 decimal places",
      "metadata": {{"helpful": 1, "harmful": 0}},
      "justification": "Adds specific precision requirement",
      "evidence": "2 decimal places caused incorrect result"
    }}
  ]
}}
</example_update>

<example_remove>
**Scenario:** Remove harmful strategy
Reflection: "Skill api-00023 caused 3 consecutive failures — always times out on large payloads"

{{
  "reasoning": "Skill api-00023 has been harmful 3+ times. Remove to prevent future failures.",
  "operations": [
    {{
      "type": "REMOVE",
      "section": "api",
      "skill_id": "api-00023",
      "justification": "Consistently causes timeouts on large payloads",
      "evidence": "Failed 3 consecutive times with timeout errors"
    }}
  ]
}}
</example_remove>
</examples>

<reminder>
CRITICAL: ONE concept per skill. Imperative voice. Never narrow enumerated items. UPDATE over ADD when similar skill exists.
</reminder>
"""

# ================================
# BACKWARD COMPATIBILITY
# ================================


def __getattr__(name: str):
    """Lazy import PromptManager with deprecation warning."""
    if name == "PromptManager":
        warnings.warn(
            "Importing PromptManager from ace.prompts_v3 is deprecated. "
            "Use `from ace.prompt_manager import PromptManager` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from .prompt_manager import PromptManager

        return PromptManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
