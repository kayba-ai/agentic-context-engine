# Tagging Redesign — Problem Analysis & Options

## Problem Summary

The current tagging system (`helpful`/`harmful`/`neutral` counters on `Skill`) produces structurally meaningless data. Tags are 100% helpful, 0% harmful, 0% neutral — not because of LLM judgment, but because of how the system is wired.

## How Tagging Works Today

There are two paths that produce tags:

### Path 1: Reflector → TagStep (tags existing skills)

**Flow:**
1. `ReflectStep` extracts `skill_ids` from the agent trace
2. `Reflector.reflect()` calls `make_skillbook_excerpt(skillbook, skill_ids)` to build context about which skills the agent cited
3. Reflector prompt includes tagging criteria (helpful/harmful/neutral) and returns `skill_tags`
4. `TagStep` iterates `skill_tags` and calls `skillbook.tag_skill(id, tag)`

**Broken for TraceAnalyser:**
- Traces from external agents (e.g. browser-use, Claude Code) are wrapped as raw dicts like `{"answer": content}` with no `skill_ids`
- `ReflectStep` (line 57-64 in `steps/reflect.py`) passes an empty `AgentOutput(reasoning="", final_answer="", skill_ids=[])`
- `make_skillbook_excerpt()` returns `""` → Reflector sees "(No strategies cited)"
- Reflector returns empty `skill_tags` → TagStep iterates nothing

**Result:** TagStep is a complete no-op for TraceAnalyser workflows.

### Path 2: SkillManager ADD (metadata on new skills)

**Flow:**
1. SkillManager prompt includes ADD operation examples with hardcoded `"metadata": {"helpful": 1, "harmful": 0}`
2. Every new skill is born with `helpful=1, harmful=0, neutral=0`

**Root cause:** The prompt examples (line ~627 in `implementations/prompts.py`) hardcode these values. The SkillManager prompt never asks for TAG operations — it only creates ADD/UPDATE/REMOVE.

**Result:** Every skill starts 100% helpful by construction, and nothing ever increments harmful or neutral.

### Net Effect

| Counter | Value | Reason |
|---------|-------|--------|
| helpful | Always ≥ 1 | Hardcoded in ADD metadata |
| harmful | Always 0 | Never incremented |
| neutral | Always 0 | Never incremented |

Tags are structural artifacts, not evidence-based judgments.

## Key Files

| File | Role |
|------|------|
| `ace_next/steps/tag.py` | TagStep — applies skill_tags to skillbook |
| `ace_next/steps/reflect.py` | ReflectStep — extracts trace → Reflector args |
| `ace_next/implementations/reflector.py` | Reflector.reflect() — LLM call that produces skill_tags |
| `ace_next/implementations/helpers.py` | make_skillbook_excerpt() — builds context from skill_ids |
| `ace_next/implementations/prompts.py` | REFLECTOR_PROMPT (tagging criteria), SKILL_MANAGER_PROMPT (hardcoded metadata) |
| `ace_next/core/skillbook.py` | Skill class (helpful/harmful/neutral fields), tag_skill(), apply_metadata() |
| `ace_next/core/outputs.py` | SkillTag, ReflectorOutput.skill_tags |

## Options

### Option A: Drop Tagging (Recommended as immediate fix)

**What:** Set `metadata: {}` on ADD operations. Accept that TagStep is a no-op for TraceAnalyser.

**Changes:**
- Update SKILL_MANAGER_PROMPT ADD examples to use `"metadata": {}` instead of `"metadata": {"helpful": 1, "harmful": 0}`
- Optionally set Skill defaults to `helpful=0, harmful=0, neutral=0`

**Pros:** Honest, minimal effort, stops pretending tags mean something.
**Cons:** Tags exist in the schema but are always zero — dead code smell.

### Option B: Evidence Counting

**What:** Replace `helpful`/`harmful`/`neutral` with `evidence_count` — track how many traces contributed to or reinforced a skill.

**Changes:**
- Replace three tag fields on `Skill` with `evidence_count: int`
- Each ADD increments by 1, each UPDATE that reinforces also increments
- Remove TagStep and SkillTag entirely
- Update serialization/deserialization

**Pros:** Practically useful (skills backed by more traces are higher confidence). Simple model.
**Cons:** Schema change, loses the helpful/harmful distinction (though it was fake anyway).

### Option C: Trace-Quality-Aware Reflector

**What:** Redesign the Reflector prompt to judge trace quality from internal signals rather than correctness against ground truth.

**Signals the Reflector could use:**
- Tool call failures / retries in the trace
- Backtracking or contradictory reasoning
- Final outcome indicators (error messages, timeouts)
- Coherence of the execution path

**Changes:**
- New Reflector prompt that works without ground truth
- Trace parsing to extract quality signals
- Possibly new output schema

**Pros:** Actually meaningful quality assessment. Works without ground truth.
**Cons:** Significant redesign. Needs careful prompt engineering. May not generalize across trace formats.

### Option D: Deferred Tagging

**What:** Remove TagStep from the TraceAnalyser pipeline entirely. Only tag skills when the skillbook is deployed with a live agent that produces proper `skill_ids` citations.

**Changes:**
- TraceAnalyser pipeline skips TagStep
- Tags only happen in the standard ACE loop (Agent → Evaluate → Reflect → Tag → SkillManager)
- Clean separation: extraction phase (TraceAnalyser) vs. evaluation phase (live deployment)

**Pros:** Architecturally clean. Tags would be real evidence when they do happen.
**Cons:** Skills from TraceAnalyser never get tagged until deployed. Two-phase workflow.

## Recommended Approach

**Phase 1 (now):** Option A — stop the fake tags. Minimal change, honest data.

**Phase 2 (later):** Option D — design a proper live-evaluation loop where skills get tagged based on real agent citations and outcomes.

Options B and C are features that make more sense once there's a working pipeline producing real results to evaluate.
