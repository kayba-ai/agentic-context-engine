# Data Flow Guide: How Information Moves Through ACE

**Version**: 0.5.1
**Last Updated**: January 2025
**Audience**: Developers, system architects
**Reading Time**: 40-50 minutes

## Table of Contents

1. [Introduction](#introduction)
2. [Core Data Flow Patterns](#core-data-flow-patterns)
3. [Offline Adaptation Flow](#offline-adaptation-flow)
4. [Online Adaptation Flow](#online-adaptation-flow)
5. [Integration Pattern Flows](#integration-pattern-flows)
6. [Token Flow and Optimization](#token-flow-and-optimization)
7. [Error Handling Flows](#error-handling-flows)
8. [Observability Data Flow](#observability-data-flow)
9. [Playbook Evolution Timeline](#playbook-evolution-timeline)
10. [Real-World Examples](#real-world-examples)

---

## Introduction

This guide illustrates how data flows through the Agentic Context Engineering framework. Understanding these flows helps you:

- **Debug issues**: Trace data through the system
- **Optimize performance**: Identify bottlenecks
- **Extend the framework**: Add custom components
- **Monitor behavior**: Understand what happens at each step

### Key Data Types

The framework processes several types of data:

1. **Task Data**: Questions, context, ground truth
2. **LLM Data**: Prompts, completions, tokens
3. **Strategy Data**: Bullets, delta operations, playbook state
4. **Evaluation Data**: Feedback, metrics, success/failure
5. **Observability Data**: Costs, token usage, performance metrics

---

## Core Data Flow Patterns

### Pattern 1: Single Sample Processing

The fundamental unit of work in ACE is processing one sample:

```
┌─────────────────────────────────────────────────────────────┐
│                    Single Sample Flow                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: Sample                                              │
│  ├─ question: "What is 2+2?"                                │
│  ├─ context: ""                                             │
│  └─ ground_truth: "4"                                       │
│                                                              │
│  ↓                                                           │
│  [Generator]                                                │
│  Input: question + context + playbook.as_prompt()           │
│  LLM Call: ~1500 tokens                                     │
│  Output: GeneratorOutput                                     │
│  ├─ reasoning: "Breaking down: 2+2..."                      │
│  ├─ final_answer: "4"                                       │
│  └─ bullet_ids: ["reasoning-00001"]                         │
│                                                              │
│  ↓                                                           │
│  [Environment]                                              │
│  Input: question + final_answer + ground_truth              │
│  Logic: Check if "4" in "4"                                 │
│  Output: EnvironmentResult                                   │
│  ├─ feedback: "Correct"                                     │
│  ├─ ground_truth: "4"                                       │
│  └─ metrics: {"accuracy": 1.0}                              │
│                                                              │
│  ↓                                                           │
│  [Reflector]                                                │
│  Input: question + generator_output + feedback + playbook   │
│  LLM Call: ~2000 tokens                                     │
│  Output: ReflectorOutput                                     │
│  ├─ reasoning: "The answer was correct..."                  │
│  ├─ error_identification: "No errors"                       │
│  ├─ root_cause_analysis: "Strategy worked well"             │
│  └─ bullet_tags: [{"bullet_id": "reasoning-00001",          │
│                     "tag": "helpful", ...}]                 │
│                                                              │
│  ↓                                                           │
│  [Curator]                                                  │
│  Input: reflection + playbook                               │
│  LLM Call: ~1800 tokens                                     │
│  Output: CuratorOutput                                       │
│  ├─ reasoning: "Tagging helpful strategy"                   │
│  └─ delta_batch:                                            │
│      └─ operations: [TAG reasoning-00001 as helpful]        │
│                                                              │
│  ↓                                                           │
│  [Playbook.apply_delta()]                                   │
│  Input: delta_batch                                         │
│  Process: For each operation:                               │
│    ├─ TAG → playbook.tag_bullet("reasoning-00001", "helpful")│
│    └─ Update bullet.helpful += 1                            │
│  Output: Updated playbook                                    │
│                                                              │
│  ↓                                                           │
│  Result: AdapterStepResult                                   │
│  ├─ All inputs and outputs captured                         │
│  ├─ Playbook snapshot saved                                 │
│  └─ Ready for analysis                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Token Breakdown** (typical):
- Generator: 1,500 tokens (800 input + 700 output)
- Reflector: 2,000 tokens (1,200 input + 800 output)
- Curator: 1,800 tokens (1,000 input + 800 output)
- **Total**: ~5,300 tokens per sample

---

### Pattern 2: Multi-Role Coordination

How the three roles coordinate:

```
                ┌──────────────┐
                │   Playbook   │ (Knowledge Base)
                │  50 bullets  │
                └──────┬───────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ↓              ↓              ↓
   [Generator]    [Reflector]    [Curator]
        │              │              │
        │              │              │
   Uses bullets   Analyzes       Updates
   as context     performance    playbook
        │              │              │
        ↓              ↓              ↓
   GeneratorOut   ReflectorOut   CuratorOut
   ├─ reasoning   ├─ analysis    ├─ reasoning
   ├─ answer      ├─ tags        └─ delta_batch
   └─ cited IDs   └─ root causes
        │              │              │
        └──────────────┴──────────────┘
                       │
                       ↓
              Playbook.apply_delta()
                       │
                       ↓
              ┌──────────────┐
              │   Playbook   │ (Updated)
              │  51 bullets  │
              └──────────────┘
```

**Key Points**:
1. **Generator reads** playbook, **doesn't modify**
2. **Reflector analyzes** but **doesn't modify** playbook
3. **Curator produces deltas**, **playbook applies** them
4. **Clean separation**: Each role has one responsibility

---

### Pattern 3: Delta Application

How delta operations modify the playbook:

```
Before Delta:
┌──────────────────────────────────────┐
│ Playbook (3 bullets)                 │
│ ├─ reasoning-00001 (✓3 ✗0 ~1)       │
│ ├─ edge_cases-00001 (✓1 ✗2 ~0)      │
│ └─ validation-00001 (✓2 ✗0 ~1)      │
└──────────────────────────────────────┘

DeltaBatch:
┌──────────────────────────────────────┐
│ reasoning: "Refining strategies"     │
│ operations:                          │
│   1. ADD                             │
│      section: "reasoning"            │
│      content: "Check input types"    │
│   2. TAG                             │
│      bullet_id: "reasoning-00001"    │
│      metadata: {helpful: 1}          │
│   3. REMOVE                          │
│      bullet_id: "edge_cases-00001"   │
└──────────────────────────────────────┘

Application Process:
┌──────────────────────────────────────┐
│ Operation 1 (ADD):                   │
│   ├─ Generate ID: "reasoning-00002"  │
│   ├─ Create Bullet object            │
│   └─ Add to playbook.bullets         │
│                                      │
│ Operation 2 (TAG):                   │
│   ├─ Find "reasoning-00001"          │
│   ├─ Increment helpful: 3 → 4        │
│   └─ Update timestamp                │
│                                      │
│ Operation 3 (REMOVE):                │
│   ├─ Find "edge_cases-00001"         │
│   └─ Delete from playbook.bullets    │
└──────────────────────────────────────┘

After Delta:
┌──────────────────────────────────────┐
│ Playbook (3 bullets)                 │
│ ├─ reasoning-00001 (✓4 ✗0 ~1)       │ ← Tagged
│ ├─ reasoning-00002 (✓0 ✗0 ~0)       │ ← Added
│ └─ validation-00001 (✓2 ✗0 ~1)      │
└──────────────────────────────────────┘
```

**Delta Operation Priority**:
1. **TAG** operations (update counters)
2. **UPDATE** operations (modify content)
3. **ADD** operations (create new bullets)
4. **REMOVE** operations (delete bullets)

---

## Offline Adaptation Flow

Complete data flow for batch training with multiple epochs.

### Full Flow Diagram

```
User Code:
┌────────────────────────────────────────────────────────────┐
│ samples = [Sample(...), Sample(...), ...]                  │
│ environment = MyEnvironment()                              │
│ adapter = OfflineAdapter(playbook, generator, ...)         │
│ results = adapter.run(samples, environment, epochs=3)      │
└────────────┬───────────────────────────────────────────────┘
             │
             ↓
OfflineAdapter.run():
┌────────────────────────────────────────────────────────────┐
│ for epoch in range(3):                                     │
│   print(f"Epoch {epoch+1}/3")                              │
│   for i, sample in enumerate(samples):                     │
│                                                            │
│     ┌──────────────────────────────────────────┐          │
│     │ result = _process_single_sample()        │          │
│     │   ├─ Generator.generate()                │          │
│     │   ├─ Environment.evaluate()              │          │
│     │   ├─ Reflector.reflect()                 │          │
│     │   ├─ Curator.curate()                    │          │
│     │   └─ Playbook.apply_delta()              │          │
│     └──────────────────────────────────────────┘          │
│                                                            │
│     all_results.append(result)                             │
│                                                            │
│     # Checkpoint?                                          │
│     if checkpoint_interval and (i+1) % checkpoint_interval == 0:│
│       playbook.save_to_file(f"checkpoint_{i+1}.json")      │
│       playbook.save_to_file("latest.json")                 │
│                                                            │
│ return all_results                                         │
└────────────────────────────────────────────────────────────┘
```

### Sample-by-Sample Evolution

```
Initial State (Epoch 1):
┌──────────────────────────────────────┐
│ Playbook: 0 bullets                  │
│ Accuracy: N/A                        │
└──────────────────────────────────────┘

Sample 1: "What is 2+2?"
├─ Generator: "4" (no strategies to use)
├─ Environment: "Correct"
├─ Reflector: "Answer was correct but no strategies used"
├─ Curator: ADD "Break down arithmetic"
└─ Playbook: 1 bullet

Sample 2: "What is 3+3?"
├─ Generator: "6" (uses bullet from Sample 1)
├─ Environment: "Correct"
├─ Reflector: "Strategy [reasoning-00001] was helpful"
├─ Curator: TAG reasoning-00001 as helpful
└─ Playbook: 1 bullet (✓1)

Sample 3: "What is 10*5?"
├─ Generator: "50" (uses arithmetic strategy)
├─ Environment: "Correct"
├─ Reflector: "Strategy worked well"
├─ Curator: TAG reasoning-00001 as helpful, ADD "Handle multiplication"
└─ Playbook: 2 bullets (reasoning ✓2, math ✓0)

... (more samples) ...

Epoch 1 Complete:
┌──────────────────────────────────────┐
│ Playbook: 15 bullets                 │
│ Accuracy: 85%                        │
│ Avg helpful tags: 3.2 per bullet     │
└──────────────────────────────────────┘

Epoch 2 (with learned strategies):
Sample 1 (revisited): "What is 2+2?"
├─ Generator: "4" (now uses 3 relevant strategies)
├─ Environment: "Correct"
├─ Reflector: "Strategies very helpful"
├─ Curator: TAG 3 strategies as helpful
└─ Playbook: 15 bullets (counters increased)

... (refinement continues) ...

Epoch 3 Complete:
┌──────────────────────────────────────┐
│ Playbook: 18 bullets                 │
│ Accuracy: 95%                        │
│ Avg helpful tags: 7.8 per bullet     │
│ Removed: 2 harmful bullets           │
└──────────────────────────────────────┘
```

### Checkpoint Data Flow

```
During Training:
┌────────────────────────────────────────┐
│ Sample 47 processed                    │
│   ↓                                    │
│ Check: 47 % 10 == 0? No                │
│   ↓                                    │
│ Sample 48 processed                    │
│   ↓                                    │
│ Check: 48 % 10 == 0? No                │
│   ↓                                    │
│ Sample 49 processed                    │
│   ↓                                    │
│ Check: 49 % 10 == 0? No                │
│   ↓                                    │
│ Sample 50 processed                    │
│   ↓                                    │
│ Check: 50 % 10 == 0? Yes!              │
│   ↓                                    │
│ Checkpoint Save:                       │
│   ├─ playbook.to_dict()                │
│   ├─ Save to checkpoint_50.json        │
│   └─ Save to latest.json               │
└────────────────────────────────────────┘

Checkpoint Files:
checkpoints/
├── checkpoint_10.json   (after sample 10)
├── checkpoint_20.json   (after sample 20)
├── checkpoint_30.json   (after sample 30)
├── checkpoint_40.json   (after sample 40)
├── checkpoint_50.json   (after sample 50)
└── latest.json          (always most recent = 50)
```

---

## Online Adaptation Flow

Real-time learning from streaming samples.

### Flow Diagram

```
Production System:
┌────────────────────────────────────────┐
│ adapter = OnlineAdapter(...)           │
│                                        │
│ while True:                            │
│   sample = get_next_user_query()       │
│   result = adapter.process(sample, env)│
│   send_response(result.answer)         │
└────────────────────────────────────────┘

OnlineAdapter.process():
┌────────────────────────────────────────┐
│ self.step_number += 1                  │
│                                        │
│ ┌──────────────────────────────┐      │
│ │ _process_single_sample()     │      │
│ │  ├─ Generator                │      │
│ │  ├─ Environment              │      │
│ │  ├─ Reflector                │      │
│ │  ├─ Curator                  │      │
│ │  └─ Playbook.apply_delta()   │      │
│ └──────────────────────────────┘      │
│                                        │
│ result.step_number = self.step_number  │
│ return result                          │
└────────────────────────────────────────┘

Timeline:
┌────────────────────────────────────────┐
│ t=0: User query arrives                │
│ t=1: Generator produces answer         │
│ t=2: Send response to user             │
│ t=2: Environment evaluates (background)│
│ t=3: Reflector analyzes (background)   │
│ t=4: Curator updates (background)      │
│ t=5: Playbook updated for next query   │
│                                        │
│ t=6: Next user query arrives           │
│      (uses updated playbook!)          │
└────────────────────────────────────────┘
```

### Continuous Learning Timeline

```
Step 1 (no playbook):
User: "What is 2+2?"
  ↓ Generator (no strategies)
Answer: "4"
  ↓ Learn
Playbook: 1 bullet added

Step 2 (with 1 bullet):
User: "What is 3+3?"
  ↓ Generator (uses 1 strategy)
Answer: "6"
  ↓ Learn
Playbook: 1 bullet tagged helpful

Step 3 (with refined strategies):
User: "What is 10/2?"
  ↓ Generator (uses helpful strategy)
Answer: "5"
  ↓ Learn
Playbook: 2 bullets (1 tagged, 1 added)

... continues indefinitely ...

Step 100:
Playbook: 25 bullets (refined over 100 interactions)
Accuracy: Continuously improving
```

---

## Integration Pattern Flows

### ACELiteLLM Flow (Full Pipeline)

```
User:
┌────────────────────────────────────────┐
│ agent = ACELiteLLM(model="gpt-4")      │
│ answer = agent.ask("Question?")        │
└────────────┬───────────────────────────┘
             │
             ↓
ACELiteLLM.ask():
┌────────────────────────────────────────┐
│ 1. GENERATE                            │
│    ├─ self.generator.generate()        │
│    └─ output = GeneratorOutput(...)    │
│                                        │
│ 2. EVALUATE (simple check)             │
│    ├─ Check if ground_truth in answer  │
│    └─ feedback = "Correct"/"Incorrect" │
│                                        │
│ 3. REFLECT                             │
│    ├─ self.reflector.reflect()         │
│    └─ reflection = ReflectorOutput(...) │
│                                        │
│ 4. CURATE                              │
│    ├─ self.curator.curate()            │
│    └─ curation = CuratorOutput(...)    │
│                                        │
│ 5. APPLY DELTA                         │
│    └─ self.playbook.apply_delta(...)   │
│                                        │
│ 6. SAVE (if playbook_path provided)    │
│    └─ self.playbook.save_to_file(...)  │
│                                        │
│ 7. RETURN                              │
│    └─ return output.final_answer       │
└────────────────────────────────────────┘
```

### ACEAgent Flow (Browser Integration)

```
User:
┌────────────────────────────────────────┐
│ ace_browser = ACEAgent(                │
│     agent=browser_agent,               │
│     llm=llm,                           │
│     playbook_path="./browser.json"     │
│ )                                      │
│ result = await ace_browser.run()       │
└────────────┬───────────────────────────┘
             │
             ↓
ACEAgent.run():
┌────────────────────────────────────────┐
│ 1. INJECT (if playbook not empty)      │
│    ├─ context = wrap_playbook_context()│
│    └─ self.agent.system_prompt += ctx  │
│                                        │
│ 2. EXECUTE (external agent)            │
│    ├─ result = await self.agent.run()  │
│    └─ (browser agent runs normally)    │
│                                        │
│ 3. LEARN (ACE analyzes)                │
│    ├─ task = self.agent.task           │
│    ├─ feedback = _extract_feedback()   │
│    ├─ reflection = reflector.reflect() │
│    ├─ curation = curator.curate()      │
│    └─ playbook.apply_delta()           │
│                                        │
│ 4. SAVE                                │
│    └─ playbook.save_to_file()          │
│                                        │
│ 5. RETURN                              │
│    └─ return result                    │
└────────────────────────────────────────┘

Data Flow Detail:
┌────────────────────────────────────────┐
│ Playbook (before):                     │
│   "When filling forms, check required  │
│    fields first"                       │
│                                        │
│ ↓ INJECT                               │
│                                        │
│ Browser Agent Prompt (enhanced):       │
│   "Fill out the form at example.com    │
│                                        │
│    # Learned Strategies:               │
│    - When filling forms, check required│
│      fields first"                     │
│                                        │
│ ↓ EXECUTE                              │
│                                        │
│ Browser Agent: (uses strategy)         │
│   1. Navigate to example.com           │
│   2. Check required fields (*)         │
│   3. Fill required fields first        │
│   4. Fill optional fields              │
│   5. Submit                            │
│                                        │
│ ↓ LEARN                                │
│                                        │
│ Reflector: "Strategy was helpful"      │
│ Curator: TAG strategy as helpful,      │
│          ADD "Validate inputs before   │
│               submit"                  │
│                                        │
│ Playbook (after):                      │
│   - "When filling forms..." (✓1)       │
│   - "Validate inputs before submit" (✓0)│
└────────────────────────────────────────┘
```

### ACELangChain Flow

```
User:
┌────────────────────────────────────────┐
│ ace_chain = ACELangChain(              │
│     chain=langchain_chain,             │
│     llm=llm,                           │
│     playbook_path="./chain.json"       │
│ )                                      │
│ result = ace_chain.run(input_data)     │
└────────────┬───────────────────────────┘
             │
             ↓
ACELangChain.run():
┌────────────────────────────────────────┐
│ 1. INJECT (if input is dict)           │
│    ├─ context = wrap_playbook_context()│
│    └─ input_data["playbook_context"]=ctx│
│                                        │
│ 2. EXECUTE (LangChain chain)           │
│    ├─ output = self.chain.invoke()     │
│    └─ (chain runs normally)            │
│                                        │
│ 3. LEARN                               │
│    ├─ Extract question/answer          │
│    ├─ reflection = reflector.reflect() │
│    ├─ curation = curator.curate()      │
│    └─ playbook.apply_delta()           │
│                                        │
│ 4. SAVE                                │
│    └─ playbook.save_to_file()          │
│                                        │
│ 5. RETURN                              │
│    └─ return output                    │
└────────────────────────────────────────┘
```

---

## Token Flow and Optimization

### Token Usage by Component

```
Per Sample Token Breakdown:
┌────────────────────────────────────────┐
│ GENERATOR:                             │
│   Input:                               │
│     ├─ Question: 50 tokens             │
│     ├─ Context: 100 tokens             │
│     ├─ Prompt template: 200 tokens     │
│     └─ Playbook (TOON): 450 tokens     │
│     ─────────────────────────────      │
│     Total input: 800 tokens            │
│   Output:                              │
│     ├─ Reasoning: 500 tokens           │
│     └─ Final answer: 200 tokens        │
│     ─────────────────────────────      │
│     Total output: 700 tokens           │
│   ═══════════════════════════════      │
│   Generator total: 1,500 tokens        │
│                                        │
│ REFLECTOR:                             │
│   Input:                               │
│     ├─ Question: 50 tokens             │
│     ├─ Generator output: 700 tokens    │
│     ├─ Feedback: 50 tokens             │
│     ├─ Prompt template: 250 tokens     │
│     └─ Playbook: 150 tokens            │
│     ─────────────────────────────      │
│     Total input: 1,200 tokens          │
│   Output:                              │
│     ├─ Analysis: 600 tokens            │
│     └─ Bullet tags: 200 tokens         │
│     ─────────────────────────────      │
│     Total output: 800 tokens           │
│   ═══════════════════════════════      │
│   Reflector total: 2,000 tokens        │
│                                        │
│ CURATOR:                               │
│   Input:                               │
│     ├─ Reflection: 800 tokens          │
│     ├─ Prompt template: 200 tokens     │
│     └─ Playbook: 450 tokens            │
│     ─────────────────────────────      │
│     Total input: 1,450 tokens          │
│   Output:                              │
│     ├─ Reasoning: 300 tokens           │
│     └─ Delta operations: 200 tokens    │
│     ─────────────────────────────      │
│     Total output: 500 tokens           │
│   ═══════════════════════════════      │
│   Curator total: 1,950 tokens          │
│                                        │
│ ═══════════════════════════════        │
│ TOTAL PER SAMPLE: ~5,450 tokens        │
│ ═══════════════════════════════        │
└────────────────────────────────────────┘

Cost Example (GPT-4):
├─ Input: ~3,500 tokens × $0.03/1K = $0.105
├─ Output: ~1,950 tokens × $0.06/1K = $0.117
└─ Total per sample: ~$0.22
```

### TOON Encoding Impact

```
JSON Format (before TOON):
┌────────────────────────────────────────┐
│ {                                      │
│   "bullets": {                         │
│     "reasoning-00001": {               │
│       "id": "reasoning-00001",         │
│       "section": "reasoning",          │
│       "content": "Break down complex", │
│       "helpful": 5,                    │
│       "harmful": 0,                    │
│       "neutral": 1,                    │
│       "created_at": "2024-01-15...",   │
│       "updated_at": "2024-01-15..."    │
│     },                                 │
│     ... (49 more bullets)              │
│   }                                    │
│ }                                      │
│                                        │
│ Tokens: ~4,200                         │
└────────────────────────────────────────┘

TOON Format (after):
┌────────────────────────────────────────┐
│ # Playbook                             │
│ ## reasoning                           │
│ - [reasoning-00001] Break down complex │
│   (✓5 ✗0 ~1)                           │
│ - [reasoning-00002] Check edge cases   │
│   (✓3 ✗0 ~0)                           │
│ ## edge_cases                          │
│ - [edge_cases-00001] Validate inputs   │
│   (✓2 ✗1 ~0)                           │
│ ... (47 more bullets)                  │
│                                        │
│ Tokens: ~1,600                         │
└────────────────────────────────────────┘

Savings: 4,200 - 1,600 = 2,600 tokens (62%)
```

### Token Flow Timeline

```
Training 100 Samples:

Without TOON:
├─ Per sample: 7,800 tokens (4,200 playbook + 3,600 other)
├─ 100 samples: 780,000 tokens
└─ Cost (GPT-4): ~$34

With TOON:
├─ Per sample: 5,200 tokens (1,600 playbook + 3,600 other)
├─ 100 samples: 520,000 tokens
└─ Cost (GPT-4): ~$23

Savings: $11 (32% reduction)
```

---

## Error Handling Flows

### JSON Parse Failure Recovery

```
Normal Flow:
┌────────────────────────────────────────┐
│ Generator LLM Call                     │
│   ↓                                    │
│ Response: '{"reasoning":"...","final...'│
│   ↓                                    │
│ json.loads() → Success                 │
│   ↓                                    │
│ Return GeneratorOutput                 │
└────────────────────────────────────────┘

Error Flow:
┌────────────────────────────────────────┐
│ Generator LLM Call                     │
│   ↓                                    │
│ Response: 'The answer is 4 because...' │
│   (not JSON!)                          │
│   ↓                                    │
│ json.loads() → JSONDecodeError         │
│   ↓                                    │
│ Extract JSON from markdown             │
│   (try to find ```json ... ```)        │
│   ↓                                    │
│ Still fails? → Retry with retry_prompt │
│   ↓                                    │
│ LLM Call 2:                            │
│   Prompt: original + retry_prompt      │
│   "Please return ONLY valid JSON"      │
│   ↓                                    │
│ Response: '{"reasoning":"...","final...'│
│   ↓                                    │
│ json.loads() → Success                 │
│   ↓                                    │
│ Return GeneratorOutput                 │
└────────────────────────────────────────┘

Retry Prompt Flow:
┌────────────────────────────────────────┐
│ Original Response (invalid):           │
│   "The answer is 4"                    │
│                                        │
│ Retry Prompt Construction:             │
│   original_prompt +                    │
│   "\n\nYour response: 'The answer...'"+│
│   "\n\nPlease return ONLY valid JSON:" │
│                                        │
│ LLM sees:                              │
│   [Full context + failed attempt +     │
│    instruction to fix]                 │
│                                        │
│ Success Rate: ~95% on retry            │
└────────────────────────────────────────┘
```

### Fallback Model Flow

```
Primary Model Failure:
┌────────────────────────────────────────┐
│ LiteLLMClient(                         │
│     model="gpt-4",                     │
│     fallback_models=["gpt-3.5-turbo",  │
│                     "claude-2"]        │
│ )                                      │
│                                        │
│ complete() called:                     │
│   ↓                                    │
│ Try gpt-4 → Timeout error              │
│   ↓                                    │
│ Try gpt-3.5-turbo → Success            │
│   ↓                                    │
│ Return response                        │
│   (with metadata about which model used)│
└────────────────────────────────────────┘
```

---

## Observability Data Flow

### Opik Integration Flow

```
LLM Call:
┌────────────────────────────────────────┐
│ LiteLLMClient.complete():              │
│   ├─ Call litellm.completion()         │
│   ├─ Get response with usage stats     │
│   │   ├─ prompt_tokens: 800            │
│   │   ├─ completion_tokens: 700        │
│   │   └─ cost: $0.105                  │
│   │                                    │
│   └─ if self.opik:                     │
│       self.opik.track_llm_call(        │
│           model="gpt-4",               │
│           prompt_tokens=800,           │
│           completion_tokens=700,       │
│           cost=0.105,                  │
│           role="generator"             │
│       )                                │
└────────────┬───────────────────────────┘
             │
             ↓
OpikIntegration.track_llm_call():
┌────────────────────────────────────────┐
│ Prepare trace data:                    │
│   {                                    │
│     "model": "gpt-4",                  │
│     "prompt_tokens": 800,              │
│     "completion_tokens": 700,          │
│     "total_tokens": 1500,              │
│     "cost": 0.105,                     │
│     "role": "generator",               │
│     "timestamp": "2024-01-15T10:30:00" │
│   }                                    │
│   ↓                                    │
│ self.client.log_trace(data)            │
│   ↓                                    │
│ Async upload to Opik                   │
└────────────────────────────────────────┘
             │
             ↓
Opik Dashboard:
┌────────────────────────────────────────┐
│ Real-time metrics:                     │
│   ├─ Total cost: $0.105                │
│   ├─ Tokens used: 1,500                │
│   ├─ Role: Generator                   │
│   ├─ Model: gpt-4                      │
│   └─ Timestamp: 10:30:00               │
│                                        │
│ Aggregated view:                       │
│   ├─ Generator: $15.50 (15K calls)     │
│   ├─ Reflector: $22.30 (15K calls)     │
│   ├─ Curator: $18.20 (15K calls)       │
│   └─ Total: $56.00                     │
└────────────────────────────────────────┘
```

---

## Playbook Evolution Timeline

### Timeline View

```
t=0 (Initial):
┌──────────────────────────────────────┐
│ Playbook: 0 bullets                  │
│ Performance: Baseline                │
└──────────────────────────────────────┘

t=10 samples:
┌──────────────────────────────────────┐
│ Playbook: 5 bullets                  │
│   ├─ reasoning-00001 (✓2 ✗0 ~1)     │
│   ├─ reasoning-00002 (✓1 ✗0 ~0)     │
│   ├─ edge_cases-00001 (✓0 ✗1 ~1)    │
│   ├─ validation-00001 (✓1 ✗0 ~0)    │
│   └─ math-00001 (✓1 ✗0 ~0)          │
│ Performance: +15%                    │
└──────────────────────────────────────┘

t=50 samples:
┌──────────────────────────────────────┐
│ Playbook: 15 bullets                 │
│   (10 new, 0 removed)                │
│ Top strategies:                      │
│   ├─ reasoning-00001 (✓12 ✗0 ~3)    │
│   ├─ math-00001 (✓8 ✗0 ~1)          │
│   └─ validation-00001 (✓7 ✗0 ~2)    │
│ Underperforming:                     │
│   └─ edge_cases-00001 (✓1 ✗3 ~2)    │
│ Performance: +35%                    │
└──────────────────────────────────────┘

t=100 samples:
┌──────────────────────────────────────┐
│ Playbook: 18 bullets                 │
│   (3 new, 2 removed)                 │
│ Removed:                             │
│   ├─ edge_cases-00001 (harmful)      │
│   └─ reasoning-00007 (duplicate)     │
│ Refined strategies (via UPDATE):     │
│   ├─ reasoning-00001 (updated 2×)    │
│   └─ math-00001 (updated 1×)         │
│ Performance: +48%                    │
└──────────────────────────────────────┘

t=500 samples (converged):
┌──────────────────────────────────────┐
│ Playbook: 25 bullets (stable)        │
│ Delta operations per sample: 0.8     │
│   (mostly TAG, few ADD/UPDATE)       │
│ Performance: +62% (plateaued)        │
└──────────────────────────────────────┘
```

---

## Real-World Examples

### Example 1: Math Q&A Training

```
Dataset: 100 arithmetic questions
Goal: Learn math-solving strategies

Sample Flow (Sample 42):
┌────────────────────────────────────────┐
│ Question: "What is 17 * 23?"          │
│                                        │
│ GENERATOR (uses playbook with 8 bullets):│
│   Reasoning:                           │
│     "Using [math-00001] break down:   │
│      17 * 23 = 17 * 20 + 17 * 3       │
│             = 340 + 51 = 391"         │
│   Answer: "391"                        │
│   Cited: ["math-00001", "reasoning-00002"]│
│                                        │
│ ENVIRONMENT:                           │
│   Check: "391" == "391" → Correct     │
│   Feedback: "Correct! Good work."     │
│                                        │
│ REFLECTOR:                             │
│   Analysis:                            │
│     "Strategy math-00001 was very     │
│      helpful for breaking down        │
│      multiplication. Strategy         │
│      reasoning-00002 helped structure │
│      the calculation."                │
│   Tags:                                │
│     - math-00001: helpful             │
│     - reasoning-00002: helpful        │
│                                        │
│ CURATOR:                               │
│   Decision:                            │
│     "Both strategies effective,       │
│      tag them as helpful. No new      │
│      insights to add."                │
│   Operations:                          │
│     - TAG math-00001 (helpful)        │
│     - TAG reasoning-00002 (helpful)   │
│                                        │
│ PLAYBOOK UPDATE:                       │
│   math-00001: ✓4 → ✓5                 │
│   reasoning-00002: ✓3 → ✓4            │
└────────────────────────────────────────┘

After 100 samples:
├─ Accuracy: 92%
├─ Playbook: 12 bullets
│   ├─ Math strategies: 5 bullets
│   ├─ Reasoning: 4 bullets
│   └─ Edge cases: 3 bullets
└─ Top strategy: math-00001 (✓42 uses)
```

### Example 2: Browser Form Filling

```
Task: Fill out registration forms
Integration: ACEAgent + browser-use

Run 1 (no playbook):
┌────────────────────────────────────────┐
│ INJECT: (no strategies to inject)     │
│                                        │
│ EXECUTE:                               │
│   Browser agent fills form:            │
│     - Enters email                     │
│     - Enters password                  │
│     - Clicks submit                    │
│   Result: Error - "Email invalid"      │
│                                        │
│ LEARN:                                 │
│   Reflector: "Failed due to invalid   │
│               email format"            │
│   Curator: ADD "Validate email format │
│                 before submit"         │
│                                        │
│ Playbook: 1 bullet added               │
└────────────────────────────────────────┘

Run 2 (with 1 strategy):
┌────────────────────────────────────────┐
│ INJECT:                                │
│   # Learned Strategies:                │
│   - Validate email format before submit│
│                                        │
│ EXECUTE:                               │
│   Browser agent (uses strategy):       │
│     - Enters email                     │
│     - Validates format ✓               │
│     - Enters password                  │
│     - Clicks submit                    │
│   Result: Error - "Password too short" │
│                                        │
│ LEARN:                                 │
│   Reflector: "Email validation worked,│
│               but password failed"     │
│   Curator: TAG email strategy helpful, │
│            ADD "Check password length" │
│                                        │
│ Playbook: 2 bullets (1 tagged, 1 added)│
└────────────────────────────────────────┘

Run 10 (mature playbook):
┌────────────────────────────────────────┐
│ Playbook: 8 strategies                 │
│   ├─ Email validation (✓7 ✗0)         │
│   ├─ Password length check (✓6 ✗0)    │
│   ├─ Check required fields (✓5 ✗0)    │
│   ├─ Fill required first (✓4 ✗0)      │
│   └─ ... (4 more)                      │
│                                        │
│ Success rate: 90%                      │
│   (was 10% without ACE)                │
└────────────────────────────────────────┘
```

---

## Summary

This guide covered all major data flows in the ACE framework:

✅ **Core Patterns**: Single sample processing, multi-role coordination, delta application
✅ **Adaptation**: Offline batch training, online streaming learning
✅ **Integration**: ACELiteLLM, ACEAgent, ACELangChain flows
✅ **Optimization**: Token flow, TOON encoding impact
✅ **Error Handling**: JSON parse recovery, fallback models
✅ **Observability**: Opik integration, cost tracking
✅ **Evolution**: Playbook growth and refinement over time
✅ **Real-World**: Math Q&A and browser automation examples

**Key Takeaways**:

1. **Data flows in cycles**: Sample → Generate → Evaluate → Reflect → Curate → Apply
2. **Roles are stateless**: Only playbook maintains state
3. **Deltas are incremental**: Small updates, not full rewrites
4. **Token efficiency matters**: TOON encoding saves 16-62%
5. **Observability is automatic**: Cost tracking with zero config
6. **Playbooks evolve**: From empty to refined over samples

---

**Related Documentation**:
- [COMPREHENSIVE_GUIDE.md](./COMPREHENSIVE_GUIDE.md) - High-level overview
- [ARCHITECTURE_DEEP_DIVE.md](./ARCHITECTURE_DEEP_DIVE.md) - Technical architecture
- [COMPONENT_REFERENCE.md](./COMPONENT_REFERENCE.md) - API reference
- [DEVELOPER_GUIDE.md](./DEVELOPER_GUIDE.md) - Implementation patterns

---

*Last updated: January 2025*
*Framework version: 0.5.1*
