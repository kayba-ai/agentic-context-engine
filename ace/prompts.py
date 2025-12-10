"""
Prompt templates for ACE roles - fully customizable for your use case.

These default prompts are adapted from the ACE paper. You can customize them
to better suit your specific task by providing your own templates when
initializing the Agent, Reflector, and SkillManager.

Customization Example:
    >>> from ace import Agent
    >>> from ace.llm_providers import LiteLLMClient
    >>>
    >>> # Custom agent prompt for code tasks
    >>> code_agent_prompt = '''
    ... You are a senior developer. Use the skillbook to write clean code.
    ...
    ... Skillbook: {skillbook}
    ... Reflection: {reflection}
    ... Task: {question}
    ... Requirements: {context}
    ...
    ... Return JSON with:
    ... - reasoning: Your approach
    ... - skill_ids: Applied strategies
    ... - final_answer: The code solution
    ... '''
    >>>
    >>> client = LiteLLMClient(model="gpt-4")
    >>> agent = Agent(client, prompt_template=code_agent_prompt)

Prompt Variables:
    Agent:
        - {skillbook}: Current skillbook strategies
        - {reflection}: Recent reflection context
        - {question}: The question/task to solve
        - {context}: Additional requirements or context

    Reflector:
        - {question}: Original question
        - {reasoning}: Agent's reasoning
        - {prediction}: Agent's answer
        - {ground_truth}: Correct answer if available
        - {feedback}: Environment feedback
        - {skillbook_excerpt}: Relevant skillbook skills used

    SkillManager:
        - {progress}: Training progress summary
        - {stats}: Skillbook statistics
        - {reflection}: Latest reflection analysis
        - {skillbook}: Current full skillbook
        - {question_context}: Question and feedback context

Tips for Custom Prompts:
    1. Keep JSON output format consistent
    2. Be specific about your domain (math, code, writing, etc.)
    3. Add task-specific instructions and constraints
    4. Test with your actual use cases
    5. Iterate based on the quality of generated strategies
"""

# Default Agent prompt - produces answers using skillbook strategies
AGENT_PROMPT = """\
You are an expert assistant that must solve the task using the provided skillbook of strategies.
Apply relevant skills, avoid known mistakes, and show step-by-step reasoning.

Skillbook:
{skillbook}

Recent reflection:
{reflection}

Question:
{question}

Additional context:
{context}

Respond with a compact JSON object:
{{
  "reasoning": "<step-by-step chain of thought>",
  "skill_ids": ["<id1>", "<id2>", "..."],
  "final_answer": "<concise final answer>"
}}
"""


# Default Reflector prompt - analyzes what went right/wrong
REFLECTOR_PROMPT = """\
You are a senior reviewer diagnosing the agent's trajectory.
Use the skillbook, model reasoning, and feedback to identify mistakes and actionable insights.
Output must be a single valid JSON object. Do NOT include analysis text or explanations outside the JSON.
Begin the response with `{{` and end with `}}`.

Question:
{question}
Model reasoning:
{reasoning}
Model prediction: {prediction}
Ground truth (if available): {ground_truth}
Feedback: {feedback}
Skillbook excerpts consulted:
{skillbook_excerpt}

Return JSON:
{{
  "reasoning": "<analysis>",
  "error_identification": "<what went wrong>",
  "root_cause_analysis": "<why it happened>",
  "correct_approach": "<what should be done>",
  "key_insight": "<reusable takeaway>",
  "skill_tags": [
    {{"id": "<skill-id>", "tag": "helpful|harmful|neutral"}}
  ]
}}
"""


# Default SkillManager prompt - updates skillbook based on reflections
SKILL_MANAGER_PROMPT = """\
You are the skill manager of the ACE skillbook. Merge the latest reflection into structured updates.
Only add genuinely new material. Do not regenerate the entire skillbook.
Respond with a single valid JSON object only—no analysis or extra narration.

Training progress: {progress}
Skillbook stats: {stats}

Recent reflection:
{reflection}

Current skillbook:
{skillbook}

Question context:
{question_context}

Respond with JSON:
{{
  "reasoning": "<how you decided on the updates>",
  "operations": [
    {{
      "type": "ADD|UPDATE|TAG|REMOVE",
      "section": "<section name>",
      "content": "<skill text>",
      "skill_id": "<optional existing id>",
      "metadata": {{"helpful": 1, "harmful": 0}}
    }}
  ]
}}
If no updates are required, return an empty list for "operations".
"""

# Backward compatibility aliases
GENERATOR_PROMPT = AGENT_PROMPT
CURATOR_PROMPT = SKILL_MANAGER_PROMPT
