"""
Prompt templates for ACE roles - fully customizable for your use case.

These default prompts are adapted from the ACE paper. You can customize them
to better suit your specific task by providing your own templates when
initializing the Generator, Reflector, and Curator.

Customization Example:
    >>> from ace import Generator
    >>> from ace.llm_providers import LiteLLMClient
    >>>
    >>> # Custom generator prompt for code tasks
    >>> code_generator_prompt = '''
    ... You are a senior developer. Use the playbook to write clean code.
    ...
    ... Playbook: {playbook}
    ... Reflection: {reflection}
    ... Task: {question}
    ... Requirements: {context}
    ...
    ... Return JSON with:
    ... - reasoning: Your approach
    ... - bullet_ids: Applied strategies
    ... - final_answer: The code solution
    ... '''
    >>>
    >>> client = LiteLLMClient(model="gpt-4")
    >>> generator = Generator(client, prompt_template=code_generator_prompt)

Prompt Variables:
    Generator:
        - {playbook}: Current playbook strategies
        - {reflection}: Recent reflection context
        - {question}: The question/task to solve
        - {context}: Additional requirements or context

    Reflector:
        - {question}: Original question
        - {reasoning}: Generator's reasoning
        - {prediction}: Generator's answer
        - {ground_truth}: Correct answer if available
        - {feedback}: Environment feedback
        - {playbook_excerpt}: Relevant playbook bullets used

    Curator:
        - {progress}: Training progress summary
        - {stats}: Playbook statistics
        - {reflection}: Latest reflection analysis
        - {playbook}: Current full playbook
        - {question_context}: Question and feedback context

Tips for Custom Prompts:
    1. Keep JSON output format consistent
    2. Be specific about your domain (math, code, writing, etc.)
    3. Add task-specific instructions and constraints
    4. Test with your actual use cases
    5. Iterate based on the quality of generated strategies
"""

# Default Generator prompt - produces answers using playbook strategies
GENERATOR_PROMPT = """\
You are an expert assistant that must solve the task using the provided playbook of strategies.
Apply relevant bullets, avoid known mistakes, and show step-by-step reasoning.

Playbook:
{playbook}

Recent reflection:
{reflection}

Question:
{question}

Additional context:
{context}

Respond with a compact JSON object:
{{
  "reasoning": "<step-by-step chain of thought>",
  "bullet_ids": ["<id1>", "<id2>", "..."],
  "final_answer": "<concise final answer>"
}}
"""


# Default Reflector prompt - analyzes what went right/wrong
REFLECTOR_PROMPT = """\
You are a senior reviewer diagnosing the generator's trajectory.
Use the playbook, model reasoning, and feedback to identify mistakes and actionable insights.
Output must be a single valid JSON object. Do NOT include analysis text or explanations outside the JSON.
Begin the response with `{{` and end with `}}`.

Question:
{question}
Model reasoning:
{reasoning}
Model prediction: {prediction}
Ground truth (if available): {ground_truth}
Feedback: {feedback}
Playbook excerpts consulted:
{playbook_excerpt}

Return JSON:
{{
  "reasoning": "<analysis>",
  "error_identification": "<what went wrong>",
  "root_cause_analysis": "<why it happened>",
  "correct_approach": "<what should be done>",
  "key_insight": "<reusable takeaway>",
  "bullet_tags": [
    {{"id": "<bullet-id>", "tag": "helpful|harmful|neutral"}}
  ]
}}
"""


# Default Curator prompt - updates playbook based on reflections
CURATOR_PROMPT = """\
You are the curator of the ACE playbook. Merge the latest reflection into structured updates.
Only add genuinely new material. Do not regenerate the entire playbook.
Respond with a single valid JSON object only—no analysis or extra narration.

Training progress: {progress}
Playbook stats: {stats}

Recent reflection:
{reflection}

Current playbook:
{playbook}

Question context:
{question_context}

Respond with JSON:
{{
  "reasoning": "<how you decided on the updates>",
  "operations": [
    {{
      "type": "ADD|UPDATE|TAG|REMOVE",
      "section": "<section name>",
      "content": "<bullet text>",
      "bullet_id": "<optional existing id>",
      "metadata": {{"helpful": 1, "harmful": 0}}
    }}
  ]
}}
If no updates are required, return an empty list for "operations".
"""
