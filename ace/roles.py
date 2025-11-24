"""Generator, Reflector, and Curator components."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .delta import DeltaBatch
from .llm import LLMClient
from .playbook import Playbook
from .prompts import CURATOR_PROMPT, GENERATOR_PROMPT, REFLECTOR_PROMPT

# Import Opik tracing with graceful degradation
try:
    from .observability.tracers import maybe_track
except ImportError:
    # Mock decorator if observability not available
    from typing import TypeVar, Callable

    F = TypeVar("F", bound=Callable[..., Any])

    def maybe_track(
        name: Optional[str] = None, tags: Optional[List[str]] = None, **kwargs: Any
    ) -> Callable[[F], F]:
        def decorator(func: F) -> F:
            return func

        return decorator


def _safe_json_loads(text: str) -> Dict[str, Any]:
    # Strip markdown code blocks if present
    text = text.strip()

    # Handle opening fence (with or without language identifier)
    if text.startswith("```json"):
        text = text[7:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()

    # Handle closing fence (if present)
    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # Check if this looks like incomplete JSON (truncated response)
        if "Unterminated string" in str(exc) or "Expecting" in str(exc):
            # Try to detect if this is a truncation issue
            if text.count("{") > text.count("}") or text.rstrip().endswith('"'):
                raise ValueError(
                    f"LLM response appears to be truncated JSON. This may indicate the response was cut off mid-generation. Original error: {exc}\nPartial text: {text[:200]}..."
                ) from exc

        debug_path = Path("logs/json_failures.log")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8") as fh:
            fh.write("----\n")
            fh.write(repr(text))
            fh.write("\n")
        raise ValueError(f"LLM response is not valid JSON: {exc}\n{text}") from exc
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from LLM.")
    return data


def _format_optional(value: Optional[str]) -> str:
    return value or "(none)"


def extract_cited_bullet_ids(text: str) -> List[str]:
    """
    Extract bullet IDs cited in text using [id-format] notation.

    Parses text to find all bullet ID citations in format [section-00001].
    Used to track which strategies were applied by analyzing reasoning traces.

    Args:
        text: Text containing bullet citations (reasoning, thoughts, etc.)

    Returns:
        List of unique bullet IDs in order of first appearance.
        Empty list if no citations found.

    Example:
        >>> reasoning = "Following [general-00042], I verified the data. Using [geo-00003] for lookup."
        >>> extract_cited_bullet_ids(reasoning)
        ['general-00042', 'geo-00003']

        >>> # Filter to specific text (exclude tool outputs)
        >>> clean_text = get_agent_thoughts_only(history)
        >>> cited_ids = extract_cited_bullet_ids(clean_text)
        ['strategy-001']

    Note:
        Pattern matches: [word_characters-digits]
        Deduplicates while preserving order of first occurrence.
    """
    import re

    # Match [section-digits] pattern
    matches = re.findall(r"\[([a-zA-Z_]+-\d+)\]", text)
    # Deduplicate while preserving order
    return list(dict.fromkeys(matches))


@dataclass
class GeneratorOutput:
    reasoning: str
    final_answer: str
    bullet_ids: List[str]
    raw: Dict[str, Any]


class Generator:
    """
    Produces answers using the current playbook of strategies.

    The Generator is one of three core ACE roles. It takes a question and
    uses the accumulated strategies in the playbook to produce reasoned answers.

    Args:
        llm: The LLM client to use for generation
        prompt_template: Custom prompt template (uses GENERATOR_PROMPT by default)
        max_retries: Maximum attempts if JSON parsing fails (default: 3)
        retry_prompt: Additional instruction appended on retry for JSON failures (default: English JSON reminder)

    Example:
        >>> from ace import Generator, LiteLLMClient, Playbook
        >>> client = LiteLLMClient(model="gpt-3.5-turbo")
        >>> generator = Generator(client)
        >>> playbook = Playbook()
        >>>
        >>> output = generator.generate(
        ...     question="What is the capital of France?",
        ...     context="Answer concisely",
        ...     playbook=playbook
        ... )
        >>> print(output.final_answer)
        Paris

    Custom Prompt Example:
        >>> custom_prompt = '''
        ... Use this playbook: {playbook}
        ... Question: {question}
        ... Context: {context}
        ... Reflection: {reflection}
        ... Return JSON with: reasoning, bullet_ids, final_answer
        ... '''
        >>> generator = Generator(client, prompt_template=custom_prompt)
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = GENERATOR_PROMPT,
        *,
        max_retries: int = 3,
        retry_prompt: str = "\n\nIMPORTANT: Return ONLY a single valid JSON object. Escape all quotes properly or use single quotes. Do not include any additional text outside the JSON.",
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries
        self.retry_prompt = retry_prompt

    @maybe_track(
        name="generator_generate",
        tags=["ace-framework", "role", "generator"],
        project_name="ace-roles",
    )
    def generate(
        self,
        *,
        question: str,
        context: Optional[str],
        playbook: Playbook,
        reflection: Optional[str] = None,
        **kwargs: Any,
    ) -> GeneratorOutput:
        return self._generate_impl(
            question=question,
            context=context,
            playbook=playbook,
            reflection=reflection,
            **kwargs,
        )

    def _generate_impl(
        self,
        *,
        question: str,
        context: Optional[str],
        playbook: Playbook,
        reflection: Optional[str] = None,
        **kwargs: Any,
    ) -> GeneratorOutput:
        """
        Generate an answer using the playbook strategies.

        Args:
            question: The question to answer
            context: Additional context or requirements
            playbook: The current playbook of strategies
            reflection: Optional reflection from previous attempts
            **kwargs: Additional arguments passed to the LLM

        Returns:
            GeneratorOutput with reasoning, final_answer, and bullet_ids used
        """
        base_prompt = self.prompt_template.format(
            playbook=playbook.as_prompt() or "(empty playbook)",
            reflection=_format_optional(reflection),
            question=question,
            context=_format_optional(context),
        )
        prompt = base_prompt
        last_error: Optional[Exception] = None

        # Filter out non-LLM kwargs (like 'sample' used for ReplayGenerator)
        llm_kwargs = {k: v for k, v in kwargs.items() if k != "sample"}

        for attempt in range(self.max_retries):
            response = self.llm.complete(prompt, **llm_kwargs)
            try:
                data = _safe_json_loads(response.text)
                reasoning = str(data.get("reasoning", ""))
                final_answer = str(data.get("final_answer", ""))

                # Extract bullet IDs from reasoning citations
                bullet_ids = extract_cited_bullet_ids(reasoning)

                return GeneratorOutput(
                    reasoning=reasoning,
                    final_answer=final_answer,
                    bullet_ids=bullet_ids,
                    raw=data,
                )
            except ValueError as err:
                last_error = err
                if attempt + 1 >= self.max_retries:
                    break
                # Append retry instruction to help LLM produce valid JSON
                # Configurable via retry_prompt parameter (supports different languages/models)
                prompt = base_prompt + self.retry_prompt
        raise RuntimeError("Generator failed to produce valid JSON.") from last_error


class ReplayGenerator:
    """
    Replays pre-recorded responses instead of calling an LLM.

    Useful for offline training from historical data (logs, traces, etc.)
    where you want ACE to learn from actual past interactions without
    generating new responses.

    Supports two modes:
    1. **Dict-based**: Lookup responses by question in a mapping (original mode)
    2. **Sample-based**: Read response directly from sample object/metadata (new mode)

    Args:
        responses: Dict mapping questions to their pre-recorded answers (optional)
        default_response: Response to return if question not found (default: "")

    Examples:
        Dict-based mode (original):
        >>> responses = {
        ...     "What is 2+2?": "4",
        ...     "What is the capital of France?": "Paris"
        ... }
        >>> generator = ReplayGenerator(responses)
        >>> output = generator.generate(
        ...     question="What is 2+2?",
        ...     context="",
        ...     playbook=Playbook()
        ... )
        >>> print(output.final_answer)
        4

        Sample-based mode (for list-based datasets):
        >>> # Sample with response in metadata
        >>> sample = {'question': '...', 'metadata': {'response': 'answer'}}
        >>> generator = ReplayGenerator()  # No dict needed
        >>> output = generator.generate(
        ...     question=sample['question'],
        ...     context='',
        ...     playbook=Playbook(),
        ...     sample=sample  # Pass sample in kwargs
        ... )
        >>> print(output.final_answer)
        answer
    """

    def __init__(
        self, responses: Optional[Dict[str, str]] = None, default_response: str = ""
    ) -> None:
        self.responses = responses if responses is not None else {}
        self.default_response = default_response

    def _extract_response_from_sample(
        self, sample: Any
    ) -> tuple[Optional[str], Optional[str]]:
        """
        Extract response from sample object using multiple fallback strategies.

        Args:
            sample: Sample object (can be dataclass, dict, or other)

        Returns:
            Tuple of (response_text, source_name) or (None, None) if not found
        """
        # Try sample.metadata['response'] (Sample dataclass)
        if hasattr(sample, "metadata") and isinstance(sample.metadata, dict):
            response = sample.metadata.get("response")
            if response:
                return response, "sample_metadata"

        # Try sample['metadata']['response'] (nested dict)
        if isinstance(sample, dict) and "metadata" in sample:
            if isinstance(sample["metadata"], dict):
                response = sample["metadata"].get("response")
                if response:
                    return response, "sample_dict_metadata"

        # Try sample['response'] (direct dict)
        if isinstance(sample, dict):
            response = sample.get("response")
            if response:
                return response, "sample_dict_direct"

        return None, None

    @maybe_track(
        name="replay_generator_generate",
        tags=["ace-framework", "role", "replay-generator"],
        project_name="ace-roles",
    )
    def generate(
        self,
        *,
        question: str,
        context: Optional[str],
        playbook: Playbook,
        reflection: Optional[str] = None,
        **kwargs: Any,
    ) -> GeneratorOutput:
        """
        Return the pre-recorded response for the given question.

        Resolution priority:
        1. Check if 'sample' in kwargs and extract response from sample.metadata or sample dict
        2. Look up question in responses dict
        3. Use default_response as fallback

        Args:
            question: The question to answer
            context: Additional context (ignored in replay)
            playbook: The current playbook (ignored in replay)
            reflection: Optional reflection (ignored in replay)
            **kwargs: Additional arguments. Can include 'sample' for sample-based mode.

        Returns:
            GeneratorOutput with the replayed answer

        Raises:
            ValueError: If no response can be found and no default is set
        """
        # Resolution priority:
        # 1. sample.metadata['response'] (preferred for Sample dataclass)
        # 2. sample['metadata']['response'] (dict with nested metadata)
        # 3. sample['response'] (dict with direct response)
        # 4. responses dict lookup by question
        # 5. default_response (fallback)

        final_answer = None
        response_source = None

        # Priority 1-3: Extract from sample if provided
        if "sample" in kwargs:
            sample = kwargs["sample"]
            final_answer, response_source = self._extract_response_from_sample(sample)

        # Priority 4: Look up in responses dict
        if not final_answer and question in self.responses:
            final_answer = self.responses[question]
            response_source = "responses_dict"

        # Priority 5: Use default response
        if not final_answer and self.default_response:
            final_answer = self.default_response
            response_source = "default_response"

        # Validation: Ensure we have a response
        if not final_answer:
            raise ValueError(
                f"ReplayGenerator could not find response for question: '{question[:100]}...'. "
                f"Checked: sample={('sample' in kwargs)}, "
                f"responses_dict={question in self.responses}, "
                f"default_response={bool(self.default_response)}. "
                "Ensure sample has 'response' field or provide default_response."
            )

        # Create metadata for observability
        reasoning_map: Dict[str, str] = {
            "sample_metadata": "[Replayed from sample.metadata]",
            "sample_dict_metadata": "[Replayed from sample dict metadata]",
            "sample_dict_direct": "[Replayed from sample dict]",
            "responses_dict": "[Replayed from responses dict]",
            "default_response": "[Replayed using default response]",
        }
        reasoning = reasoning_map.get(
            response_source if response_source else "", "[Replayed - source unknown]"
        )

        # Return GeneratorOutput matching the interface
        return GeneratorOutput(
            reasoning=reasoning,
            final_answer=final_answer,
            bullet_ids=[],  # No bullets used in replay
            raw={
                "reasoning": reasoning,
                "final_answer": final_answer,
                "bullet_ids": [],
                "replay_metadata": {
                    "response_source": response_source,
                    "question_found_in_dict": question in self.responses,
                    "sample_provided": "sample" in kwargs,
                    "total_responses_in_mapping": len(self.responses),
                },
            },
        )


@dataclass
class BulletTag:
    id: str
    tag: str


@dataclass
class ReflectorOutput:
    reasoning: str
    error_identification: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    bullet_tags: List[BulletTag]
    raw: Dict[str, Any]


class Reflector:
    """
    Analyzes generator outputs to extract lessons and improve strategies.

    The Reflector is the second ACE role. It analyzes the Generator's output
    and environment feedback to understand what went right or wrong, classifying
    which playbook bullets were helpful, harmful, or neutral.

    Args:
        llm: The LLM client to use for reflection
        prompt_template: Custom prompt template (uses REFLECTOR_PROMPT by default)
        max_retries: Maximum attempts if JSON parsing fails (default: 3)
        retry_prompt: Additional instruction appended on retry for JSON failures (default: English JSON reminder)

    Example:
        >>> from ace import Reflector, LiteLLMClient
        >>> client = LiteLLMClient(model="gpt-3.5-turbo")
        >>> reflector = Reflector(client)
        >>>
        >>> reflection = reflector.reflect(
        ...     question="What is 2+2?",
        ...     context="Show your work",
        ...     generator_trajectory="Reasoning: 2+2 = 4",
        ...     final_answer="4",
        ...     execution_feedback="Correct!",
        ...     playbook=playbook
        ... )
        >>> print(reflection.diagnosis)
        Successfully solved the arithmetic problem
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = REFLECTOR_PROMPT,
        *,
        max_retries: int = 3,
        retry_prompt: str = "\n\nIMPORTANT: Return ONLY a single valid JSON object. Escape all quotes properly or use single quotes. Do not include any additional text outside the JSON.",
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries
        self.retry_prompt = retry_prompt

    @maybe_track(
        name="reflector_reflect",
        tags=["ace-framework", "role", "reflector"],
        project_name="ace-roles",
    )
    def reflect(
        self,
        *,
        question: str,
        generator_output: GeneratorOutput,
        playbook: Playbook,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        return self._reflect_impl(
            question=question,
            generator_output=generator_output,
            playbook=playbook,
            ground_truth=ground_truth,
            feedback=feedback,
            **kwargs,
        )

    def _reflect_impl(
        self,
        *,
        question: str,
        generator_output: GeneratorOutput,
        playbook: Playbook,
        ground_truth: Optional[str],
        feedback: Optional[str],
        max_refinement_rounds: int = 1,
        **kwargs: Any,
    ) -> ReflectorOutput:
        playbook_excerpt = _make_playbook_excerpt(playbook, generator_output.bullet_ids)

        # Format playbook section based on citation presence
        if playbook_excerpt:
            playbook_context = f"Strategies Applied:\n{playbook_excerpt}"
        else:
            playbook_context = "(No strategies cited - outcome-based learning)"

        base_prompt = self.prompt_template.format(
            question=question,
            reasoning=generator_output.reasoning,
            prediction=generator_output.final_answer,
            ground_truth=_format_optional(ground_truth),
            feedback=_format_optional(feedback),
            playbook_excerpt=playbook_context,
        )
        result: Optional[ReflectorOutput] = None
        prompt = base_prompt
        last_error: Optional[Exception] = None

        # Filter out non-LLM kwargs (like 'sample' used for ReplayGenerator)
        llm_kwargs = {k: v for k, v in kwargs.items() if k != "sample"}

        for round_idx in range(max_refinement_rounds):
            prompt = base_prompt
            for attempt in range(self.max_retries):
                response = self.llm.complete(
                    prompt, refinement_round=round_idx, **llm_kwargs
                )
                try:
                    data = _safe_json_loads(response.text)
                    bullet_tags: List[BulletTag] = []
                    tags_payload = data.get("bullet_tags", [])
                    if isinstance(tags_payload, Sequence):
                        for item in tags_payload:
                            if (
                                isinstance(item, dict)
                                and "id" in item
                                and "tag" in item
                            ):
                                bullet_tags.append(
                                    BulletTag(
                                        id=str(item["id"]), tag=str(item["tag"]).lower()
                                    )
                                )
                    candidate = ReflectorOutput(
                        reasoning=str(data.get("reasoning", "")),
                        error_identification=str(data.get("error_identification", "")),
                        root_cause_analysis=str(data.get("root_cause_analysis", "")),
                        correct_approach=str(data.get("correct_approach", "")),
                        key_insight=str(data.get("key_insight", "")),
                        bullet_tags=bullet_tags,
                        raw=data,
                    )
                    result = candidate
                    # Early exit if we already have actionable output
                    if bullet_tags or candidate.key_insight:
                        return candidate
                    break
                except ValueError as err:
                    last_error = err
                    if attempt + 1 >= self.max_retries:
                        break
                    # Append retry instruction to help LLM produce valid JSON
                    # Configurable via retry_prompt parameter (supports different languages/models)
                    prompt = base_prompt + self.retry_prompt
        if result is None:
            raise RuntimeError("Reflector failed to produce a result.") from last_error
        return result


@dataclass
class CuratorOutput:
    delta: DeltaBatch
    raw: Dict[str, Any]


class Curator:
    """
    Transforms reflections into actionable playbook updates.

    The Curator is the third ACE role. It analyzes the Reflector's output
    and decides how to update the playbook - adding new strategies, updating
    existing ones, or removing harmful patterns.

    Args:
        llm: The LLM client to use for curation
        prompt_template: Custom prompt template (uses CURATOR_PROMPT by default)
        max_retries: Maximum attempts if JSON parsing fails (default: 3)
        retry_prompt: Additional instruction appended on retry for JSON failures (default: English JSON reminder)

    Example:
        >>> from ace import Curator, LiteLLMClient
        >>> client = LiteLLMClient(model="gpt-4")
        >>> curator = Curator(client)
        >>>
        >>> # Process reflection to get delta updates
        >>> output = curator.curate(
        ...     reflection=reflection_output,
        ...     playbook=playbook,
        ...     question_context="Math problem solving",
        ...     progress="5/10 problems solved correctly"
        ... )
        >>> # Apply the delta to update playbook
        >>> playbook.apply_delta(output.delta)

    Custom Prompt Example:
        >>> custom_prompt = '''
        ... Progress: {progress}
        ... Stats: {stats}
        ... Reflection: {reflection}
        ... Playbook: {playbook}
        ... Context: {question_context}
        ... Decide what changes to make. Return JSON with delta operations.
        ... '''
        >>> curator = Curator(client, prompt_template=custom_prompt)

    The Curator emits DeltaOperations:
        - ADD: Add new strategy bullets
        - UPDATE: Modify existing bullets
        - TAG: Update helpful/harmful counts
        - REMOVE: Delete unhelpful bullets
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = CURATOR_PROMPT,
        *,
        max_retries: int = 3,
        retry_prompt: str = "\n\nIMPORTANT: Return ONLY a single valid JSON object. The JSON must be complete with ALL required fields:\n- reasoning (string)\n- deduplication_check (object)\n- operations (array)\n- quality_metrics (object with avg_atomicity, operations_count, estimated_impact)\nEscape all quotes properly and ensure the JSON is complete and well-formed.",
    ) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_retries = max_retries
        self.retry_prompt = retry_prompt

    @maybe_track(
        name="curator_curate",
        tags=["ace-framework", "role", "curator"],
        project_name="ace-roles",
    )
    def curate(
        self,
        *,
        reflection: ReflectorOutput,
        playbook: Playbook,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> CuratorOutput:
        return self._curate_impl(
            reflection=reflection,
            playbook=playbook,
            question_context=question_context,
            progress=progress,
            **kwargs,
        )

    def _curate_impl(
        self,
        *,
        reflection: ReflectorOutput,
        playbook: Playbook,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> CuratorOutput:
        """
        Generate delta operations to update the playbook based on reflection.

        Args:
            reflection: The Reflector's analysis of what went right/wrong
            playbook: Current playbook to potentially update
            question_context: Description of the task domain or question type
            progress: Current progress summary (e.g., "5/10 correct")
            **kwargs: Additional arguments passed to the LLM

        Returns:
            CuratorOutput containing the delta operations to apply

        Raises:
            RuntimeError: If unable to produce valid JSON after max_retries
        """
        base_prompt = self.prompt_template.format(
            progress=progress,
            stats=json.dumps(playbook.stats()),
            reflection=json.dumps(reflection.raw, ensure_ascii=False, indent=2),
            playbook=playbook.as_prompt() or "(empty playbook)",
            question_context=question_context,
        )
        prompt = base_prompt
        last_error: Optional[Exception] = None

        # Filter out non-LLM kwargs (like 'sample' used for ReplayGenerator)
        llm_kwargs = {k: v for k, v in kwargs.items() if k != "sample"}

        for attempt in range(self.max_retries):
            response = self.llm.complete(prompt, **llm_kwargs)
            try:
                data = _safe_json_loads(response.text)
                delta = DeltaBatch.from_json(data)
                return CuratorOutput(delta=delta, raw=data)
            except ValueError as err:
                last_error = err
                if attempt + 1 >= self.max_retries:
                    break
                # Append retry instruction to help LLM produce valid JSON
                # Configurable via retry_prompt parameter (supports different languages/models)
                prompt = base_prompt + self.retry_prompt
        raise RuntimeError("Curator failed to produce valid JSON.") from last_error


def _make_playbook_excerpt(playbook: Playbook, bullet_ids: Sequence[str]) -> str:
    lines: List[str] = []
    seen = set()
    for bullet_id in bullet_ids:
        if bullet_id in seen:
            continue
        bullet = playbook.get_bullet(bullet_id)
        if bullet:
            seen.add(bullet_id)
            lines.append(f"[{bullet.id}] {bullet.content}")
    return "\n".join(lines)
