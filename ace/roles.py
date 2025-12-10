"""Agent, Reflector, and SkillManager components."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from .updates import UpdateBatch
from .llm import LLMClient
from .skillbook import Skillbook
from .prompts_v2_1 import PromptManager

# Use PromptManager to get v2.1 prompts with {current_date} filled in
_prompt_manager = PromptManager(default_version="2.1")
AGENT_PROMPT = _prompt_manager.get_agent_prompt()
REFLECTOR_PROMPT = _prompt_manager.get_reflector_prompt()
SKILL_MANAGER_PROMPT = _prompt_manager.get_skill_manager_prompt()

if TYPE_CHECKING:
    from .deduplication import DeduplicationManager

logger = logging.getLogger(__name__)

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


def extract_cited_skill_ids(text: str) -> List[str]:
    """
    Extract skill IDs cited in text using [id-format] notation.

    Parses text to find all skill ID citations in format [section-00001].
    Used to track which strategies were applied by analyzing reasoning traces.

    Args:
        text: Text containing skill citations (reasoning, thoughts, etc.)

    Returns:
        List of unique skill IDs in order of first appearance.
        Empty list if no citations found.

    Example:
        >>> reasoning = "Following [general-00042], I verified the data. Using [geo-00003] for lookup."
        >>> extract_cited_skill_ids(reasoning)
        ['general-00042', 'geo-00003']

        >>> # Filter to specific text (exclude tool outputs)
        >>> clean_text = get_agent_thoughts_only(history)
        >>> cited_ids = extract_cited_skill_ids(clean_text)
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


class AgentOutput(BaseModel):
    """Output from the Agent role containing reasoning and answer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reasoning: str = Field(..., description="Step-by-step reasoning process")
    final_answer: str = Field(..., description="The final answer to the question")
    skill_ids: List[str] = Field(
        default_factory=list, description="IDs of strategies cited in reasoning"
    )
    raw: Dict[str, Any] = Field(
        default_factory=dict, description="Raw LLM response data"
    )


class Agent:
    """
    Produces answers using the current skillbook of strategies.

    The Agent is one of three core ACE roles. It takes a question and
    uses the accumulated strategies in the skillbook to produce reasoned answers.

    Args:
        llm: The LLM client to use for generation
        prompt_template: Custom prompt template (uses AGENT_PROMPT by default)
        max_retries: Maximum validation retries via Instructor (default: 3)

    Example:
        >>> from ace import Agent, LiteLLMClient, Skillbook
        >>> client = LiteLLMClient(model="gpt-3.5-turbo")
        >>> agent = Agent(client)
        >>> skillbook = Skillbook()
        >>>
        >>> output = agent.generate(
        ...     question="What is the capital of France?",
        ...     context="Answer concisely",
        ...     skillbook=skillbook
        ... )
        >>> print(output.final_answer)
        Paris

    Custom Prompt Example:
        >>> custom_prompt = '''
        ... Use this skillbook: {skillbook}
        ... Question: {question}
        ... Context: {context}
        ... Reflection: {reflection}
        ... Return JSON with: reasoning, skill_ids, final_answer
        ... '''
        >>> agent = Agent(client, prompt_template=custom_prompt)
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = AGENT_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        # Auto-wrap with Instructor if not already wrapped
        # Use duck typing to detect Instructor capability (supports mocking)
        if hasattr(llm, "complete_structured"):
            self.llm = llm
        else:
            from .llm_providers.instructor_client import wrap_with_instructor

            self.llm = wrap_with_instructor(llm, max_retries=max_retries)  # type: ignore[assignment]

        self.prompt_template = prompt_template
        self.max_retries = max_retries

    @maybe_track(
        name="agent_generate",
        tags=["ace-framework", "role", "agent"],
        project_name="ace-roles",
    )
    def generate(
        self,
        *,
        question: str,
        context: Optional[str],
        skillbook: Skillbook,
        reflection: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentOutput:
        return self._generate_impl(
            question=question,
            context=context,
            skillbook=skillbook,
            reflection=reflection,
            **kwargs,
        )

    def _generate_impl(
        self,
        *,
        question: str,
        context: Optional[str],
        skillbook: Skillbook,
        reflection: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentOutput:
        """
        Generate an answer using the skillbook strategies.

        Args:
            question: The question to answer
            context: Additional context or requirements
            skillbook: The current skillbook of strategies
            reflection: Optional reflection from previous attempts
            **kwargs: Additional arguments passed to the LLM

        Returns:
            AgentOutput with reasoning, final_answer, and skill_ids used
        """
        base_prompt = self.prompt_template.format(
            skillbook=skillbook.as_prompt() or "(empty skillbook)",
            reflection=_format_optional(reflection),
            question=question,
            context=_format_optional(context),
        )

        # Filter out non-LLM kwargs (like 'sample' used for ReplayAgent)
        llm_kwargs = {k: v for k, v in kwargs.items() if k != "sample"}

        # Use Instructor for automatic validation (always available - core dependency)
        output = self.llm.complete_structured(base_prompt, AgentOutput, **llm_kwargs)
        output.skill_ids = extract_cited_skill_ids(output.reasoning)
        return output


class ReplayAgent:
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
        >>> agent = ReplayAgent(responses)
        >>> output = agent.generate(
        ...     question="What is 2+2?",
        ...     context="",
        ...     skillbook=Skillbook()
        ... )
        >>> print(output.final_answer)
        4

        Sample-based mode (for list-based datasets):
        >>> # Sample with response in metadata
        >>> sample = {'question': '...', 'metadata': {'response': 'answer'}}
        >>> agent = ReplayAgent()  # No dict needed
        >>> output = agent.generate(
        ...     question=sample['question'],
        ...     context='',
        ...     skillbook=Skillbook(),
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
        name="replay_agent_generate",
        tags=["ace-framework", "role", "replay-agent"],
        project_name="ace-roles",
    )
    def generate(
        self,
        *,
        question: str,
        context: Optional[str],
        skillbook: Skillbook,
        reflection: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentOutput:
        """
        Return the pre-recorded response for the given question.

        Resolution priority:
        1. Check if 'sample' in kwargs and extract response from sample.metadata or sample dict
        2. Look up question in responses dict
        3. Use default_response as fallback

        Args:
            question: The question to answer
            context: Additional context (ignored in replay)
            skillbook: The current skillbook (ignored in replay)
            reflection: Optional reflection (ignored in replay)
            **kwargs: Additional arguments. Can include 'sample' for sample-based mode.

        Returns:
            AgentOutput with the replayed answer

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
                f"ReplayAgent could not find response for question: '{question[:100]}...'. "
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

        # Return AgentOutput matching the interface
        return AgentOutput(
            reasoning=reasoning,
            final_answer=final_answer,
            skill_ids=[],  # No skills used in replay
            raw={
                "reasoning": reasoning,
                "final_answer": final_answer,
                "skill_ids": [],
                "replay_metadata": {
                    "response_source": response_source,
                    "question_found_in_dict": question in self.responses,
                    "sample_provided": "sample" in kwargs,
                    "total_responses_in_mapping": len(self.responses),
                },
            },
        )


class ExtractedLearning(BaseModel):
    """A single learning extracted by the Reflector from task execution."""

    learning: str = Field(..., description="The extracted learning or insight")
    atomicity_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="How atomic/focused this learning is"
    )
    evidence: str = Field(
        default="", description="Evidence from execution supporting this learning"
    )


class SkillTag(BaseModel):
    """Classification tag for a skill strategy (helpful/harmful/neutral)."""

    id: str = Field(..., description="The skill ID being tagged")
    tag: str = Field(
        ..., description="Classification: 'helpful', 'harmful', or 'neutral'"
    )


class ReflectorOutput(BaseModel):
    """Output from the Reflector role containing analysis and skill classifications."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reasoning: str = Field(..., description="Overall reasoning about the outcome")
    error_identification: str = Field(
        default="", description="Description of what went wrong (if applicable)"
    )
    root_cause_analysis: str = Field(
        default="", description="Analysis of why errors occurred"
    )
    correct_approach: str = Field(
        ..., description="What the correct approach should be"
    )
    key_insight: str = Field(
        ..., description="The main lesson learned from this iteration"
    )
    extracted_learnings: List[ExtractedLearning] = Field(
        default_factory=list, description="Learnings extracted from task execution"
    )
    skill_tags: List[SkillTag] = Field(
        default_factory=list, description="Classifications of strategy effectiveness"
    )
    raw: Dict[str, Any] = Field(
        default_factory=dict, description="Raw LLM response data"
    )


class Reflector:
    """
    Analyzes agent outputs to extract lessons and improve strategies.

    The Reflector is the second ACE role. It analyzes the Agent's output
    and environment feedback to understand what went right or wrong, classifying
    which skillbook skills were helpful, harmful, or neutral.

    Args:
        llm: The LLM client to use for reflection
        prompt_template: Custom prompt template (uses REFLECTOR_PROMPT by default)
        max_retries: Maximum validation retries via Instructor (default: 3)

    Example:
        >>> from ace import Reflector, LiteLLMClient
        >>> client = LiteLLMClient(model="gpt-3.5-turbo")
        >>> reflector = Reflector(client)
        >>>
        >>> reflection = reflector.reflect(
        ...     question="What is 2+2?",
        ...     agent_output=agent_output,
        ...     skillbook=skillbook,
        ...     ground_truth="4",
        ...     feedback="Correct!"
        ... )
        >>> print(reflection.key_insight)
        Successfully solved the arithmetic problem
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = REFLECTOR_PROMPT,
        *,
        max_retries: int = 3,
    ) -> None:
        # Auto-wrap with Instructor if not already wrapped
        # Use duck typing to detect Instructor capability (supports mocking)
        if hasattr(llm, "complete_structured"):
            self.llm = llm
        else:
            from .llm_providers.instructor_client import wrap_with_instructor

            self.llm = wrap_with_instructor(llm, max_retries=max_retries)  # type: ignore[assignment]

        self.prompt_template = prompt_template
        self.max_retries = max_retries

    @maybe_track(
        name="reflector_reflect",
        tags=["ace-framework", "role", "reflector"],
        project_name="ace-roles",
    )
    def reflect(
        self,
        *,
        question: str,
        agent_output: AgentOutput,
        skillbook: Skillbook,
        ground_truth: Optional[str] = None,
        feedback: Optional[str] = None,
        **kwargs: Any,
    ) -> ReflectorOutput:
        return self._reflect_impl(
            question=question,
            agent_output=agent_output,
            skillbook=skillbook,
            ground_truth=ground_truth,
            feedback=feedback,
            **kwargs,
        )

    def _reflect_impl(
        self,
        *,
        question: str,
        agent_output: AgentOutput,
        skillbook: Skillbook,
        ground_truth: Optional[str],
        feedback: Optional[str],
        max_refinement_rounds: int = 1,
        **kwargs: Any,
    ) -> ReflectorOutput:
        skillbook_excerpt = _make_skillbook_excerpt(skillbook, agent_output.skill_ids)

        # Format skillbook section based on citation presence
        if skillbook_excerpt:
            skillbook_context = f"Strategies Applied:\n{skillbook_excerpt}"
        else:
            skillbook_context = "(No strategies cited - outcome-based learning)"

        base_prompt = self.prompt_template.format(
            question=question,
            reasoning=agent_output.reasoning,
            prediction=agent_output.final_answer,
            ground_truth=_format_optional(ground_truth),
            feedback=_format_optional(feedback),
            skillbook_excerpt=skillbook_context,
        )

        # Filter out non-LLM kwargs (like 'sample' used for ReplayAgent)
        llm_kwargs = {k: v for k, v in kwargs.items() if k != "sample"}

        # Use Instructor for automatic validation (always available - core dependency)
        return self.llm.complete_structured(base_prompt, ReflectorOutput, **llm_kwargs)


class SkillManagerOutput(BaseModel):
    """Output from the SkillManager role containing skillbook update operations."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    update: UpdateBatch = Field(
        ..., description="Batch of update operations to apply to skillbook"
    )
    consolidation_operations: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Operations to consolidate similar skills (MERGE, DELETE, KEEP, UPDATE)",
    )
    raw: Dict[str, Any] = Field(
        default_factory=dict, description="Raw LLM response data"
    )


class SkillManager:
    """
    Transforms reflections into actionable skillbook updates.

    The SkillManager is the third ACE role. It analyzes the Reflector's output
    and decides how to update the skillbook - adding new strategies, updating
    existing ones, or removing harmful patterns.

    Args:
        llm: The LLM client to use for skill management
        prompt_template: Custom prompt template (uses SKILL_MANAGER_PROMPT by default)
        max_retries: Maximum validation retries via Instructor (default: 3)
        dedup_manager: Optional DeduplicationManager for skill deduplication

    Example:
        >>> from ace import SkillManager, LiteLLMClient
        >>> client = LiteLLMClient(model="gpt-4")
        >>> skill_manager = SkillManager(client)
        >>>
        >>> # Process reflection to get update operations
        >>> output = skill_manager.update_skills(
        ...     reflection=reflection_output,
        ...     skillbook=skillbook,
        ...     question_context="Math problem solving",
        ...     progress="5/10 problems solved correctly"
        ... )
        >>> # Apply the update to skillbook
        >>> skillbook.apply_update(output.update)

    With Deduplication:
        >>> from ace.deduplication import DeduplicationManager, DeduplicationConfig
        >>> dedup_manager = DeduplicationManager(DeduplicationConfig())
        >>> skill_manager = SkillManager(client, dedup_manager=dedup_manager)
        >>> # SkillManager will now include similarity reports in prompts
        >>> # and handle MERGE/DELETE/KEEP/UPDATE consolidation operations

    Custom Prompt Example:
        >>> custom_prompt = '''
        ... Progress: {progress}
        ... Stats: {stats}
        ... Reflection: {reflection}
        ... Skillbook: {skillbook}
        ... Context: {question_context}
        ... Similarity Report: {similarity_report}
        ... Decide what changes to make. Return JSON with update operations.
        ... '''
        >>> skill_manager = SkillManager(client, prompt_template=custom_prompt)

    The SkillManager emits UpdateOperations:
        - ADD: Add new strategy skills
        - UPDATE: Modify existing skills
        - TAG: Update helpful/harmful counts
        - REMOVE: Delete unhelpful skills

    With deduplication enabled, also handles ConsolidationOperations:
        - MERGE: Combine similar skills
        - DELETE: Soft-delete redundant skills
        - KEEP: Mark similar skills as intentionally separate
        - UPDATE: Refine content to differentiate similar skills
    """

    def __init__(
        self,
        llm: LLMClient,
        prompt_template: str = SKILL_MANAGER_PROMPT,
        *,
        max_retries: int = 3,
        dedup_manager: Optional["DeduplicationManager"] = None,
    ) -> None:
        # Auto-wrap with Instructor if not already wrapped
        # Use duck typing to detect Instructor capability (supports mocking)
        if hasattr(llm, "complete_structured"):
            self.llm = llm
        else:
            from .llm_providers.instructor_client import wrap_with_instructor

            self.llm = wrap_with_instructor(llm, max_retries=max_retries)  # type: ignore[assignment]

        self.prompt_template = prompt_template
        self.max_retries = max_retries
        self.dedup_manager = dedup_manager

    @maybe_track(
        name="skill_manager_update_skills",
        tags=["ace-framework", "role", "skill-manager"],
        project_name="ace-roles",
    )
    def update_skills(
        self,
        *,
        reflection: ReflectorOutput,
        skillbook: Skillbook,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> SkillManagerOutput:
        return self._update_skills_impl(
            reflection=reflection,
            skillbook=skillbook,
            question_context=question_context,
            progress=progress,
            **kwargs,
        )

    def _update_skills_impl(
        self,
        *,
        reflection: ReflectorOutput,
        skillbook: Skillbook,
        question_context: str,
        progress: str,
        **kwargs: Any,
    ) -> SkillManagerOutput:
        """
        Generate update operations to modify the skillbook based on reflection.

        If a DeduplicationManager is configured, this method will:
        1. Generate a similarity report for similar skill pairs
        2. Include the report in the prompt for the SkillManager to handle
        3. Parse and apply consolidation operations from the response

        Args:
            reflection: The Reflector's analysis of what went right/wrong
            skillbook: Current skillbook to potentially update
            question_context: Description of the task domain or question type
            progress: Current progress summary (e.g., "5/10 correct")
            **kwargs: Additional arguments passed to the LLM

        Returns:
            SkillManagerOutput containing the update operations to apply

        Raises:
            RuntimeError: If unable to produce valid JSON after max_retries
        """
        # Get similarity report if deduplication is enabled
        similarity_report = None
        if self.dedup_manager is not None:
            similarity_report = self.dedup_manager.get_similarity_report(skillbook)
            if similarity_report:
                logger.info("Including similarity report in SkillManager prompt")

        # Serialize reflection with all meaningful fields (not just empty 'raw')
        reflection_data = {
            "reasoning": reflection.reasoning,
            "error_identification": reflection.error_identification,
            "root_cause_analysis": reflection.root_cause_analysis,
            "correct_approach": reflection.correct_approach,
            "key_insight": reflection.key_insight,
            "extracted_learnings": [
                l.model_dump() for l in reflection.extracted_learnings
            ],
        }

        base_prompt = self.prompt_template.format(
            progress=progress,
            stats=json.dumps(skillbook.stats()),
            reflection=json.dumps(reflection_data, ensure_ascii=False, indent=2),
            skillbook=skillbook.as_prompt() or "(empty skillbook)",
            question_context=question_context,
        )

        # Append similarity report if available
        if similarity_report:
            base_prompt = base_prompt + "\n\n" + similarity_report

        # Filter out non-LLM kwargs (like 'sample' used for ReplayAgent)
        llm_kwargs = {k: v for k, v in kwargs.items() if k != "sample"}

        # Use Instructor for automatic validation (always available - core dependency)
        output = self.llm.complete_structured(
            base_prompt, SkillManagerOutput, **llm_kwargs
        )

        # Apply consolidation operations if deduplication is enabled
        if self.dedup_manager is not None and output.consolidation_operations:
            response_data = {
                "consolidation_operations": output.consolidation_operations
            }
            applied_ops = self.dedup_manager.apply_operations_from_response(
                response_data, skillbook
            )
            if applied_ops:
                logger.info(f"Applied {len(applied_ops)} consolidation operations")

        return output


def _make_skillbook_excerpt(skillbook: Skillbook, skill_ids: Sequence[str]) -> str:
    lines: List[str] = []
    seen = set()
    for skill_id in skill_ids:
        if skill_id in seen:
            continue
        skill = skillbook.get_skill(skill_id)
        if skill:
            seen.add(skill_id)
            lines.append(f"[{skill.id}] {skill.content}")
    return "\n".join(lines)
