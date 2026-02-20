"""Agent role — produces answers using the current skillbook of strategies."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from ..llm import LLMClient
from ..skillbook import Skillbook
from ..prompt_manager import PromptManager
from ._helpers import (
    _format_optional,
    extract_cited_skill_ids,
    maybe_track,
)

# Default prompt (v2.1 with {current_date} filled in)
_prompt_manager = PromptManager(default_version="2.1")
AGENT_PROMPT = _prompt_manager.get_agent_prompt()


# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

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
        self.llm = llm
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


# ---------------------------------------------------------------------------
# ReplayAgent
# ---------------------------------------------------------------------------

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
        context: Optional[str] = None,
        skillbook: Optional[Skillbook] = None,
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
