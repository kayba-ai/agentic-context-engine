"""LLM client abstractions used by ACE components."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Deque, Dict, Optional, Union

if TYPE_CHECKING:
    import torch


@dataclass
class LLMResponse:
    """Container for LLM outputs."""

    text: str
    raw: Optional[Dict[str, Any]] = None


class LLMClient(ABC):
    """Abstract interface so ACE can plug into any chat/completions API."""

    def __init__(self, model: Optional[str] = None) -> None:
        self.model = model

    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Return the model text for a given prompt."""


class DummyLLMClient(LLMClient):
    """Deterministic LLM stub for testing and dry runs."""

    def __init__(self, responses: Optional[Deque[str]] = None) -> None:
        super().__init__(model="dummy")
        self._responses: Deque[str] = responses or deque()

    def queue(self, text: str) -> None:
        """Enqueue a response to be used on the next completion call."""
        self._responses.append(text)

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        if not self._responses:
            raise RuntimeError("DummyLLMClient ran out of queued responses.")
        return LLMResponse(text=self._responses.popleft())


class TransformersLLMClient(LLMClient):
    """LLM client powered by `transformers` pipelines for chat-style models."""

    def __init__(
        self,
        model_path: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
        device_map: Union[str, Dict[str, int]] = "auto",
        torch_dtype: Union[str, "torch.dtype"] = "auto",
        trust_remote_code: bool = True,
        system_prompt: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model=model_path)

        # Import transformers lazily to avoid mandatory dependency for all users.
        from transformers import AutoTokenizer, pipeline  # type: ignore[import-untyped]

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self._pipeline = pipeline(
            "text-generation",
            model=model_path,
            tokenizer=self._tokenizer,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self._system_prompt = system_prompt or (
            "You are a JSON-only assistant that MUST reply with a single valid JSON object without extra text.\n"
            "Reasoning: low\n"
            "Do not expose analysis or chain-of-thought. Respond using the final JSON only."
        )
        self._defaults: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0.0,
            "return_full_text": False,
        }
        if generation_kwargs:
            self._defaults.update(generation_kwargs)

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        call_kwargs = dict(self._defaults)
        kwargs = dict(kwargs)
        kwargs.pop("refinement_round", None)
        call_kwargs.update(kwargs)

        # Build chat-formatted messages to leverage harmony template.
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        outputs = self._pipeline(messages, **call_kwargs)
        text = self._postprocess_text(self._extract_text(outputs))
        return LLMResponse(text=text, raw={"outputs": outputs})

    def _extract_text(self, outputs: Any) -> str:
        """Normalize pipeline outputs into a single string response."""
        if not outputs:
            return ""
        candidate = outputs[0]

        # Newer transformers versions return {"generated_text": [{"role": ..., "content": ...}, ...]}
        if isinstance(candidate, dict) and "generated_text" in candidate:
            generated = candidate["generated_text"]
            if isinstance(generated, list):
                # Grab the assistant role content if present.
                for message in generated:
                    if isinstance(message, dict) and message.get("role") == "assistant":
                        content = message.get("content")
                        if isinstance(content, str):
                            return content.strip()
                # Fallback to last item's content/text.
                last = generated[-1]
                if isinstance(last, dict):
                    return str(last.get("content") or last.get("text") or "")
                return str(last)
            if isinstance(generated, dict):
                return str(generated.get("content") or generated.get("text") or "")
            return str(generated)

        # Older versions might return {"generated_text": "..."}
        if isinstance(candidate, dict) and isinstance(
            candidate.get("generated_text"), str
        ):
            return candidate["generated_text"].strip()

        # Ultimate fallback: string representation.
        return str(candidate).strip()

    def _postprocess_text(self, text: str) -> str:
        """Trim analyzer prefixes and isolate JSON payloads when present."""
        trimmed = text.strip()
        if not trimmed:
            return trimmed

        marker = "assistantfinal"
        if marker in trimmed:
            trimmed = trimmed.split(marker, 1)[1].strip()

        if trimmed.startswith(marker):
            trimmed = trimmed[len(marker) :].strip()

        # Attempt to extract the first JSON object substring.
        if trimmed and trimmed[0] != "{":
            start = trimmed.find("{")
            end = trimmed.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = trimmed[start : end + 1].strip()
                candidate_clean = candidate.replace("\r", " ").replace("\n", " ")
                try:
                    json.loads(candidate_clean)
                    return candidate_clean
                except json.JSONDecodeError:
                    pass

        return trimmed.replace("\r", " ").replace("\n", " ")
