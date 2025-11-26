"""LangChain integration for ACE using langchain-litellm."""

from typing import Optional, Dict, Any, AsyncIterator, Iterator
import logging

Router: Optional[type]

try:
    from langchain_litellm import ChatLiteLLM, ChatLiteLLMRouter
    from litellm import Router as LiteLLMRouter

    LANGCHAIN_AVAILABLE = True
    Router = LiteLLMRouter  # type: ignore[assignment]
except ImportError:
    LANGCHAIN_AVAILABLE = False
    ChatLiteLLM = None  # type: ignore
    ChatLiteLLMRouter = None  # type: ignore
    Router = None  # type: ignore[assignment]

from ..llm import LLMClient, LLMResponse

logger = logging.getLogger(__name__)


class LangChainLiteLLMClient(LLMClient):
    """
    LangChain wrapper for LiteLLM integration.

    This client provides integration with LangChain's ChatLiteLLM and ChatLiteLLMRouter,
    enabling advanced features like model routing, load balancing, and integration
    with the broader LangChain ecosystem.

    Args:
        model: The model name to use (e.g., "gpt-4", "claude-3-sonnet")
        router: Optional LiteLLM Router for load balancing across models
        temperature: Temperature for response generation (0.0 to 1.0)
        max_tokens: Maximum tokens in response
        **kwargs: Additional parameters passed to ChatLiteLLM/ChatLiteLLMRouter

    Example:
        Basic usage:
        ```python
        client = LangChainLiteLLMClient(model="gpt-4", temperature=0.0)
        response = client.complete("What is the capital of France?")
        ```

        With router for load balancing:
        ```python
        from litellm import Router

        model_list = [
            {"model_name": "gpt-4", "litellm_params": {"model": "gpt-4"}},
            {"model_name": "gpt-4", "litellm_params": {"model": "azure/gpt-4"}},
        ]
        router = Router(model_list=model_list)
        client = LangChainLiteLLMClient(model="gpt-4", router=router)
        ```
    """

    def __init__(
        self,
        model: str,
        router: Optional[Any] = None,  # Router from litellm
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. "
                "Install it with: pip install ace-framework[langchain] "
                "or pip install langchain-litellm"
            )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize the appropriate LangChain client
        if router:
            logger.info(
                f"Initializing LangChainLiteLLMClient with router for model: {model}"
            )
            self.llm = ChatLiteLLMRouter(
                router=router,
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            self.is_router = True
        else:
            logger.info(f"Initializing LangChainLiteLLMClient for model: {model}")
            self.llm = ChatLiteLLM(
                model=model, temperature=temperature, max_tokens=max_tokens, **kwargs
            )
            self.is_router = False

    def _filter_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out ACE-specific parameters that shouldn't go to LangChain."""
        ace_specific_params = {"refinement_round", "max_refinement_rounds"}
        return {k: v for k, v in kwargs.items() if k not in ace_specific_params}

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Complete a prompt using the LangChain LiteLLM client.

        Args:
            prompt: The prompt to complete
            **kwargs: Additional parameters for the completion

        Returns:
            LLMResponse with the completion text and metadata
        """
        filtered_kwargs = self._filter_kwargs(kwargs)

        try:
            response = self.llm.invoke(prompt, **filtered_kwargs)

            # Extract metadata from response
            metadata = {
                "model": response.response_metadata.get("model", self.model),
                "finish_reason": response.response_metadata.get("finish_reason"),
            }

            # Add usage information if available
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                metadata["usage"] = {
                    "prompt_tokens": response.usage_metadata.get("input_tokens"),
                    "completion_tokens": response.usage_metadata.get("output_tokens"),
                    "total_tokens": response.usage_metadata.get("total_tokens"),
                }

            # Add router information if using router
            if self.is_router:
                metadata["router"] = True
                metadata["model_used"] = response.response_metadata.get(
                    "model_name", self.model
                )

            return LLMResponse(text=response.content, raw=metadata)

        except Exception as e:
            logger.error(f"Error in LangChain completion: {e}")
            raise

    async def acomplete(self, prompt: str, **kwargs) -> LLMResponse:
        """
        Asynchronously complete a prompt using the LangChain LiteLLM client.

        Args:
            prompt: The prompt to complete
            **kwargs: Additional parameters for the completion

        Returns:
            LLMResponse with the completion text and metadata
        """
        filtered_kwargs = self._filter_kwargs(kwargs)

        try:
            response = await self.llm.ainvoke(prompt, **filtered_kwargs)

            # Extract metadata from response
            metadata = {
                "model": response.response_metadata.get("model", self.model),
                "finish_reason": response.response_metadata.get("finish_reason"),
            }

            # Add usage information if available
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                metadata["usage"] = {
                    "prompt_tokens": response.usage_metadata.get("input_tokens"),
                    "completion_tokens": response.usage_metadata.get("output_tokens"),
                    "total_tokens": response.usage_metadata.get("total_tokens"),
                }

            # Add router information if using router
            if self.is_router:
                metadata["router"] = True
                metadata["model_used"] = response.response_metadata.get(
                    "model_name", self.model
                )

            return LLMResponse(text=response.content, raw=metadata)

        except Exception as e:
            logger.error(f"Error in async LangChain completion: {e}")
            raise

    def complete_with_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """
        Complete a prompt with streaming using the LangChain LiteLLM client.

        Args:
            prompt: The prompt to complete
            **kwargs: Additional parameters for the completion

        Yields:
            Tokens as they are generated
        """
        filtered_kwargs = self._filter_kwargs(kwargs)

        try:
            for chunk in self.llm.stream(prompt, **filtered_kwargs):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Error in LangChain streaming: {e}")
            raise

    async def acomplete_with_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        """
        Asynchronously complete a prompt with streaming using the LangChain LiteLLM client.

        Args:
            prompt: The prompt to complete
            **kwargs: Additional parameters for the completion

        Yields:
            Tokens as they are generated
        """
        filtered_kwargs = self._filter_kwargs(kwargs)

        try:
            async for chunk in self.llm.astream(prompt, **filtered_kwargs):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Error in async LangChain streaming: {e}")
            raise
