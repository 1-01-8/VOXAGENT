"""Abstract LLM provider interface and factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from voice_optimized_rag.config import VORConfig


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, context: str = "") -> str:
        """Generate a complete response.

        Args:
            prompt: The user prompt / instruction.
            context: Optional retrieved context to include.

        Returns:
            The full generated text.
        """

    @abstractmethod
    async def stream(self, prompt: str, context: str = "") -> AsyncIterator[str]:
        """Stream a response token by token.

        Args:
            prompt: The user prompt / instruction.
            context: Optional retrieved context to include.

        Yields:
            Text chunks as they are generated.
        """

    def _build_messages(self, prompt: str, context: str) -> list[dict[str, str]]:
        """Build a chat messages list from prompt and context."""
        messages: list[dict[str, str]] = []
        if context:
            messages.append({
                "role": "system",
                "content": (
                    "Use the following context to answer the user's question. "
                    "If the context is not relevant, answer from your own knowledge.\n\n"
                    f"Context:\n{context}"
                ),
            })
        messages.append({"role": "user", "content": prompt})
        return messages


def create_llm(config: VORConfig) -> LLMProvider:
    """Factory function to create an LLM provider from config."""
    provider = config.llm_provider

    if provider == "openai":
        from voice_optimized_rag.llm.openai_provider import OpenAIProvider
        return OpenAIProvider(
            api_key=config.llm_api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
            base_url=config.llm_base_url,
        )
    elif provider == "anthropic":
        from voice_optimized_rag.llm.anthropic_provider import AnthropicProvider
        return AnthropicProvider(
            api_key=config.llm_api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
        )
    elif provider == "ollama":
        from voice_optimized_rag.llm.ollama_provider import OllamaProvider
        return OllamaProvider(
            model=config.llm_model,
            base_url=config.llm_base_url or "http://localhost:11434",
            temperature=config.llm_temperature,
        )
    elif provider == "gemini":
        from voice_optimized_rag.llm.gemini_provider import GeminiProvider
        # Use gemini_api_key if set, otherwise fall back to llm_api_key
        api_key = config.gemini_api_key or config.llm_api_key
        return GeminiProvider(
            api_key=api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
            vertex_project=config.vertex_project,
            vertex_location=config.vertex_location,
        )
    elif provider == "siliconflow":
        from voice_optimized_rag.llm.siliconflow_provider import SiliconFlowProvider
        return SiliconFlowProvider(
            api_key=config.llm_api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
            base_url=config.llm_base_url or "https://api.siliconflow.cn/v1",
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
