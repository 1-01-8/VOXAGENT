"""Embedding provider abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from voice_optimized_rag.config import VORConfig


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: List of strings to embed.

        Returns:
            numpy array of shape (len(texts), dimension).
        """

    async def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text. Returns shape (dimension,)."""
        result = await self.embed([text])
        return result[0]


class OpenAIEmbedding(EmbeddingProvider):
    """Embedding provider using OpenAI's API.

    Supports both standard OpenAI API and Salesforce Research Gateway.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        base_url: str | None = None,
    ) -> None:
        import os
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Install openai: pip install voice-optimized-rag[openai]")

        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        url = base_url or os.environ.get("OPENAI_BASE_URL")

        if url and "gateway.salesforceresearch.ai" in url:
            self._client = AsyncOpenAI(
                api_key="dummy",
                base_url=url,
                default_headers={"X-Api-Key": key},
            )
        else:
            kwargs: dict = {"api_key": key}
            if url:
                kwargs["base_url"] = url
            self._client = AsyncOpenAI(**kwargs)

        self._model = model
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> np.ndarray:
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return np.array([item.embedding for item in response.data], dtype=np.float32)


class OllamaEmbedding(EmbeddingProvider):
    """Embedding provider using Ollama's local API."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        dim: int = 768,
    ) -> None:
        import httpx
        self._client = httpx.AsyncClient(timeout=60.0)
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> np.ndarray:
        response = await self._client.post(
            f"{self._base_url}/api/embed",
            json={"model": self._model, "input": texts},
        )
        response.raise_for_status()
        data = response.json()
        return np.array(data["embeddings"], dtype=np.float32)


def create_embedding_provider(config: VORConfig) -> EmbeddingProvider:
    """Factory function to create an embedding provider from config."""
    provider = config.embedding_provider

    if provider == "openai":
        return OpenAIEmbedding(
            api_key=config.llm_api_key,
            model=config.embedding_model,
            dim=config.embedding_dimension,
            base_url=config.llm_base_url,
        )
    elif provider == "ollama":
        return OllamaEmbedding(
            model=config.embedding_model,
            base_url=config.llm_base_url or "http://localhost:11434",
            dim=config.embedding_dimension,
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
