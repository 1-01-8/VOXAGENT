"""OpenAI LLM provider implementation.

Supports both standard OpenAI API and Salesforce Research Gateway.
When the base_url contains 'gateway.salesforceresearch.ai', the API key is
passed via X-Api-Key header with a dummy api_key (matching the gateway auth pattern).
"""

from __future__ import annotations

import os
from typing import AsyncIterator

from voice_optimized_rag.llm.base import LLMProvider


def _is_salesforce_gateway(url: str | None) -> bool:
    """Check if a URL points to Salesforce Research Gateway."""
    return bool(url and "gateway.salesforceresearch.ai" in url)


class OpenAIProvider(LLMProvider):
    """LLM provider using OpenAI's API.

    Authentication (in order of priority):
      1. Standard: api_key used directly (works with api.openai.com)
      2. Gateway:  api_key + base_url containing 'gateway.salesforceresearch.ai'
                   → key is passed via X-Api-Key header with dummy api_key
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("Install openai: pip install voice-optimized-rag[openai]")

        # Resolve from env if not provided
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        url = base_url or os.environ.get("OPENAI_BASE_URL")

        if _is_salesforce_gateway(url):
            # Salesforce gateway: pass key via X-Api-Key header
            self._client = AsyncOpenAI(
                api_key="dummy",
                base_url=url,
                default_headers={"X-Api-Key": key},
            )
        else:
            # Standard OpenAI API
            kwargs: dict = {"api_key": key}
            if url:
                kwargs["base_url"] = url
            self._client = AsyncOpenAI(**kwargs)

        self._model = model
        self._temperature = temperature

    async def generate(self, prompt: str, context: str = "") -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(prompt, context),
            temperature=self._temperature,
        )
        return response.choices[0].message.content or ""

    async def stream(self, prompt: str, context: str = "") -> AsyncIterator[str]:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(prompt, context),
            temperature=self._temperature,
            stream=True,
        )
        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
