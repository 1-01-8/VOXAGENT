"""Ollama LLM provider implementation for local models."""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from voice_optimized_rag.llm.base import LLMProvider, ToolCall, ToolCallingResponse


class OllamaProvider(LLMProvider):
    """LLM provider using a local Ollama server."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._client = httpx.AsyncClient(timeout=60.0)

    @property
    def supports_function_calling(self) -> bool:
        return True

    async def generate(self, prompt: str, context: str = "") -> str:
        import time, logging
        _log = logging.getLogger("ollama")
        messages = self._build_messages(prompt, context)
        t0 = time.perf_counter()
        response = await self._client.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": self._temperature, "num_predict": 120},
            },
        )
        response.raise_for_status()
        data = response.json()
        ms = (time.perf_counter() - t0) * 1000
        eval_count = data.get("eval_count", 0)
        eval_dur_ms = data.get("eval_duration", 0) / 1e6
        prompt_eval_ms = data.get("prompt_eval_duration", 0) / 1e6
        _log.info(
            f"[LLM-TIMER] wall={ms:.0f}ms prompt_eval={prompt_eval_ms:.0f}ms "
            f"gen={eval_dur_ms:.0f}ms tokens={eval_count} "
            f"tok/s={eval_count/(eval_dur_ms/1000) if eval_dur_ms else 0:.1f}"
        )
        return data["message"]["content"]

    async def stream(self, prompt: str, context: str = "") -> AsyncIterator[str]:
        messages = self._build_messages(prompt, context)
        async with self._client.stream(
            "POST",
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": True,
                "options": {"temperature": self._temperature, "num_predict": 120},
            },
        ) as response:
            response.raise_for_status()
            import json
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    content = data.get("message", {}).get("content", "")
                    if content:
                        yield content

    async def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict],
        context: str = "",
        tool_choice: str = "auto",
    ) -> ToolCallingResponse:
        messages = self._build_messages(prompt, context)
        response = await self._client.post(
            f"{self._base_url}/api/chat",
            json={
                "model": self._model,
                "messages": messages,
                "stream": False,
                "tools": tools,
                "tool_choice": tool_choice,
                "options": {"temperature": self._temperature, "num_predict": 120},
            },
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        content = message.get("content", "") or ""
        tool_calls: list[ToolCall] = []

        for tool_call in message.get("tool_calls", []) or []:
            function = tool_call.get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            tool_calls.append(ToolCall(
                name=function.get("name", ""),
                arguments=arguments or {},
                call_id=tool_call.get("id", ""),
            ))

        return ToolCallingResponse(
            content=content,
            tool_calls=[call for call in tool_calls if call.name],
            finish_reason=data.get("done_reason", ""),
        )
