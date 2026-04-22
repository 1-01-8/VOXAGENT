"""Lightweight LLM call tracing helpers for local debugging."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from voice_optimized_rag.llm.base import LLMProvider, ToolCallingResponse


def _normalize_preview(text: str, max_chars: int = 120) -> str:
    normalized = " ".join(text.strip().split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3] + "..."


@dataclass(frozen=True)
class LLMTraceEvent:
    method: str
    provider: str
    model: str
    prompt_chars: int
    context_chars: int
    prompt_preview: str
    latency_ms: float
    response_chars: int = 0
    response_preview: str = ""
    tool_names: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    tool_choice: str = "auto"
    finish_reason: str = ""
    streamed_chunks: int = 0
    error: str = ""


class LLMTraceRecorder:
    """Collect recent LLM calls for a single local interaction trace."""

    def __init__(self, provider: str, model: str, max_events: int = 20) -> None:
        self._provider = provider
        self._model = model
        self._max_events = max_events
        self._events: list[LLMTraceEvent] = []

    def clear(self) -> None:
        self._events.clear()

    @property
    def events(self) -> list[LLMTraceEvent]:
        return list(self._events)

    def record(self, event: LLMTraceEvent) -> None:
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]

    def format_lines(self) -> list[str]:
        if not self._events:
            return ["llm_trace: no llm calls"]

        lines: list[str] = []
        for index, event in enumerate(self._events, 1):
            summary = (
                f"llm[{index}] method={event.method}, provider={event.provider}, model={event.model}, "
                f"prompt_chars={event.prompt_chars}, context_chars={event.context_chars}, "
                f"latency_ms={event.latency_ms:.1f}"
            )
            if event.tool_names:
                summary += f", tools={','.join(event.tool_names[:5])}"
            if event.tool_calls:
                summary += f", tool_calls={' | '.join(event.tool_calls[:3])}"
            if event.finish_reason:
                summary += f", finish_reason={event.finish_reason}"
            if event.streamed_chunks:
                summary += f", streamed_chunks={event.streamed_chunks}"
            if event.error:
                summary += f", error={event.error}"
            if event.response_preview:
                summary += f", response={event.response_preview}"
            summary += f", prompt={event.prompt_preview}"
            lines.append(summary)
        return lines

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model


class TraceableLLMProvider(LLMProvider):
    """Wrapper that records LLM calls without changing provider behavior."""

    def __init__(
        self,
        inner: LLMProvider,
        recorder: LLMTraceRecorder,
    ) -> None:
        self._inner = inner
        self._recorder = recorder

    @property
    def supports_function_calling(self) -> bool:
        return self._inner.supports_function_calling

    async def generate(self, prompt: str, context: str = "") -> str:
        start = time.perf_counter()
        try:
            response = await self._inner.generate(prompt, context=context)
        except Exception as error:
            self._record_base_event(
                method="generate",
                prompt=prompt,
                context=context,
                latency_ms=(time.perf_counter() - start) * 1000,
                error=f"{type(error).__name__}: {error}",
            )
            raise

        self._record_base_event(
            method="generate",
            prompt=prompt,
            context=context,
            latency_ms=(time.perf_counter() - start) * 1000,
            response=response,
        )
        return response

    async def stream(self, prompt: str, context: str = "") -> AsyncIterator[str]:
        start = time.perf_counter()
        collected: list[str] = []
        chunk_count = 0
        try:
            async for chunk in self._inner.stream(prompt, context=context):
                collected.append(chunk)
                chunk_count += 1
                yield chunk
        except Exception as error:
            self._record_base_event(
                method="stream",
                prompt=prompt,
                context=context,
                latency_ms=(time.perf_counter() - start) * 1000,
                response="".join(collected),
                streamed_chunks=chunk_count,
                error=f"{type(error).__name__}: {error}",
            )
            raise

        self._record_base_event(
            method="stream",
            prompt=prompt,
            context=context,
            latency_ms=(time.perf_counter() - start) * 1000,
            response="".join(collected),
            streamed_chunks=chunk_count,
        )

    async def complete_with_tools(
        self,
        prompt: str,
        tools: list[dict[str, Any]],
        context: str = "",
        tool_choice: str = "auto",
    ) -> ToolCallingResponse:
        start = time.perf_counter()
        tool_names = [tool.get("function", {}).get("name", "unknown") for tool in tools]
        try:
            response = await self._inner.complete_with_tools(
                prompt=prompt,
                tools=tools,
                context=context,
                tool_choice=tool_choice,
            )
        except Exception as error:
            self._recorder.record(
                LLMTraceEvent(
                    method="complete_with_tools",
                    provider=self._recorder.provider,
                    model=self._recorder.model,
                    prompt_chars=len(prompt),
                    context_chars=len(context),
                    prompt_preview=_normalize_preview(prompt),
                    latency_ms=(time.perf_counter() - start) * 1000,
                    tool_names=tool_names,
                    tool_choice=tool_choice,
                    error=f"{type(error).__name__}: {error}",
                )
            )
            raise

        call_summaries = [
            f"{call.name}({json.dumps(call.arguments, ensure_ascii=False)})"
            for call in response.tool_calls
        ]
        self._recorder.record(
            LLMTraceEvent(
                method="complete_with_tools",
                provider=self._recorder.provider,
                model=self._recorder.model,
                prompt_chars=len(prompt),
                context_chars=len(context),
                prompt_preview=_normalize_preview(prompt),
                latency_ms=(time.perf_counter() - start) * 1000,
                response_chars=len(response.content),
                response_preview=_normalize_preview(response.content),
                tool_names=tool_names,
                tool_calls=call_summaries,
                tool_choice=tool_choice,
                finish_reason=response.finish_reason,
            )
        )
        return response

    def _record_base_event(
        self,
        method: str,
        prompt: str,
        context: str,
        latency_ms: float,
        response: str = "",
        streamed_chunks: int = 0,
        error: str = "",
    ) -> None:
        self._recorder.record(
            LLMTraceEvent(
                method=method,
                provider=self._recorder.provider,
                model=self._recorder.model,
                prompt_chars=len(prompt),
                context_chars=len(context),
                prompt_preview=_normalize_preview(prompt),
                latency_ms=latency_ms,
                response_chars=len(response),
                response_preview=_normalize_preview(response),
                streamed_chunks=streamed_chunks,
                error=error,
            )
        )