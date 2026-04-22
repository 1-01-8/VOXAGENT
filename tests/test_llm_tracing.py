"""Tests for llm/tracing.py — LLM 调用轨迹记录。"""

from __future__ import annotations

import pytest

from voice_optimized_rag.llm.base import ToolCall, ToolCallingResponse
from voice_optimized_rag.llm.tracing import LLMTraceRecorder, TraceableLLMProvider


@pytest.mark.asyncio
async def test_generate_trace_records_prompt_and_response(mock_llm):
    recorder = LLMTraceRecorder(provider="mock", model="test-model")
    llm = TraceableLLMProvider(mock_llm, recorder)

    response = await llm.generate("介绍一下产品", context="产品上下文")

    assert response == "Mock response"
    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert event.method == "generate"
    assert event.prompt_chars == len("介绍一下产品")
    assert event.context_chars == len("产品上下文")
    assert event.response_preview == "Mock response"


@pytest.mark.asyncio
async def test_complete_with_tools_trace_records_tool_calls(mock_llm):
    mock_llm._supports_function_calling = True
    mock_llm.tool_call_queue = [
        ToolCallingResponse(
            content="",
            tool_calls=[ToolCall(name="apply_refund", arguments={"order_id": "10392"})],
            finish_reason="tool_calls",
        )
    ]
    recorder = LLMTraceRecorder(provider="mock", model="test-model")
    llm = TraceableLLMProvider(mock_llm, recorder)

    response = await llm.complete_with_tools(
        prompt="请处理退款",
        tools=[{"function": {"name": "apply_refund"}}],
    )

    assert response.finish_reason == "tool_calls"
    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert event.method == "complete_with_tools"
    assert event.tool_names == ["apply_refund"]
    assert "apply_refund" in event.tool_calls[0]