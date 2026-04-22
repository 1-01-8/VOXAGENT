"""Tests for agent/function_calling_agent.py — 原生函数调用 Agent。"""

from __future__ import annotations

import pytest

from tests.conftest import MockTool
from voice_optimized_rag.agent.base_tool import BaseTool, ToolParameter, ToolResult
from voice_optimized_rag.agent.tools.finance_tools import ApplyRefundTool
from voice_optimized_rag.agent.function_calling_agent import FunctionCallingAgent
from voice_optimized_rag.agent.permission_guard import PermissionGuard
from voice_optimized_rag.core.conversation_stream import EventType
from voice_optimized_rag.dialogue.session import AgentDomain, IntentType, SessionContext, SlotInfo, TaskStatus
from voice_optimized_rag.llm.base import ToolCall, ToolCallingResponse


class TestFunctionCallingAgent:
    @pytest.mark.asyncio
    async def test_native_tool_call_flow(self, mock_llm, stream):
        mock_llm._supports_function_calling = True
        mock_llm.tool_call_queue = [
            ToolCallingResponse(tool_calls=[
                ToolCall(name="query_order", arguments={"order_id": "ORD-001"})
            ]),
            ToolCallingResponse(content="订单 ORD-001 已发货。"),
        ]

        tool = MockTool(name="query_order", description="查询订单")
        guard = PermissionGuard(stream, confirm_timeout=0.2)
        agent = FunctionCallingAgent(
            llm=mock_llm,
            tools=[tool],
            permission_guard=guard,
            stream=stream,
            max_iterations=4,
            tool_retry=0,
        )

        session = SessionContext()
        result = await agent.execute("帮我查订单 ORD-001", session)

        assert "已发货" in result
        assert tool.call_count == 1
        assert session.task_status == TaskStatus.COMPLETED
        assert mock_llm.last_tools[0]["function"]["name"] == "query_order"

    @pytest.mark.asyncio
    async def test_clean_final_answer_prefix(self, mock_llm, stream):
        mock_llm._supports_function_calling = True
        mock_llm.tool_call_queue = [
            ToolCallingResponse(content="Final Answer: 退款申请已提交。"),
        ]

        tool = MockTool(name="apply_refund", description="退款")
        guard = PermissionGuard(stream, confirm_timeout=0.2)
        agent = FunctionCallingAgent(
            llm=mock_llm,
            tools=[tool],
            permission_guard=guard,
            stream=stream,
            max_iterations=3,
            tool_retry=0,
        )

        session = SessionContext()
        result = await agent.execute("我要退款", session)

        assert result == "退款申请已提交。"

    @pytest.mark.asyncio
    async def test_fallback_to_react_when_native_not_supported(self, mock_llm, stream):
        mock_llm._response = "Thought: 已完成\nFinal Answer: 当前有优惠活动。"
        tool = MockTool(name="check_promotion", description="查询促销")
        guard = PermissionGuard(stream, confirm_timeout=0.2)
        agent = FunctionCallingAgent(
            llm=mock_llm,
            tools=[tool],
            permission_guard=guard,
            stream=stream,
            max_iterations=3,
            tool_retry=0,
        )

        session = SessionContext()
        result = await agent.execute("现在有什么优惠", session)

        assert "优惠活动" in result
        assert "Final Answer" in mock_llm.last_prompt

    @pytest.mark.asyncio
    async def test_missing_params_do_not_trigger_confirmation(self, mock_llm, stream):
        mock_llm._supports_function_calling = True
        mock_llm.tool_call_queue = [
            ToolCallingResponse(tool_calls=[
                ToolCall(name="apply_refund", arguments={})
            ]),
            ToolCallingResponse(content="请提供订单号和退款原因。"),
        ]

        guard = PermissionGuard(stream, confirm_timeout=0.2)
        agent = FunctionCallingAgent(
            llm=mock_llm,
            tools=[ApplyRefundTool()],
            permission_guard=guard,
            stream=stream,
            max_iterations=4,
            tool_retry=0,
        )

        session = SessionContext()
        result = await agent.execute("我要退款", session)

        assert "订单号" in result
        confirm_events = [event for event in stream.history if event.event_type == EventType.CONFIRM_REQUIRED]
        assert confirm_events == []

    @pytest.mark.asyncio
    async def test_session_slots_fill_missing_tool_args(self, mock_llm, stream):
        class RefundDraftTool(BaseTool):
            def __init__(self) -> None:
                self.calls: list[dict] = []

            @property
            def name(self) -> str:
                return "apply_refund"

            @property
            def description(self) -> str:
                return "提交退款申请"

            @property
            def permission_level(self) -> int:
                return 1

            @property
            def parameters(self) -> list[ToolParameter]:
                return [
                    ToolParameter(name="order_id", description="订单号"),
                    ToolParameter(name="reason", description="退款原因"),
                ]

            async def execute(self, **kwargs):
                self.calls.append(kwargs)
                return ToolResult(success=True, message="退款申请已提交")

        mock_llm._supports_function_calling = True
        mock_llm.tool_call_queue = [
            ToolCallingResponse(tool_calls=[
                ToolCall(name="apply_refund", arguments={})
            ]),
            ToolCallingResponse(content="退款申请已提交"),
        ]

        tool = RefundDraftTool()
        guard = PermissionGuard(stream, confirm_timeout=0.2)
        agent = FunctionCallingAgent(
            llm=mock_llm,
            tools=[tool],
            permission_guard=guard,
            stream=stream,
            max_iterations=4,
            tool_retry=0,
        )

        session = SessionContext(current_intent=IntentType.TASK, current_domain=AgentDomain.FINANCE)
        session.slots["order_id"] = SlotInfo(name="order_id", value="10392")
        result = await agent.execute("因为买错了", session)

        assert result == "退款申请已提交"
        assert tool.calls == [{"order_id": "10392", "reason": "买错了"}]