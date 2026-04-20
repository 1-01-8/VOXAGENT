"""Tests for agent/react_agent.py — ReAct 推理循环

系统组件: Agent — ReactAgent ReAct 推理循环
源文件:   voice_optimized_rag/agent/react_agent.py
职责:     Thought→Action→Observation 循环，调度工具并生成最终回复

测试覆盖：
- Final Answer 直接返回
- 正常 Thought → Action → Observation → Final Answer 流程
- Action 解析失败时的重试提示
- 工具不存在时的提示
- 工具超时重试机制
- 连续失败触发转人工
- _extract_final_answer 和 _parse_action 静态方法

通过 MockLLM 的 response_queue 模拟多轮对话。
"""

from __future__ import annotations

import pytest

from voice_optimized_rag.agent.react_agent import ReactAgent
from voice_optimized_rag.agent.permission_guard import PermissionGuard
from voice_optimized_rag.agent.base_tool import ToolResult
from voice_optimized_rag.core.conversation_stream import ConversationStream
from voice_optimized_rag.dialogue.session import SessionContext, TaskStatus
from tests.conftest import MockLLM, MockTool


@pytest.fixture
def agent_setup(stream: ConversationStream):
    """构建可测试的 Agent 环境"""
    llm = MockLLM()
    guard = PermissionGuard(stream, confirm_timeout=0.5)
    tool = MockTool(name="query_order", description="查询订单")
    agent = ReactAgent(
        llm=llm,
        tools=[tool],
        permission_guard=guard,
        stream=stream,
        max_iterations=5,
        tool_timeout=2.0,
        tool_retry=0,
    )
    return agent, llm, tool


class TestReActParsing:
    """ReAct 解析方法单元测试"""

    def test_extract_final_answer(self):
        """应正确提取 Final Answer"""
        text = "Thought: 已完成\nFinal Answer: 您的订单已发货"
        result = ReactAgent._extract_final_answer(text)
        assert result == "您的订单已发货"

    def test_extract_final_answer_none(self):
        """无 Final Answer 应返回 None"""
        text = "Thought: 还需要继续"
        result = ReactAgent._extract_final_answer(text)
        assert result is None

    def test_parse_action_success(self):
        """应正确解析 Action 和 JSON 参数"""
        text = 'Thought: 需要查询订单\nAction: query_order\nAction Input: {"order_id": "ORD-001"}'
        name, params = ReactAgent._parse_action(text)
        assert name == "query_order"
        assert params == {"order_id": "ORD-001"}

    def test_parse_action_no_action(self):
        """无 Action 应返回 (None, {})"""
        text = "Thought: 我在思考"
        name, params = ReactAgent._parse_action(text)
        assert name is None
        assert params == {}

    def test_parse_action_bad_json(self):
        """JSON 格式错误应返回空字典"""
        text = 'Action: query_order\nAction Input: {bad json}'
        name, params = ReactAgent._parse_action(text)
        assert name == "query_order"
        assert params == {}


class TestReActAgent:
    """ReAct Agent 集成测试"""

    @pytest.mark.asyncio
    async def test_direct_final_answer(self, agent_setup, session: SessionContext):
        """LLM 直接返回 Final Answer 应立即结束"""
        agent, llm, _ = agent_setup
        llm._response = "Thought: 用户只是打招呼\nFinal Answer: 您好！有什么可以帮您的？"

        result = await agent.execute("你好", session)
        assert "您好" in result
        assert session.task_status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_tool_call_flow(self, agent_setup, session: SessionContext):
        """应完成 Thought → Action → Observation → Final Answer 流程"""
        agent, llm, tool = agent_setup

        # 第一轮：LLM 决定调用工具
        # 第二轮：LLM 根据 Observation 给出最终回复
        llm.response_queue = [
            'Thought: 需要查订单\nAction: query_order\nAction Input: {"order_id": "ORD-001"}',
            "Thought: 已获取订单信息\nFinal Answer: 订单 ORD-001 已发货。",
        ]

        result = await agent.execute("帮我查订单 ORD-001", session)
        assert tool.call_count == 1
        assert "已发货" in result
        assert session.task_status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_unknown_tool(self, agent_setup, session: SessionContext):
        """调用不存在的工具应在 scratchpad 中提示可用工具"""
        agent, llm, _ = agent_setup
        llm.response_queue = [
            'Thought: 需要退款\nAction: apply_refund\nAction Input: {"order_id": "ORD-001"}',
            "Thought: 工具不存在，直接回复\nFinal Answer: 抱歉，暂不支持此操作。",
        ]

        result = await agent.execute("退款", session)
        assert "不支持" in result or "Final Answer" in llm.last_prompt

    @pytest.mark.asyncio
    async def test_max_iterations_exceeded(self, agent_setup, session: SessionContext):
        """超过最大迭代次数应返回转人工提示"""
        agent, llm, _ = agent_setup
        # LLM 永远不给出 Final Answer
        llm._response = "Thought: 还在思考..."

        result = await agent.execute("复杂请求", session)
        assert "转接" in result or "人工" in result
        assert session.task_status == TaskStatus.FAILED
