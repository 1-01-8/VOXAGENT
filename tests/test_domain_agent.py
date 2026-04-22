"""Tests for agent/domain_agent.py — 三领域 Agent 工厂"""

from __future__ import annotations

import pytest

from voice_optimized_rag.agent.domain_agent import create_domain_agents
from voice_optimized_rag.agent.permission_guard import PermissionGuard
from voice_optimized_rag.dialogue.session import AgentDomain, SessionContext


class TestDomainAgentFactory:
    @pytest.mark.asyncio
    async def test_create_three_domain_agents(self, mock_llm, stream):
        guard = PermissionGuard(stream, confirm_timeout=0.2)
        agents = create_domain_agents(
            llm=mock_llm,
            permission_guard=guard,
            stream=stream,
            max_iterations=3,
            tool_retry=0,
        )

        assert set(agents.keys()) == {
            AgentDomain.SALES,
            AgentDomain.AFTER_SALES,
            AgentDomain.FINANCE,
        }

    @pytest.mark.asyncio
    async def test_sales_agent_prompt_uses_sales_tools(self, mock_llm, stream):
        guard = PermissionGuard(stream, confirm_timeout=0.2)
        agents = create_domain_agents(
            llm=mock_llm,
            permission_guard=guard,
            stream=stream,
            max_iterations=2,
            tool_retry=0,
        )

        mock_llm._response = "Thought: 已完成\nFinal Answer: 当前有优惠活动。"
        session = SessionContext()
        result = await agents[AgentDomain.SALES].execute("现在有什么优惠", session)

        assert "优惠活动" in result
        assert "销售 Agent" in mock_llm.last_prompt
        assert "check_promotion" in mock_llm.last_prompt
        assert "apply_refund" not in mock_llm.last_prompt

    @pytest.mark.asyncio
    async def test_finance_agent_prompt_uses_finance_tools(self, mock_llm, stream):
        guard = PermissionGuard(stream, confirm_timeout=0.2)
        agents = create_domain_agents(
            llm=mock_llm,
            permission_guard=guard,
            stream=stream,
            max_iterations=2,
            tool_retry=0,
        )

        mock_llm._response = "Thought: 已完成\nFinal Answer: 退款申请已记录。"
        session = SessionContext()
        result = await agents[AgentDomain.FINANCE].execute("帮我申请退款", session)

        assert "退款申请" in result
        assert "财务 Agent" in mock_llm.last_prompt
        assert "apply_refund" in mock_llm.last_prompt
        assert "update_address" not in mock_llm.last_prompt