"""Tests for dialogue/domain_router.py — 业务域路由模块"""

from __future__ import annotations

import pytest

from voice_optimized_rag.dialogue.domain_router import DomainRouter
from voice_optimized_rag.dialogue.session import AgentDomain, IntentType, SessionContext


@pytest.fixture
def router(mock_llm) -> DomainRouter:
    return DomainRouter(mock_llm)


class TestDomainRouter:
    @pytest.mark.asyncio
    async def test_finance_keyword_match(self, router: DomainRouter, session: SessionContext):
        domain = await router.classify("我想申请退款", session, intent=IntentType.TASK)
        assert domain == AgentDomain.FINANCE
        assert session.current_domain == AgentDomain.FINANCE

    @pytest.mark.asyncio
    async def test_after_sales_keyword_match(self, router: DomainRouter, session: SessionContext):
        domain = await router.classify("帮我查一下订单物流", session, intent=IntentType.TASK)
        assert domain == AgentDomain.AFTER_SALES

    @pytest.mark.asyncio
    async def test_sales_keyword_match(self, router: DomainRouter, session: SessionContext):
        domain = await router.classify("现在有什么优惠活动", session, intent=IntentType.KNOWLEDGE)
        assert domain == AgentDomain.SALES

    @pytest.mark.asyncio
    async def test_goods_catalog_routes_to_sales(self, router: DomainRouter, session: SessionContext):
        domain = await router.classify("商品目录", session, intent=IntentType.KNOWLEDGE)
        assert domain == AgentDomain.SALES

    @pytest.mark.asyncio
    async def test_llm_fallback(self, router: DomainRouter, mock_llm, session: SessionContext):
        mock_llm._response = "after_sales"
        domain = await router.classify("这个事情帮我处理一下", session, intent=IntentType.TASK)
        assert domain == AgentDomain.AFTER_SALES
        assert mock_llm.call_count >= 1

    @pytest.mark.asyncio
    async def test_error_fallback_to_task_default(self, router: DomainRouter, session: SessionContext):
        async def _raise(*args, **kwargs):
            raise RuntimeError("LLM down")

        router._llm.generate = _raise
        domain = await router.classify("帮我处理一下这个单子", session, intent=IntentType.TASK)
        assert domain == AgentDomain.AFTER_SALES

    @pytest.mark.asyncio
    async def test_task_follow_up_reuses_previous_domain(self, router: DomainRouter, session: SessionContext):
        session.turn_count = 2
        session.current_domain = AgentDomain.FINANCE

        domain = await router.classify("我的订单号是10392", session, intent=IntentType.TASK)

        assert domain == AgentDomain.FINANCE

    @pytest.mark.asyncio
    async def test_sales_knowledge_follow_up_reuses_sales_domain(self, router: DomainRouter, session: SessionContext):
        session.turn_count = 2
        session.current_domain = AgentDomain.SALES

        domain = await router.classify("nove", session, intent=IntentType.KNOWLEDGE)

        assert domain == AgentDomain.SALES