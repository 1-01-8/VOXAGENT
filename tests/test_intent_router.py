"""Tests for dialogue/intent_router.py — 意图路由模块

系统组件: Dialogue — IntentRouter 三路意图路由
源文件:   voice_optimized_rag/dialogue/intent_router.py
职责:     将用户输入分类为 task / knowledge / chitchat，触发转人工

测试覆盖：
- 任务型关键词快速匹配
- 知识型关键词快速匹配
- 转人工关键词检测（优先级最高）
- LLM 兜底分类
- LLM 异常时降级到知识检索
"""

from __future__ import annotations

import pytest

from voice_optimized_rag.dialogue.intent_router import IntentRouter
from voice_optimized_rag.dialogue.session import IntentType, SessionContext


@pytest.fixture
def router(mock_llm) -> IntentRouter:
    return IntentRouter(mock_llm)


class TestIntentRouter:
    """意图路由核心测试"""

    @pytest.mark.asyncio
    async def test_task_keyword_match(self, router: IntentRouter, session: SessionContext):
        """包含任务型关键词应直接路由到 TASK"""
        intent = await router.classify("我想查订单", session)
        assert intent == IntentType.TASK

    @pytest.mark.asyncio
    async def test_knowledge_keyword_match(self, router: IntentRouter, session: SessionContext):
        """包含知识型关键词应路由到 KNOWLEDGE"""
        intent = await router.classify("这个产品多少钱", session)
        assert intent == IntentType.KNOWLEDGE

    @pytest.mark.asyncio
    async def test_transfer_keyword_sets_flag(self, router: IntentRouter, session: SessionContext):
        """转人工关键词应设置 transfer_requested 标记"""
        intent = await router.classify("我要转人工", session)
        assert intent == IntentType.CHITCHAT  # 转人工归为 chitchat
        assert session.transfer_requested is True
        assert "转人工" in session.transfer_reason

    @pytest.mark.asyncio
    async def test_llm_fallback_task(self, router: IntentRouter, mock_llm, session: SessionContext):
        """无关键词命中时应调用 LLM 分类"""
        mock_llm._response = "task"
        intent = await router.classify("帮我处理一下这个事情", session)
        assert intent == IntentType.TASK
        assert mock_llm.call_count >= 1

    @pytest.mark.asyncio
    async def test_llm_fallback_knowledge(self, router: IntentRouter, mock_llm, session: SessionContext):
        """LLM 返回 knowledge 应路由到 KNOWLEDGE"""
        mock_llm._response = "knowledge"
        intent = await router.classify("这个东西好不好用", session)
        assert intent == IntentType.KNOWLEDGE

    @pytest.mark.asyncio
    async def test_llm_fallback_chitchat(self, router: IntentRouter, mock_llm, session: SessionContext):
        """LLM 返回 chitchat 或不明确应路由到 CHITCHAT"""
        mock_llm._response = "chitchat"
        intent = await router.classify("你好啊", session)
        assert intent == IntentType.CHITCHAT

    @pytest.mark.asyncio
    async def test_llm_error_fallback(self, router: IntentRouter, mock_llm, session: SessionContext):
        """LLM 调用失败时应安全降级到 KNOWLEDGE"""
        mock_llm._response = "should not reach"

        # 用一个会抛异常的 LLM 替代
        async def _raise(*args, **kwargs):
            raise RuntimeError("LLM down")

        router._llm.generate = _raise
        intent = await router.classify("这是个复杂的问题", session)
        assert intent == IntentType.KNOWLEDGE  # 安全降级

    @pytest.mark.asyncio
    async def test_transfer_keyword_priority(self, router: IntentRouter, session: SessionContext):
        """转人工关键词优先级应高于任务关键词"""
        # "投诉" 同时出现在 TASK_KEYWORDS 和 TRANSFER_KEYWORDS 中
        intent = await router.classify("我要投诉，找你们经理", session)
        # 转人工关键词 "找你们经理" 应优先命中
        assert session.transfer_requested is True
