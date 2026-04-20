"""Tests for dialogue/transfer_policy.py — 转人工策略模块

系统组件: Dialogue — TransferPolicy 转人工策略
源文件:   voice_optimized_rag/dialogue/transfer_policy.py
职责:     三条件触发转人工（愤怒阈值/Agent失败/高风险词）

测试覆盖：
- 持续愤怒达到阈值触发转人工
- Agent 连续失败触发转人工
- 高风险关键词触发转人工
- 正常对话不触发转人工
- 已标记 transfer_requested 直接返回 True
- 转人工后发布 TRANSFER_REQUEST 事件
"""

from __future__ import annotations

import pytest

from voice_optimized_rag.core.conversation_stream import ConversationStream, EventType
from voice_optimized_rag.dialogue.session import EmotionState, SessionContext
from voice_optimized_rag.dialogue.transfer_policy import TransferPolicy


@pytest.fixture
def policy(stream: ConversationStream) -> TransferPolicy:
    return TransferPolicy(stream, angry_threshold=2, max_agent_failures=3)


class TestTransferPolicy:
    """转人工策略测试"""

    @pytest.mark.asyncio
    async def test_normal_no_transfer(self, policy: TransferPolicy, session: SessionContext):
        """正常对话不应触发转人工"""
        result = await policy.evaluate(session, "你好，我想买个东西")
        assert result is False
        assert session.transfer_requested is False

    @pytest.mark.asyncio
    async def test_angry_threshold(self, policy: TransferPolicy, session: SessionContext, stream: ConversationStream):
        """连续愤怒 >= 阈值应触发转人工"""
        session.update_emotion(EmotionState.ANGRY)
        session.update_emotion(EmotionState.ANGRY)
        assert session.consecutive_angry_turns == 2  # == threshold

        result = await policy.evaluate(session, "")
        assert result is True
        assert "持续愤怒" in session.transfer_reason

        # 应发布 TRANSFER_REQUEST 事件
        assert any(e.event_type == EventType.TRANSFER_REQUEST for e in stream.history)

    @pytest.mark.asyncio
    async def test_agent_failures(self, policy: TransferPolicy, session: SessionContext):
        """Agent 连续失败 >= 3 应触发转人工"""
        for _ in range(3):
            session.record_agent_failure()
        result = await policy.evaluate(session, "")
        assert result is True
        assert "连续失败" in session.transfer_reason

    @pytest.mark.asyncio
    async def test_high_risk_keyword(self, policy: TransferPolicy, session: SessionContext):
        """高风险关键词应立即触发转人工"""
        result = await policy.evaluate(session, "我要找律师起诉你们")
        assert result is True
        assert "高风险关键词" in session.transfer_reason

    @pytest.mark.asyncio
    async def test_already_requested(self, policy: TransferPolicy, session: SessionContext):
        """已标记 transfer_requested 应直接返回 True"""
        session.transfer_requested = True
        result = await policy.evaluate(session, "随便说什么")
        assert result is True

    @pytest.mark.asyncio
    async def test_angry_below_threshold(self, policy: TransferPolicy, session: SessionContext):
        """愤怒轮次未达阈值不应触发"""
        session.update_emotion(EmotionState.ANGRY)  # 仅 1 轮 < 阈值 2
        result = await policy.evaluate(session, "")
        assert result is False

    @pytest.mark.asyncio
    async def test_vip_angry_triggers_transfer(self, policy: TransferPolicy, session: SessionContext):
        """VIP 客户情绪不佳应触发转人工（即使愤怒轮次未达阈值）"""
        session.user_profile.vip_level = 3
        session.update_emotion(EmotionState.ANGRY)  # 仅 1 轮，低于普通阈值
        result = await policy.evaluate(session, "")
        assert result is True
        assert "VIP" in session.transfer_reason

    @pytest.mark.asyncio
    async def test_vip_neutral_no_transfer(self, policy: TransferPolicy, session: SessionContext):
        """VIP 客户情绪正常不应触发转人工"""
        session.user_profile.vip_level = 5
        session.update_emotion(EmotionState.NEUTRAL)
        result = await policy.evaluate(session, "")
        assert result is False
