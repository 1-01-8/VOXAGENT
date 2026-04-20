"""Tests for dialogue/emotion_detector.py — 情绪检测模块

系统组件: Dialogue — EmotionDetector 情绪检测引擎
源文件:   voice_optimized_rag/dialogue/emotion_detector.py
职责:     将 SenseVoice 情绪标签映射为系统枚举，驱动转人工判定

测试覆盖：
- SenseVoice 标签映射到系统标准标签
- 情绪变化时发布 EMOTION_CHANGE 事件
- 情绪不变时不发布事件
- 未知标签降级为 NEUTRAL
"""

from __future__ import annotations

import pytest

from voice_optimized_rag.core.conversation_stream import ConversationStream, EventType
from voice_optimized_rag.dialogue.emotion_detector import EmotionDetector
from voice_optimized_rag.dialogue.session import EmotionState, SessionContext


@pytest.fixture
def detector(stream: ConversationStream) -> EmotionDetector:
    return EmotionDetector(stream)


class TestEmotionDetector:
    """情绪检测器测试"""

    @pytest.mark.asyncio
    async def test_neutral_to_angry(self, detector: EmotionDetector, session: SessionContext, stream: ConversationStream):
        """情绪从 neutral 变为 angry 应更新 session 并发布事件"""
        result = await detector.update("angry", session)
        assert result == EmotionState.ANGRY
        assert session.emotion == EmotionState.ANGRY
        assert session.consecutive_angry_turns == 1
        # 应有事件被发布到 stream
        assert any(e.event_type == EventType.EMOTION_CHANGE for e in stream.history)

    @pytest.mark.asyncio
    async def test_same_emotion_no_event(self, detector: EmotionDetector, session: SessionContext, stream: ConversationStream):
        """情绪不变时不应发布事件"""
        # session 默认 NEUTRAL，再次传入 neutral
        await detector.update("neutral", session)
        emotion_events = [e for e in stream.history if e.event_type == EventType.EMOTION_CHANGE]
        assert len(emotion_events) == 0

    @pytest.mark.asyncio
    async def test_unknown_tag_fallback(self, detector: EmotionDetector, session: SessionContext):
        """未知情绪标签应降级为 NEUTRAL"""
        result = await detector.update("surprised", session)
        assert result == EmotionState.NEUTRAL

    @pytest.mark.asyncio
    async def test_happy_tag(self, detector: EmotionDetector, session: SessionContext):
        """happy 标签应正确映射"""
        result = await detector.update("happy", session)
        assert result == EmotionState.HAPPY
        assert session.emotion == EmotionState.HAPPY

    @pytest.mark.asyncio
    async def test_case_insensitive(self, detector: EmotionDetector, session: SessionContext):
        """大小写不敏感"""
        result = await detector.update("ANGRY", session)
        assert result == EmotionState.ANGRY
