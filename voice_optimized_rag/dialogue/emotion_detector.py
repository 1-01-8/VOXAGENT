"""
情绪检测器 —— 读取 SenseVoice 情绪标签 + 维护状态机

本模块不运行独立的情绪识别模型，而是：
1. 从 SenseVoice STT 的输出中读取 emotion 字段
2. 维护情绪状态机（追踪连续愤怒轮次等）
3. 在情绪变化时发布 EMOTION_CHANGE 事件

触发转人工的情绪条件由 TransferPolicy 统一判断。
"""

from __future__ import annotations

from voice_optimized_rag.core.conversation_stream import (
    ConversationStream,
    EventType,
    StreamEvent,
)
from voice_optimized_rag.dialogue.session import EmotionState, SessionContext
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("emotion_detector")

# SenseVoice 原始标签到系统标准标签的映射
_SENSEVOICE_TO_EMOTION: dict[str, EmotionState] = {
    "neutral": EmotionState.NEUTRAL,
    "happy": EmotionState.HAPPY,
    "angry": EmotionState.ANGRY,
    "sad": EmotionState.SAD,
    "confused": EmotionState.CONFUSED,
}


class EmotionDetector:
    """
    情绪状态机

    每次 STT 返回结果后调用 update()，传入 SenseVoice 的情绪标签。
    当情绪发生变化时，自动发布 EMOTION_CHANGE 事件到对话事件流。
    """

    def __init__(self, stream: ConversationStream) -> None:
        self._stream = stream

    async def update(
        self,
        raw_emotion: str,
        session: SessionContext,
    ) -> EmotionState:
        """
        更新情绪状态

        Args:
            raw_emotion: SenseVoice 输出的情绪标签字符串
            session: 当前会话上下文

        Returns:
            更新后的 EmotionState
        """
        new_emotion = _SENSEVOICE_TO_EMOTION.get(
            raw_emotion.lower(), EmotionState.NEUTRAL
        )
        old_emotion = session.emotion

        session.update_emotion(new_emotion)

        if new_emotion != old_emotion:
            logger.info(f"Emotion changed: {old_emotion.value} → {new_emotion.value}")
            await self._stream.publish(StreamEvent(
                event_type=EventType.EMOTION_CHANGE,
                text=new_emotion.value,
                metadata={
                    "previous": old_emotion.value,
                    "consecutive_angry": session.consecutive_angry_turns,
                },
            ))

        return new_emotion
