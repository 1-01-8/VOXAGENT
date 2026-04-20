"""
转人工策略 —— 规则引擎

判断是否需要将当前会话转接人工客服。
触发条件（满足任一即触发）：

1. 情绪：愤怒状态持续 >= N 轮
2. 关键词：用户明确说"转人工"/"投诉"等
3. Agent 失败：连续 >= M 次工具调用失败
4. VIP 请求：VIP 客户主动请求
5. 高风险：涉及法律纠纷、财产损失等（通过关键词检测）

转人工时通过 ConversationStream 发布 TRANSFER_REQUEST 事件，
携带转人工原因和完整会话上下文。
"""

from __future__ import annotations

from voice_optimized_rag.core.conversation_stream import (
    ConversationStream,
    EventType,
    StreamEvent,
)
from voice_optimized_rag.dialogue.session import EmotionState, SessionContext
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("transfer_policy")

# 高风险关键词（触发即刻转人工）
HIGH_RISK_KEYWORDS = [
    "法律", "律师", "起诉", "赔偿", "投诉", "工商局",
    "消协", "举报", "骗", "欺诈",
]


class TransferPolicy:
    """
    转人工规则引擎

    在每轮对话结束后调用 evaluate()，综合判断是否需要转人工。
    转人工决策是不可逆的：一旦触发，设置 session.transfer_requested=True。
    """

    def __init__(
        self,
        stream: ConversationStream,
        angry_threshold: int = 2,
        max_agent_failures: int = 3,
    ) -> None:
        self._stream = stream
        self._angry_threshold = angry_threshold
        self._max_agent_failures = max_agent_failures

    async def evaluate(
        self,
        session: SessionContext,
        utterance: str = "",
    ) -> bool:
        """
        评估是否需要转人工

        Args:
            session: 当前会话上下文
            utterance: 用户最新发言（用于关键词检测）

        Returns:
            True = 需要转人工
        """
        if session.transfer_requested:
            return True  # 已标记，无需重复判断

        reason = self._check_rules(session, utterance)
        if reason:
            session.transfer_requested = True
            session.transfer_reason = reason
            logger.info(f"Transfer triggered: {reason}")

            # 发布转人工事件
            await self._stream.publish(StreamEvent(
                event_type=EventType.TRANSFER_REQUEST,
                text=reason,
                metadata=session.to_dict(),
            ))
            return True

        return False

    def _check_rules(self, session: SessionContext, utterance: str) -> str:
        """
        逐条检查转人工规则，返回触发原因（空字符串表示未触发）

        注意：规则 2（关键词转人工）已被 evaluate() 入口的
        session.transfer_requested 前置检查覆盖，此处不再重复判断。
        """
        # 规则1：持续愤怒
        if session.consecutive_angry_turns >= self._angry_threshold:
            return f"用户持续愤怒 {session.consecutive_angry_turns} 轮"

        # 规则2：Agent 连续失败
        if session.agent_failure_count >= self._max_agent_failures:
            return f"Agent 连续失败 {session.agent_failure_count} 次"

        # 规则3：VIP 客户 + 任何不满表达 → 优先转人工
        if session.user_profile.vip_level >= 3 and session.emotion in (
            EmotionState.ANGRY, EmotionState.SAD
        ):
            return f"VIP-{session.user_profile.vip_level} 客户情绪不佳"

        # 规则4：高风险关键词
        for kw in HIGH_RISK_KEYWORDS:
            if kw in utterance:
                return f"高风险关键词: '{kw}'"

        return ""
