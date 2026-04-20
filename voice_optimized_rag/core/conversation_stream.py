"""
对话事件流 —— 异步事件总线

职责：
- 作为 Memory Router（发布者）和 Slow Thinker（订阅者）之间的通信管道
- 维护最近 N 轮对话的滑动窗口历史，供预测和上下文使用
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator


class EventType(Enum):
    """对话事件类型枚举"""
    USER_UTTERANCE = "user_utterance"        # 用户说话（最频繁，触发预取）
    AGENT_RESPONSE = "agent_response"        # Agent 回复完成（写入历史）
    SILENCE_DETECTED = "silence_detected"   # 检测到静音（利用空档预取更多内容）
    TOPIC_SHIFT = "topic_shift"             # 话题切换（清除旧缓存，预取新话题）
    PRIORITY_RETRIEVAL = "priority_retrieval"  # 紧急检索（Fast Talker 缓存未命中时发出）
    CONFIRM_REQUIRED = "confirm_required"    # Agent 请求用户二次确认
    CONFIRM_RESPONSE = "confirm_response"    # 用户确认结果（yes/no）
    EMOTION_CHANGE = "emotion_change"        # 情绪状态变化
    TRANSFER_REQUEST = "transfer_request"    # 触发转人工


@dataclass
class StreamEvent:
    """对话流中的单个事件"""
    event_type: EventType
    text: str = ""                                      # 事件携带的文本内容
    metadata: dict = field(default_factory=dict)        # 附加元数据（如置信度、来源等）
    timestamp: float = field(default_factory=time.time) # 事件发生的时间戳


class ConversationStream:
    """
    异步事件总线，带滑动窗口对话历史

    设计模式：发布-订阅（Pub-Sub）
    - Memory Router 调用 publish() 推送事件
    - Slow Thinker 调用 subscribe() 获取异步迭代器，持续监听事件
    - 每个订阅者拥有独立的 asyncio.Queue，互不影响

    滑动窗口：使用 collections.deque(maxlen=N) 自动丢弃最旧记录，
    保留最近 window_size 条事件作为对话上下文。
    """

    def __init__(self, window_size: int = 10) -> None:
        # 所有订阅者的队列列表（每个订阅者一个独立 Queue）
        self._subscribers: list[asyncio.Queue[StreamEvent]] = []
        # 滑动窗口历史：deque 满后自动从左侧丢弃最旧事件
        self._history: deque[StreamEvent] = deque(maxlen=window_size)
        self._window_size = window_size

    async def publish(self, event: StreamEvent) -> None:
        """
        发布事件到所有订阅者，同时写入历史记录

        发布是广播式的：每个订阅者的队列都会收到这个事件。
        使用 await queue.put() 而非 put_nowait()，确保背压传播（队列满时等待）。
        """
        self._history.append(event)                    # 写入滑动窗口历史
        for queue in self._subscribers:
            await queue.put(event)                     # 广播给每个订阅者

    def subscribe(self) -> AsyncIterator[StreamEvent]:
        """
        创建一个新订阅，返回异步迭代器

        每次调用 subscribe() 都会创建一个新的独立队列，
        订阅者通过 async for event in subscription 持续接收事件。
        """
        queue: asyncio.Queue[StreamEvent] = asyncio.Queue()
        self._subscribers.append(queue)
        # _SubscriptionIterator 封装了队列的 async for 逻辑，并在取消时自动清理
        return _SubscriptionIterator(queue, self._subscribers)

    @property
    def history(self) -> list[StreamEvent]:
        """返回当前滑动窗口内的全部历史事件"""
        return list(self._history)

    def get_conversation_text(self, max_turns: int | None = None) -> str:
        """
        将近期对话格式化为文本，供 LLM Prompt 使用

        只提取 USER_UTTERANCE 和 AGENT_RESPONSE 两类事件，
        格式化为 "User: ...\nAssistant: ..." 的多轮对话文本。

        Args:
            max_turns: 最多取最近 N 条事件（None 表示全部窗口内容）

        Returns:
            格式化后的对话字符串
        """
        events = list(self._history)
        if max_turns:
            events = events[-max_turns:]   # 取最近 N 条（切片从右端取）

        lines: list[str] = []
        for event in events:
            if event.event_type == EventType.USER_UTTERANCE:
                lines.append(f"User: {event.text}")
            elif event.event_type == EventType.AGENT_RESPONSE:
                lines.append(f"Assistant: {event.text}")
        return "\n".join(lines)

    def clear(self) -> None:
        """清空对话历史（话题大幅跳跃时可调用）"""
        self._history.clear()


class _SubscriptionIterator:
    """
    订阅队列的异步迭代器

    实现了 Python 异步迭代器协议（__aiter__ + __anext__），
    使得 Slow Thinker 可以用 async for event in subscription 持续消费事件。

    在 asyncio.CancelledError（任务被取消）时自动从订阅列表中移除自身，
    防止 Memory Router 继续向已停止的订阅者发送事件。
    """

    def __init__(
        self,
        queue: asyncio.Queue[StreamEvent],
        subscribers: list[asyncio.Queue[StreamEvent]],
    ) -> None:
        self._queue = queue                # 本订阅者专属的事件队列
        self._subscribers = subscribers   # 全局订阅者列表（用于退订时删除自身）

    def __aiter__(self) -> _SubscriptionIterator:
        return self  # 异步迭代器自身即为迭代器

    async def __anext__(self) -> StreamEvent:
        """等待队列中的下一个事件（阻塞直到有事件到来）"""
        try:
            return await self._queue.get()
        except asyncio.CancelledError:
            # 任务被取消（如 SlowThinker.stop() 被调用）时，从全局列表移除自身
            if self._queue in self._subscribers:
                self._subscribers.remove(self._queue)
            raise  # 重新抛出，让调用方的 try/except CancelledError 捕获

    async def unsubscribe(self) -> None:
        """主动退订，将此队列从全局订阅者列表中移除"""
        if self._queue in self._subscribers:
            self._subscribers.remove(self._queue)
