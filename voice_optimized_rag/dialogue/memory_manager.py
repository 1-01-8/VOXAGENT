"""
多轮记忆管理 —— 短期/中期/长期三级记忆

- 短期记忆：保留最近 N 轮对话原文，直接作为 LLM 上下文
- 中期摘要：超过 N 轮后，调用 LLM 将旧对话压缩为摘要
- 长期存储：会话结束后，关键信息异步写入持久存储（V2 对接 CRM）

设计原则：节省 Token 消耗的同时保留关键信息。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from voice_optimized_rag.dialogue.session import SessionContext
from voice_optimized_rag.llm.base import LLMProvider
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("memory_manager")

COMPRESS_PROMPT = """请将以下多轮对话压缩为一段简洁的摘要，保留关键信息（用户需求、已确认的信息、待解决的问题）。

对话内容：
{conversation}

已有摘要：
{existing_summary}

请输出更新后的摘要（200字以内）："""


@dataclass
class MemoryTurn:
    """单轮对话记录"""
    role: str       # "user" or "assistant"
    content: str
    emotion: str = ""
    turn_index: int = 0


class MemoryManager:
    """
    三级记忆管理器

    管理策略：
    ┌─────────────────────────────────────────────────┐
    │  最近 N 轮（短期）  │  原文保留，直接拼入 Prompt  │
    ├─────────────────────────────────────────────────┤
    │  N 轮之前（中期）   │  LLM 压缩为摘要            │
    ├─────────────────────────────────────────────────┤
    │  会话结束后（长期）  │  关键信息写入持久存储       │
    └─────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        llm: LLMProvider,
        short_term_turns: int = 10,
    ) -> None:
        self._llm = llm
        self._short_term_turns = short_term_turns
        self._turns: deque[MemoryTurn] = deque()
        self._turn_counter: int = 0
        self._summary: str = ""

    async def add_turn(
        self,
        role: str,
        content: str,
        session: SessionContext,
        emotion: str = "",
    ) -> None:
        """
        添加一轮对话

        如果短期记忆已满（超过 N 轮），自动触发压缩。
        """
        self._turn_counter += 1
        turn = MemoryTurn(
            role=role,
            content=content,
            emotion=emotion,
            turn_index=self._turn_counter,
        )
        self._turns.append(turn)

        # 超过短期记忆容量时，压缩旧对话
        if len(self._turns) > self._short_term_turns:
            await self._compress_old_turns(session)

    async def _compress_old_turns(self, session: SessionContext) -> None:
        """将超出短期窗口的旧对话压缩为摘要"""
        # 取出需要压缩的轮次（保留最近 N 轮）
        turns_to_compress: list[MemoryTurn] = []
        while len(self._turns) > self._short_term_turns:
            turns_to_compress.append(self._turns.popleft())

        if not turns_to_compress:
            return

        # 格式化要压缩的对话
        conversation_text = "\n".join(
            f"{'用户' if t.role == 'user' else '客服'}: {t.content}"
            for t in turns_to_compress
        )

        try:
            prompt = COMPRESS_PROMPT.format(
                conversation=conversation_text,
                existing_summary=self._summary or "（无）",
            )
            self._summary = await self._llm.generate(prompt)
            session.history_summary = self._summary
            logger.info(f"Compressed {len(turns_to_compress)} turns into summary")
        except Exception as e:
            logger.warning(f"Memory compression failed: {e}")
            # 压缩失败时，用简单拼接作为兜底
            self._summary += f"\n{conversation_text[:200]}..."

    def get_context(self) -> str:
        """
        获取完整的记忆上下文（摘要 + 短期原文）

        返回格式：
            [对话摘要] 之前的对话摘要内容...

            [近期对话]
            用户: ...
            客服: ...
        """
        parts: list[str] = []

        if self._summary:
            parts.append(f"[对话摘要] {self._summary}")

        if self._turns:
            recent = "\n".join(
                f"{'用户' if t.role == 'user' else '客服'}: {t.content}"
                for t in self._turns
            )
            parts.append(f"[近期对话]\n{recent}")

        return "\n\n".join(parts)

    @property
    def summary(self) -> str:
        """当前对话摘要"""
        return self._summary

    @property
    def turn_count(self) -> int:
        """总对话轮次"""
        return self._turn_counter

    def clear(self) -> None:
        """清空所有记忆（会话结束时调用）"""
        self._turns.clear()
        self._summary = ""
        self._turn_counter = 0
