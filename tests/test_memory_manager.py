"""Tests for dialogue/memory_manager.py — 多轮记忆管理模块

系统组件: Dialogue — MemoryManager 多轮记忆管理
源文件:   voice_optimized_rag/dialogue/memory_manager.py
职责:     管理短期记忆 + LLM 压缩摘要，为 Agent 提供上下文

测试覆盖：
- 短期记忆保留最近 N 轮
- 超过 N 轮时自动触发 LLM 压缩
- 压缩失败时的兜底策略
- get_context() 输出格式
- clear() 清空所有记忆
"""

from __future__ import annotations

import pytest

from voice_optimized_rag.dialogue.memory_manager import MemoryManager
from voice_optimized_rag.dialogue.session import SessionContext


@pytest.fixture
def memory(mock_llm) -> MemoryManager:
    """使用小窗口（3轮）方便测试压缩触发"""
    return MemoryManager(mock_llm, short_term_turns=3)


class TestMemoryManager:
    """记忆管理器测试"""

    @pytest.mark.asyncio
    async def test_add_within_window(self, memory: MemoryManager, session: SessionContext):
        """窗口内的对话应全部保留在短期记忆"""
        await memory.add_turn("user", "你好", session)
        await memory.add_turn("assistant", "您好", session)
        assert memory.turn_count == 2
        ctx = memory.get_context()
        assert "你好" in ctx
        assert "您好" in ctx

    @pytest.mark.asyncio
    async def test_compression_triggered(self, memory: MemoryManager, mock_llm, session: SessionContext):
        """超过窗口大小时应触发 LLM 压缩"""
        mock_llm._response = "用户咨询了产品信息"  # 模拟压缩摘要
        # 添加 4 轮（超过 short_term_turns=3）
        for i in range(4):
            await memory.add_turn("user", f"问题{i}", session)

        assert mock_llm.call_count >= 1  # 应调用 LLM 压缩
        assert memory.summary  # 应有摘要
        assert session.history_summary  # 应同步到 session

    @pytest.mark.asyncio
    async def test_compression_failure_fallback(self, memory: MemoryManager, mock_llm, session: SessionContext):
        """LLM 压缩失败时应用拼接兜底"""
        async def _raise(*a, **kw):
            raise RuntimeError("LLM down")

        mock_llm.generate = _raise

        # 添加 4 轮触发压缩
        for i in range(4):
            await memory.add_turn("user", f"问题{i}", session)

        # 应有兜底摘要（虽然不是 LLM 生成的）
        assert memory.summary  # 不为空即可

    @pytest.mark.asyncio
    async def test_get_context_format(self, memory: MemoryManager, mock_llm, session: SessionContext):
        """get_context 应包含摘要和近期对话两部分"""
        mock_llm._response = "历史摘要内容"

        for i in range(5):
            await memory.add_turn("user", f"第{i}句", session)

        ctx = memory.get_context()
        assert "[对话摘要]" in ctx
        assert "[近期对话]" in ctx

    @pytest.mark.asyncio
    async def test_clear_resets_all(self, memory: MemoryManager, session: SessionContext):
        """clear 应清空所有记忆"""
        await memory.add_turn("user", "hello", session)
        memory.clear()
        assert memory.turn_count == 0
        assert memory.summary == ""
        assert memory.get_context() == ""
