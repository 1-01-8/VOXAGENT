"""
会话日志落库 —— 异步写入会话全程记录

记录内容：
- 每轮对话文本（用户 + 客服）
- 情绪标签变化
- 意图路由决策
- 工具调用序列
- 转人工事件

日志格式为 JSON Lines，按会话 ID 分文件存储。
供质检和迭代优化使用。
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("session_logger")


class SessionLogger:
    """
    异步会话日志记录器

    每个会话一个日志文件（JSON Lines 格式），异步追加写入。
    """

    def __init__(self, log_dir: str = "logs/sessions") -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._writers: dict[str, Path] = {}

    def _get_log_path(self, session_id: str) -> Path:
        """获取会话日志文件路径"""
        if session_id not in self._writers:
            date_prefix = time.strftime("%Y%m%d")
            log_path = self._log_dir / f"{date_prefix}_{session_id[:8]}.jsonl"
            self._writers[session_id] = log_path
        return self._writers[session_id]

    async def log_event(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        """
        异步记录一条事件日志

        Args:
            session_id: 会话 ID
            event_type: 事件类型
            data: 事件数据
        """
        log_entry = {
            "timestamp": time.time(),
            "session_id": session_id,
            "event_type": event_type,
            **data,
        }

        log_path = self._get_log_path(session_id)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._write_line, log_path, log_entry)

    async def log_turn(
        self,
        session_id: str,
        turn_index: int,
        user_text: str,
        agent_text: str,
        emotion: str = "",
        intent: str = "",
        tools_called: list[str] | None = None,
    ) -> None:
        """记录一轮完整对话"""
        await self.log_event(session_id, "turn", {
            "turn_index": turn_index,
            "user_text": user_text,
            "agent_text": agent_text,
            "emotion": emotion,
            "intent": intent,
            "tools_called": tools_called or [],
        })

    async def log_transfer(
        self,
        session_id: str,
        reason: str,
        context_summary: str = "",
    ) -> None:
        """记录转人工事件"""
        await self.log_event(session_id, "transfer", {
            "reason": reason,
            "context_summary": context_summary,
        })

    async def log_session_end(
        self,
        session_id: str,
        total_turns: int,
        resolution: str = "",
    ) -> None:
        """记录会话结束"""
        await self.log_event(session_id, "session_end", {
            "total_turns": total_turns,
            "resolution": resolution,
        })

    @staticmethod
    def _write_line(path: Path, data: dict) -> None:
        """同步写入一行 JSON（在 executor 中运行）"""
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
