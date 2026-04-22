"""Tests for utils/session_logger.py — 会话日志模块

系统组件: Utils — SessionLogger 会话日志(JSONL)
源文件:   voice_optimized_rag/utils/session_logger.py
职责:     异步写入 JSONL 格式的完整对话日志，支撑离线分析与质检

测试覆盖：
- 日志文件创建
- 事件日志写入（JSON Lines 格式）
- 对话轮次记录
- 转人工记录
- 会话结束记录
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from voice_optimized_rag.utils.session_logger import SessionLogger


@pytest.fixture
def logger_dir(tmp_path: Path) -> Path:
    """使用临时目录作为日志目录"""
    return tmp_path / "test_logs"


@pytest.fixture
def slogger(logger_dir: Path) -> SessionLogger:
    return SessionLogger(str(logger_dir))


class TestSessionLogger:
    """会话日志测试"""

    @pytest.mark.asyncio
    async def test_log_creates_file(self, slogger: SessionLogger, logger_dir: Path):
        """写入日志应创建 JSONL 文件"""
        await slogger.log_event("sess-001", "test", {"key": "value"})
        files = list(logger_dir.glob("*.jsonl"))
        assert len(files) == 1

    @pytest.mark.asyncio
    async def test_log_turn_format(self, slogger: SessionLogger, logger_dir: Path):
        """对话轮次日志应包含所有必要字段"""
        await slogger.log_turn(
            session_id="sess-002",
            turn_index=1,
            user_text="你好",
            agent_text="您好",
            emotion="neutral",
            intent="out_of_scope",
        )

        files = list(logger_dir.glob("*.jsonl"))
        content = files[0].read_text(encoding="utf-8").strip()
        data = json.loads(content)
        assert data["event_type"] == "turn"
        assert data["user_text"] == "你好"
        assert data["agent_text"] == "您好"
        assert data["emotion"] == "neutral"
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_log_transfer(self, slogger: SessionLogger, logger_dir: Path):
        """转人工日志应记录原因"""
        await slogger.log_transfer("sess-003", "用户持续愤怒", "用户投诉产品质量")
        files = list(logger_dir.glob("*.jsonl"))
        data = json.loads(files[0].read_text(encoding="utf-8").strip())
        assert data["event_type"] == "transfer"
        assert "愤怒" in data["reason"]

    @pytest.mark.asyncio
    async def test_log_session_end(self, slogger: SessionLogger, logger_dir: Path):
        """会话结束日志应记录总轮次"""
        await slogger.log_session_end("sess-004", total_turns=12, resolution="resolved")
        files = list(logger_dir.glob("*.jsonl"))
        data = json.loads(files[0].read_text(encoding="utf-8").strip())
        assert data["event_type"] == "session_end"
        assert data["total_turns"] == 12

    @pytest.mark.asyncio
    async def test_multiple_events_same_session(self, slogger: SessionLogger, logger_dir: Path):
        """同一会话的多条日志应写入同一文件"""
        await slogger.log_event("sess-005", "start", {})
        await slogger.log_turn("sess-005", 1, "Q1", "A1")
        await slogger.log_turn("sess-005", 2, "Q2", "A2")
        await slogger.log_session_end("sess-005", 2)

        files = list(logger_dir.glob("*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 4
