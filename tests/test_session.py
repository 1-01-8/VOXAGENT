"""Tests for dialogue/session.py — 会话状态管理模块

系统组件: Dialogue — SessionContext 会话状态管理
源文件:   voice_optimized_rag/dialogue/session.py
职责:     管理单次通话的全局状态，包括情绪、意图、槽位、轮次计数

测试覆盖：
- SessionContext 数据结构初始化
- 情绪状态追踪（连续愤怒轮次计数）
- 对话轮次计数
- Agent 失败计数与重置
- 未填充槽位查询
- 序列化 to_dict()
"""

from __future__ import annotations

import pytest

from voice_optimized_rag.dialogue.session import (
    EmotionState,
    IntentType,
    SessionContext,
    SlotInfo,
    TaskStatus,
    UserProfile,
)


class TestSessionContext:
    """SessionContext 核心功能测试"""

    def test_default_initialization(self, session: SessionContext):
        """默认初始化应设置合理的初始值"""
        assert session.session_id  # 应自动生成 UUID
        assert session.emotion == EmotionState.NEUTRAL
        assert session.turn_count == 0
        assert session.agent_failure_count == 0
        assert session.transfer_requested is False

    def test_increment_turn(self, session: SessionContext):
        """轮次计数应正确递增"""
        session.increment_turn()
        session.increment_turn()
        assert session.turn_count == 2

    def test_update_emotion_tracks_history(self, session: SessionContext):
        """情绪更新应记录历史并更新当前状态"""
        session.update_emotion(EmotionState.HAPPY)
        session.update_emotion(EmotionState.ANGRY)
        assert session.emotion == EmotionState.ANGRY
        assert len(session.emotion_history) == 2
        assert session.emotion_history[0] == EmotionState.HAPPY

    def test_consecutive_angry_turns(self, session: SessionContext):
        """连续愤怒轮次应正确累计，非愤怒时重置"""
        session.update_emotion(EmotionState.ANGRY)
        session.update_emotion(EmotionState.ANGRY)
        assert session.consecutive_angry_turns == 2

        # 非愤怒情绪应重置计数
        session.update_emotion(EmotionState.NEUTRAL)
        assert session.consecutive_angry_turns == 0

    def test_agent_failure_tracking(self, session: SessionContext):
        """Agent 失败计数应正确累计和重置"""
        session.record_agent_failure()
        session.record_agent_failure()
        assert session.agent_failure_count == 2

        session.reset_agent_failures()
        assert session.agent_failure_count == 0

    def test_unfilled_slots(self, session: SessionContext):
        """应正确返回未填充的必填槽位"""
        session.slots = {
            "order_id": SlotInfo(name="order_id", value="ORD-001", required=True),
            "phone": SlotInfo(name="phone", value=None, required=True, prompt="请提供手机号"),
            "remark": SlotInfo(name="remark", value=None, required=False),
        }
        unfilled = session.get_unfilled_slots()
        assert len(unfilled) == 1
        assert unfilled[0].name == "phone"

    def test_to_dict_contains_key_fields(self, session: SessionContext):
        """序列化应包含所有关键字段"""
        session.increment_turn()
        session.update_emotion(EmotionState.ANGRY)
        d = session.to_dict()
        assert d["session_id"] == session.session_id
        assert d["turn_count"] == 1
        assert d["emotion"] == "angry"
        assert d["consecutive_angry"] == 1
        assert d["transfer_requested"] is False


class TestEnums:
    """枚举类型测试"""

    def test_emotion_values(self):
        """EmotionState 应包含所有预定义情绪"""
        assert EmotionState.NEUTRAL.value == "neutral"
        assert EmotionState.ANGRY.value == "angry"
        assert EmotionState.CONFUSED.value == "confused"

    def test_intent_values(self):
        """IntentType 应包含纯业务三态"""
        assert IntentType.KNOWLEDGE.value == "knowledge"
        assert IntentType.TASK.value == "task"
        assert IntentType.OUT_OF_SCOPE.value == "out_of_scope"

    def test_task_status_values(self):
        """TaskStatus 应包含完整的状态流转值"""
        statuses = [s.value for s in TaskStatus]
        assert "idle" in statuses
        assert "in_progress" in statuses
        assert "completed" in statuses
        assert "failed" in statuses
