"""
会话状态管理 —— SessionContext 数据结构

维护单次通话的完整上下文：
- 用户画像（简版，V2 对接 CRM）
- 意图槽位（正在填充的业务参数）
- 情绪状态
- 任务执行状态
- 对话历史摘要
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class EmotionState(str, Enum):
    """情绪状态标签（与 SenseVoice 输出对齐）"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    CONFUSED = "confused"
    ANGRY = "angry"
    SAD = "sad"


class TaskStatus(str, Enum):
    """当前任务执行状态"""
    IDLE = "idle"
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskWorkflow(str, Enum):
    """显式任务工作流类型。"""
    NONE = "none"
    REFUND = "refund"
    CANCEL_ORDER = "cancel_order"
    UPDATE_ADDRESS = "update_address"


class TaskStage(str, Enum):
    """显式任务状态机阶段。"""
    IDLE = "idle"
    COLLECTING = "collecting"
    CONFIRMING = "confirming"
    EXECUTING = "executing"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class IntentType(str, Enum):
    """意图类型（纯业务三态：知识 / 任务 / 超范围）"""
    KNOWLEDGE = "knowledge"      # 知识咨询型 → RAG
    TASK = "task"                # 任务执行型 → Agent
    OUT_OF_SCOPE = "out_of_scope"  # 非业务请求 → 拒答并引导


class AgentDomain(str, Enum):
    """业务域类型（三领域 Agent 路由）"""
    SALES = "sales"
    AFTER_SALES = "after_sales"
    FINANCE = "finance"


@dataclass
class SlotInfo:
    """意图槽位（业务参数收集）"""
    name: str
    value: Any = None
    required: bool = True
    prompt: str = ""  # 缺失时的追问话术


@dataclass
class UserProfile:
    """用户画像（简版，V2 对接 CRM 后扩展）"""
    user_id: str = ""
    name: str = ""
    vip_level: int = 0
    tags: list[str] = field(default_factory=list)


@dataclass
class SessionContext:
    """
    单次通话的完整会话上下文

    由 MemoryRouter 在通话开始时创建，贯穿整个通话生命周期。
    所有模块（意图路由、Agent、情绪检测、转人工）共享此上下文。
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    # 用户信息
    user_profile: UserProfile = field(default_factory=UserProfile)

    # 意图与槽位
    current_intent: IntentType = IntentType.KNOWLEDGE
    current_domain: AgentDomain = AgentDomain.SALES
    slots: dict[str, SlotInfo] = field(default_factory=dict)

    # 情绪追踪（历史限制最近 50 轮，避免长会话内存泄漏）
    emotion: EmotionState = EmotionState.NEUTRAL
    emotion_history: list[EmotionState] = field(default_factory=list)
    consecutive_angry_turns: int = 0
    _emotion_history_limit: int = 50

    # 任务状态
    task_status: TaskStatus = TaskStatus.IDLE
    task_description: str = ""
    active_workflow: TaskWorkflow = TaskWorkflow.NONE
    workflow_stage: TaskStage = TaskStage.IDLE

    # 对话统计
    turn_count: int = 0
    agent_failure_count: int = 0

    # 转人工标记
    transfer_requested: bool = False
    transfer_reason: str = ""

    # 对话历史摘要（多轮压缩后的摘要文本）
    history_summary: str = ""

    def update_emotion(self, new_emotion: EmotionState) -> None:
        """更新情绪状态，追踪连续愤怒轮次"""
        self.emotion_history.append(new_emotion)
        # 限制历史长度防止内存泄漏
        if len(self.emotion_history) > self._emotion_history_limit:
            self.emotion_history = self.emotion_history[-self._emotion_history_limit:]
        self.emotion = new_emotion
        if new_emotion == EmotionState.ANGRY:
            self.consecutive_angry_turns += 1
        else:
            self.consecutive_angry_turns = 0

    def increment_turn(self) -> None:
        """增加对话轮次计数"""
        self.turn_count += 1

    def record_agent_failure(self) -> None:
        """记录 Agent 执行失败"""
        self.agent_failure_count += 1

    def reset_agent_failures(self) -> None:
        """Agent 成功后重置失败计数"""
        self.agent_failure_count = 0

    def get_unfilled_slots(self) -> list[SlotInfo]:
        """获取所有未填充的必填槽位"""
        return [
            slot for slot in self.slots.values()
            if slot.required and slot.value is None
        ]

    def to_dict(self) -> dict:
        """序列化为字典（用于日志记录）"""
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "intent": self.current_intent.value,
            "domain": self.current_domain.value,
            "emotion": self.emotion.value,
            "task_status": self.task_status.value,
            "active_workflow": self.active_workflow.value,
            "workflow_stage": self.workflow_stage.value,
            "consecutive_angry": self.consecutive_angry_turns,
            "agent_failures": self.agent_failure_count,
            "transfer_requested": self.transfer_requested,
        }
