"""
意图识别与路由 —— 三路路由决策

根据用户发言内容判断意图类别：
- 知识咨询型 → RAG 知识检索流程
- 任务执行型 → Agent 工具调用流程
- 闲聊情绪型 → LLM 直答 + 情绪安抚

使用 LLM 进行意图分类，支持配置独立的轻量模型。
"""

from __future__ import annotations

from voice_optimized_rag.dialogue.session import IntentType, SessionContext
from voice_optimized_rag.llm.base import LLMProvider
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("intent_router")

INTENT_CLASSIFICATION_PROMPT = """你是一个意图分类器。根据用户发言和对话上下文，判断用户意图属于以下哪一类：

1. knowledge - 知识咨询型：用户在询问产品信息、价格政策、使用方法、售后规则等知识类问题
2. task - 任务执行型：用户想要执行操作，如下单、查订单、退款、修改地址、取消订单等
3. chitchat - 闲聊情绪型：用户在闲聊、表达情绪、投诉抱怨、或内容不属于以上两类

对话上下文：
{context}

用户发言：{utterance}

请只返回一个词：knowledge、task 或 chitchat"""

# 任务型意图的关键词快速匹配（避免不必要的 LLM 调用）
TASK_KEYWORDS = [
    "下单", "下订单", "买", "购买", "退款", "退货", "取消订单",
    "修改地址", "改地址", "查订单", "查物流", "快递", "发货",
    "申请退款", "换货", "投诉", "工单",
]

KNOWLEDGE_KEYWORDS = [
    "多少钱", "价格", "功能", "参数", "怎么用", "如何",
    "什么是", "区别", "对比", "优惠", "活动", "促销",
    "规格", "保修", "质保", "售后",
]

TRANSFER_KEYWORDS = [
    "转人工", "人工客服", "我要投诉", "找你们经理", "找领导",
]


class IntentRouter:
    """
    意图识别与路由器

    三级判断策略：
    1. 关键词快速匹配（零延迟）
    2. 转人工关键词检测（优先级最高）
    3. LLM 意图分类（兜底，最准确）
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm

    async def classify(
        self,
        utterance: str,
        session: SessionContext,
        conversation_text: str = "",
    ) -> IntentType:
        """
        判断用户意图类型

        Args:
            utterance: 用户当前发言
            session: 会话上下文
            conversation_text: 格式化的对话历史文本

        Returns:
            IntentType 枚举值
        """
        text_lower = utterance.lower()

        # 第一优先：转人工关键词 → 标记但仍归为 chitchat
        for kw in TRANSFER_KEYWORDS:
            if kw in text_lower:
                session.transfer_requested = True
                session.transfer_reason = f"用户主动要求: '{kw}'"
                logger.info(f"Transfer keyword detected: {kw}")
                return IntentType.CHITCHAT

        # 第二优先：任务型关键词快速匹配
        for kw in TASK_KEYWORDS:
            if kw in text_lower:
                logger.debug(f"Task keyword match: {kw}")
                return IntentType.TASK

        # 第三优先：知识型关键词快速匹配
        for kw in KNOWLEDGE_KEYWORDS:
            if kw in text_lower:
                logger.debug(f"Knowledge keyword match: {kw}")
                return IntentType.KNOWLEDGE

        # 兜底：LLM 分类
        try:
            prompt = INTENT_CLASSIFICATION_PROMPT.format(
                context=conversation_text or "（无上下文）",
                utterance=utterance,
            )
            result = await self._llm.generate(prompt)
            result = result.strip().lower()

            if "task" in result:
                return IntentType.TASK
            elif "knowledge" in result:
                return IntentType.KNOWLEDGE
            else:
                return IntentType.CHITCHAT
        except Exception as e:
            logger.warning(f"LLM intent classification failed: {e}")
            # LLM 失败时默认走知识检索（最安全的降级）
            return IntentType.KNOWLEDGE
