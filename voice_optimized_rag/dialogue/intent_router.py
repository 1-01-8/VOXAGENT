"""意图识别与路由 —— 纯业务双路 + 超范围拒答。"""

from __future__ import annotations

from voice_optimized_rag.dialogue.follow_up import looks_like_sales_knowledge_follow_up, looks_like_task_follow_up
from voice_optimized_rag.dialogue.session import AgentDomain, IntentType, SessionContext, TaskWorkflow
from voice_optimized_rag.llm.base import LLMProvider
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("intent_router")

INTENT_CLASSIFICATION_PROMPT = """你是一个意图分类器。根据用户发言和对话上下文，判断用户意图属于以下哪一类：

1. knowledge - 知识咨询型：用户在询问产品信息、价格政策、使用方法、售后规则等知识类问题
2. task - 任务执行型：用户想要执行操作，如下单、查订单、退款、修改地址、取消订单等
3. out_of_scope - 超范围：用户在闲聊、寒暄、问你是谁、让你讲笑话，或内容不属于销售、售后、财务业务

对话上下文：
{context}

用户发言：{utterance}

请只返回一个词：knowledge、task 或 out_of_scope"""

# 任务型意图的关键词快速匹配（避免不必要的 LLM 调用）
TASK_KEYWORDS = [
    "下单", "下订单", "买", "购买", "退款", "退货", "取消订单",
    "修改地址", "改地址", "查订单", "查物流", "快递", "发货",
    "申请退款", "换货", "投诉", "工单",
]

KNOWLEDGE_KEYWORDS = [
    "多少钱", "价格", "报价", "费用", "收费", "套餐", "方案",
    "功能", "参数", "怎么用", "如何",
    "什么是", "介绍", "区别", "对比", "优惠", "活动", "促销",
    "规格", "保修", "质保", "售后", "商品", "目录", "产品线", "模块", "型号", "编号",
]

TRANSFER_KEYWORDS = [
    "转人工", "人工客服", "我要投诉", "找你们经理", "找领导",
]

OUT_OF_SCOPE_KEYWORDS = [
    "你好", "hello", "hi", "你是谁", "讲个笑话", "天气", "在吗",
]


class IntentRouter:
    """
    意图识别与路由器

    三级判断策略：
    1. 关键词快速匹配（零延迟）
    2. 转人工关键词检测（优先级最高）
    3. LLM 业务意图分类（兜底，最准确）
    """

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self.last_trace = "尚无意图决策"

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

        # 第一优先：转人工关键词 → 标记并直接视为业务任务
        for kw in TRANSFER_KEYWORDS:
            if kw in text_lower:
                session.transfer_requested = True
                session.transfer_reason = f"用户主动要求: '{kw}'"
                logger.info(f"Transfer keyword detected: {kw}")
                self.last_trace = f"transfer keyword match: {kw} -> task"
                return IntentType.TASK

        # 第二优先：多轮任务补槽续接
        if (
            session.current_intent == IntentType.TASK
            and session.active_workflow != TaskWorkflow.NONE
            and looks_like_task_follow_up(utterance)
        ):
            logger.debug("Task follow-up detected from session context")
            self.last_trace = (
                f"task follow-up: active_workflow={session.active_workflow.value} -> task"
            )
            return IntentType.TASK

        if (
            session.turn_count > 1
            and
            session.current_intent == IntentType.KNOWLEDGE
            and session.current_domain == AgentDomain.SALES
            and looks_like_sales_knowledge_follow_up(utterance)
        ):
            logger.debug("Sales knowledge follow-up detected from session context")
            self.last_trace = "sales knowledge follow-up: short product/catalog cue -> knowledge"
            return IntentType.KNOWLEDGE

        # 第三优先：任务型关键词快速匹配
        for kw in TASK_KEYWORDS:
            if kw in text_lower:
                logger.debug(f"Task keyword match: {kw}")
                self.last_trace = f"task keyword match: {kw} -> task"
                return IntentType.TASK

        # 第四优先：知识型关键词快速匹配
        for kw in KNOWLEDGE_KEYWORDS:
            if kw in text_lower:
                logger.debug(f"Knowledge keyword match: {kw}")
                self.last_trace = f"knowledge keyword match: {kw} -> knowledge"
                return IntentType.KNOWLEDGE

        for kw in OUT_OF_SCOPE_KEYWORDS:
            if kw in text_lower:
                logger.debug(f"Out-of-scope keyword match: {kw}")
                self.last_trace = f"out_of_scope keyword match: {kw} -> out_of_scope"
                return IntentType.OUT_OF_SCOPE

        # 兜底：LLM 分类
        try:
            prompt = INTENT_CLASSIFICATION_PROMPT.format(
                context=conversation_text or "（无上下文）",
                utterance=utterance,
            )
            result = await self._llm.generate(prompt)
            result = result.strip().lower()

            if "task" in result:
                self.last_trace = f"llm fallback -> task ({result})"
                return IntentType.TASK
            elif "knowledge" in result:
                self.last_trace = f"llm fallback -> knowledge ({result})"
                return IntentType.KNOWLEDGE
            else:
                self.last_trace = f"llm fallback -> out_of_scope ({result})"
                return IntentType.OUT_OF_SCOPE
        except Exception as e:
            logger.warning(f"LLM intent classification failed: {e}")
            # LLM 失败时默认走知识检索（最安全的业务降级）
            self.last_trace = f"llm error fallback -> knowledge ({type(e).__name__})"
            return IntentType.KNOWLEDGE
