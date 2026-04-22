"""
业务域路由 —— 销售 / 售后 / 财务三域分类

在三路意图路由之后，再将问题细分到具体业务域：
- sales: 售前咨询、产品推荐、库存、促销、方案
- after_sales: 订单、物流、地址修改、取消、保修、维修
- finance: 退款、发票、付款、账单、对账
"""

from __future__ import annotations

from voice_optimized_rag.dialogue.follow_up import looks_like_sales_knowledge_follow_up, looks_like_task_follow_up
from voice_optimized_rag.dialogue.session import AgentDomain, IntentType, SessionContext
from voice_optimized_rag.llm.base import LLMProvider
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("domain_router")

DOMAIN_CLASSIFICATION_PROMPT = """你是一个业务域分类器。根据用户发言、当前意图和对话上下文，将请求分到以下三个业务域之一：

1. sales - 销售域：售前咨询、产品介绍、库存、促销、报价、套餐、选型、购买建议
2. after_sales - 售后域：订单状态、物流、地址修改、取消订单、换货、维修、保修、安装、故障处理
3. finance - 财务域：退款、发票、付款、账单、对账、回款、费用相关

当前意图：{intent}
对话上下文：
{context}

用户发言：{utterance}

请只返回一个词：sales、after_sales 或 finance"""

SALES_KEYWORDS = [
    "价格", "报价", "套餐", "方案", "优惠", "促销", "活动", "库存", "现货",
    "产品", "商品", "目录", "产品线", "模块", "型号", "编号", "规格", "参数", "对比", "购买", "下单", "选型", "试用",
]

AFTER_SALES_KEYWORDS = [
    "订单", "物流", "快递", "发货", "收货", "地址", "修改地址", "取消订单",
    "售后", "换货", "退货", "维修", "保修", "质保", "安装", "故障", "工单",
]

FINANCE_KEYWORDS = [
    "退款", "退费", "发票", "开票", "账单", "付款", "支付", "对账",
    "回款", "催款", "费用", "扣款", "报销",
]


class DomainRouter:
    """业务域路由器。"""

    def __init__(self, llm: LLMProvider) -> None:
        self._llm = llm
        self.last_trace = "尚无域决策"

    async def classify(
        self,
        utterance: str,
        session: SessionContext,
        intent: IntentType | None = None,
        conversation_text: str = "",
    ) -> AgentDomain:
        text_lower = utterance.lower()

        if (
            intent == IntentType.TASK
            and session.turn_count > 1
            and looks_like_task_follow_up(utterance)
        ):
            logger.debug(f"Task follow-up detected, reusing domain: {session.current_domain.value}")
            self.last_trace = f"task follow-up reuse domain: {session.current_domain.value}"
            return session.current_domain

        if (
            intent == IntentType.KNOWLEDGE
            and session.turn_count > 1
            and session.current_domain == AgentDomain.SALES
            and looks_like_sales_knowledge_follow_up(utterance)
        ):
            logger.debug("Sales knowledge follow-up detected, reusing sales domain")
            self.last_trace = "knowledge follow-up reuse domain: sales"
            return session.current_domain

        for keyword in FINANCE_KEYWORDS:
            if keyword in text_lower:
                session.current_domain = AgentDomain.FINANCE
                logger.debug(f"Finance keyword match: {keyword}")
                self.last_trace = f"finance keyword match: {keyword}"
                return AgentDomain.FINANCE

        for keyword in AFTER_SALES_KEYWORDS:
            if keyword in text_lower:
                session.current_domain = AgentDomain.AFTER_SALES
                logger.debug(f"After-sales keyword match: {keyword}")
                self.last_trace = f"after_sales keyword match: {keyword}"
                return AgentDomain.AFTER_SALES

        for keyword in SALES_KEYWORDS:
            if keyword in text_lower:
                session.current_domain = AgentDomain.SALES
                logger.debug(f"Sales keyword match: {keyword}")
                self.last_trace = f"sales keyword match: {keyword}"
                return AgentDomain.SALES

        try:
            prompt = DOMAIN_CLASSIFICATION_PROMPT.format(
                intent=(intent.value if intent else "unknown"),
                context=conversation_text or "（无上下文）",
                utterance=utterance,
            )
            result = (await self._llm.generate(prompt)).strip().lower()

            if "finance" in result:
                domain = AgentDomain.FINANCE
            elif "after_sales" in result or "after-sales" in result:
                domain = AgentDomain.AFTER_SALES
            else:
                domain = AgentDomain.SALES

            session.current_domain = domain
            self.last_trace = f"llm fallback -> {domain.value} ({result})"
            return domain
        except Exception as error:
            logger.warning(f"Domain classification failed: {error}")
            domain = self._fallback_domain(intent)
            session.current_domain = domain
            self.last_trace = f"llm error fallback -> {domain.value} ({type(error).__name__})"
            return domain

    @staticmethod
    def _fallback_domain(intent: IntentType | None) -> AgentDomain:
        if intent == IntentType.TASK:
            return AgentDomain.AFTER_SALES
        return AgentDomain.SALES