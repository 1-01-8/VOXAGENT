"""Business skill registry for sales, after-sales, and finance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from voice_optimized_rag.agent.base_tool import BaseTool
from voice_optimized_rag.agent.function_calling_agent import (
    FUNCTION_CALLING_SYSTEM_PROMPT,
    FunctionCallingAgent,
)
from voice_optimized_rag.agent.permission_guard import PermissionGuard
from voice_optimized_rag.agent.tools.finance_tools import ApplyRefundTool
from voice_optimized_rag.agent.tools.query_tools import (
    CheckPromotionTool,
    GetCustomerInfoTool,
    QueryInventoryTool,
    QueryOrderTool,
)
from voice_optimized_rag.agent.tools.write_tools import CancelOrderTool, UpdateAddressTool
from voice_optimized_rag.core.conversation_stream import ConversationStream
from voice_optimized_rag.dialogue.session import AgentDomain
from voice_optimized_rag.llm.base import LLMProvider


@dataclass(frozen=True)
class SkillSpec:
    skill_id: str
    domain: AgentDomain
    agent_name: str
    responsibility: str
    scope_rules: tuple[str, ...]
    tool_factories: tuple[Callable[[], BaseTool], ...]


class SkillRegistry:
    """Registry of business skills that can instantiate agents."""

    def __init__(self, specs: list[SkillSpec]) -> None:
        self._specs = {spec.domain: spec for spec in specs}

    @property
    def domains(self) -> set[AgentDomain]:
        return set(self._specs.keys())

    def get(self, domain: AgentDomain) -> SkillSpec:
        return self._specs[domain]

    def create_agents(
        self,
        llm: LLMProvider,
        permission_guard: PermissionGuard,
        stream: ConversationStream,
        max_iterations: int = 10,
        tool_timeout: float = 3.0,
        tool_retry: int = 1,
        max_scratchpad_chars: int = 6000,
    ) -> dict[AgentDomain, FunctionCallingAgent]:
        agents: dict[AgentDomain, FunctionCallingAgent] = {}
        for domain, spec in self._specs.items():
            agents[domain] = FunctionCallingAgent(
                llm=llm,
                tools=[factory() for factory in spec.tool_factories],
                permission_guard=permission_guard,
                stream=stream,
                agent_name=spec.agent_name,
                system_prompt_template=_build_system_prompt(spec),
                max_iterations=max_iterations,
                tool_timeout=tool_timeout,
                tool_retry=tool_retry,
                max_scratchpad_chars=max_scratchpad_chars,
            )
        return agents


def _build_system_prompt(spec: SkillSpec) -> str:
    scope_text = "\n".join(f"- {rule}" for rule in spec.scope_rules)
    return (
        FUNCTION_CALLING_SYSTEM_PROMPT
        + f"\n\n你是{spec.agent_name}，负责{spec.responsibility}。\n\n职责边界：\n{scope_text}"
    )


BUSINESS_SKILL_REGISTRY = SkillRegistry([
    SkillSpec(
        skill_id="sales_consulting",
        domain=AgentDomain.SALES,
        agent_name="销售 Agent",
        responsibility="售前咨询、库存确认、促销推荐和客户权益说明",
        scope_rules=(
            "负责产品信息、方案介绍、价格优惠、库存状态和会员权益说明。",
            "遇到订单取消、地址修改、物流异常等问题时，引导到售后 Skill。",
            "遇到退款、发票、账单、对账等问题时，引导到财务 Skill。",
        ),
        tool_factories=(
            QueryInventoryTool,
            GetCustomerInfoTool,
            CheckPromotionTool,
        ),
    ),
    SkillSpec(
        skill_id="after_sales_ops",
        domain=AgentDomain.AFTER_SALES,
        agent_name="售后 Agent",
        responsibility="订单查询、物流跟进和售后变更处理",
        scope_rules=(
            "负责订单状态、物流、地址修改、取消订单、换货维修等售后问题。",
            "涉及促销、报价、产品推荐时，引导到销售 Skill。",
            "涉及退款金额、发票和账单问题时，引导到财务 Skill。",
        ),
        tool_factories=(
            QueryOrderTool,
            UpdateAddressTool,
            CancelOrderTool,
        ),
    ),
    SkillSpec(
        skill_id="finance_ops",
        domain=AgentDomain.FINANCE,
        agent_name="财务 Agent",
        responsibility="退款、付款和账务相关处理",
        scope_rules=(
            "负责退款申请、发票账单、付款对账等财务相关事项。",
            "如需核对订单基础状态，可调用订单查询工具辅助判断。",
            "涉及产品推荐、售后物流等非财务问题时，引导到对应 Skill。",
        ),
        tool_factories=(
            QueryOrderTool,
            GetCustomerInfoTool,
            ApplyRefundTool,
        ),
    ),
])