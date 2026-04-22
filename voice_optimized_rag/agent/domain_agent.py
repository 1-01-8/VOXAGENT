"""Domain agent compatibility wrappers backed by the business skill registry."""

from __future__ import annotations

from voice_optimized_rag.agent.permission_guard import PermissionGuard
from voice_optimized_rag.agent.skill_registry import BUSINESS_SKILL_REGISTRY, SkillSpec, SkillRegistry
from voice_optimized_rag.core.conversation_stream import ConversationStream
from voice_optimized_rag.dialogue.session import AgentDomain
from voice_optimized_rag.llm.base import LLMProvider


DOMAIN_AGENT_SPECS: dict[AgentDomain, SkillSpec] = {
    domain: BUSINESS_SKILL_REGISTRY.get(domain)
    for domain in BUSINESS_SKILL_REGISTRY.domains
}


def create_domain_agents(
    llm: LLMProvider,
    permission_guard: PermissionGuard,
    stream: ConversationStream,
    max_iterations: int = 10,
    tool_timeout: float = 3.0,
    tool_retry: int = 1,
    max_scratchpad_chars: int = 6000,
) -> dict[AgentDomain, ReactAgent]:
    """构建销售、售后、财务三个领域 Agent。"""
    return BUSINESS_SKILL_REGISTRY.create_agents(
        llm=llm,
        permission_guard=permission_guard,
        stream=stream,
        max_iterations=max_iterations,
        tool_timeout=tool_timeout,
        tool_retry=tool_retry,
        max_scratchpad_chars=max_scratchpad_chars,
    )