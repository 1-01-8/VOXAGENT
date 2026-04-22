"""Tests for agent/skill_registry.py — 正式业务 SkillRegistry。"""

from __future__ import annotations

from voice_optimized_rag.agent.skill_registry import BUSINESS_SKILL_REGISTRY
from voice_optimized_rag.dialogue.session import AgentDomain


class TestSkillRegistry:
    def test_registry_contains_three_business_skills(self):
        assert BUSINESS_SKILL_REGISTRY.domains == {
            AgentDomain.SALES,
            AgentDomain.AFTER_SALES,
            AgentDomain.FINANCE,
        }

    def test_finance_skill_has_refund_tool(self):
        finance_spec = BUSINESS_SKILL_REGISTRY.get(AgentDomain.FINANCE)
        tool_names = [factory().__class__.__name__ for factory in finance_spec.tool_factories]
        assert "ApplyRefundTool" in tool_names