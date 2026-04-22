"""Tests for voice_optimized_rag/mcp_server.py — MCP 能力导出。"""

from __future__ import annotations

from voice_optimized_rag.mcp_server import serialize_skill_catalog


class TestMCPServer:
    def test_serialize_skill_catalog_contains_three_skills(self):
        catalog = serialize_skill_catalog()
        assert len(catalog) == 3
        assert {item["domain"] for item in catalog} == {"sales", "after_sales", "finance"}

    def test_finance_skill_exposes_refund_tool(self):
        catalog = serialize_skill_catalog()
        finance = next(item for item in catalog if item["domain"] == "finance")
        assert "apply_refund" in finance["tools"]