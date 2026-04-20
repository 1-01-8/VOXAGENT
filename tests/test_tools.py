"""Tests for agent/tools/ — 业务工具集

系统组件: Agent — Business Tools 业务工具集(7个)
源文件:   voice_optimized_rag/agent/tools/{query,write,finance}_tools.py
职责:     L1查询(4个)/L2写操作(2个)/L3财务(1个) 工具的接口与参数校验

测试覆盖：
- 查询类工具(L1)：缺少参数时返回失败、正常返回 Mock 数据
- 写操作工具(L2)：参数校验、成功执行
- 财务工具(L3)：参数校验、退款流程

所有工具均为 TODO/Mock 实现，测试验证接口契约和参数校验逻辑。
"""

from __future__ import annotations

import pytest

from voice_optimized_rag.agent.tools.query_tools import (
    CheckPromotionTool,
    GetCustomerInfoTool,
    QueryInventoryTool,
    QueryOrderTool,
)
from voice_optimized_rag.agent.tools.write_tools import CancelOrderTool, UpdateAddressTool
from voice_optimized_rag.agent.tools.finance_tools import ApplyRefundTool


# ─────────────── Level 1: 查询类工具 ───────────────

class TestQueryOrderTool:
    @pytest.mark.asyncio
    async def test_missing_params(self):
        """缺少订单号和手机号应返回失败"""
        tool = QueryOrderTool()
        result = await tool.execute()
        assert result.success is False
        assert "missing_params" == result.error

    @pytest.mark.asyncio
    async def test_with_order_id(self):
        """提供订单号应返回成功"""
        tool = QueryOrderTool()
        result = await tool.execute(order_id="ORD-001")
        assert result.success is True
        assert result.data["order_id"] == "ORD-001"

    def test_permission_level(self):
        assert QueryOrderTool().permission_level == 1


class TestQueryInventoryTool:
    @pytest.mark.asyncio
    async def test_missing_product(self):
        tool = QueryInventoryTool()
        result = await tool.execute()
        assert result.success is False

    @pytest.mark.asyncio
    async def test_with_product(self):
        tool = QueryInventoryTool()
        result = await tool.execute(product_name="手机壳")
        assert result.success is True
        assert "手机壳" in result.message


class TestGetCustomerInfoTool:
    @pytest.mark.asyncio
    async def test_missing_id(self):
        tool = GetCustomerInfoTool()
        result = await tool.execute()
        assert result.success is False

    @pytest.mark.asyncio
    async def test_with_id(self):
        tool = GetCustomerInfoTool()
        result = await tool.execute(customer_id="C001")
        assert result.success is True
        assert "vip_level" in result.data


class TestCheckPromotionTool:
    @pytest.mark.asyncio
    async def test_default_category(self):
        """不传参也应正常返回"""
        tool = CheckPromotionTool()
        result = await tool.execute()
        assert result.success is True
        assert "promotions" in result.data


# ─────────────── Level 2: 写操作工具 ───────────────

class TestUpdateAddressTool:
    @pytest.mark.asyncio
    async def test_missing_params(self):
        tool = UpdateAddressTool()
        result = await tool.execute(order_id="ORD-001")  # 缺少 new_address
        assert result.success is False

    @pytest.mark.asyncio
    async def test_success(self):
        tool = UpdateAddressTool()
        result = await tool.execute(order_id="ORD-001", new_address="北京市朝阳区")
        assert result.success is True
        assert "北京市朝阳区" in result.message

    def test_permission_level(self):
        assert UpdateAddressTool().permission_level == 2


class TestCancelOrderTool:
    @pytest.mark.asyncio
    async def test_missing_order_id(self):
        tool = CancelOrderTool()
        result = await tool.execute()
        assert result.success is False

    @pytest.mark.asyncio
    async def test_success(self):
        tool = CancelOrderTool()
        result = await tool.execute(order_id="ORD-002")
        assert result.success is True
        assert "ORD-002" in result.message


# ─────────────── Level 3: 财务工具 ───────────────

class TestApplyRefundTool:
    @pytest.mark.asyncio
    async def test_missing_params(self):
        tool = ApplyRefundTool()
        result = await tool.execute(order_id="ORD-003")  # 缺少 reason
        assert result.success is False

    @pytest.mark.asyncio
    async def test_success(self):
        tool = ApplyRefundTool()
        result = await tool.execute(order_id="ORD-003", reason="质量问题", amount=99.9)
        assert result.success is True
        assert "RF-" in result.data["refund_id"]

    def test_permission_level(self):
        assert ApplyRefundTool().permission_level == 3
