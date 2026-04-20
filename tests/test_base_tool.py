"""Tests for agent/base_tool.py — 工具抽象基类

系统组件: Agent — BaseTool 工具抽象层
源文件:   voice_optimized_rag/agent/base_tool.py
职责:     定义工具接口契约（name/description/permission/execute）

测试覆盖：
- ToolResult 数据结构
- BaseTool.to_prompt_description() 格式
- ToolParameter 必填/可选标注
"""

from __future__ import annotations

import pytest

from voice_optimized_rag.agent.base_tool import BaseTool, ToolParameter, ToolResult


class TestToolResult:
    """ToolResult 数据结构测试"""

    def test_success_result(self):
        """成功结果应正确初始化"""
        r = ToolResult(success=True, data={"id": 1}, message="ok")
        assert r.success is True
        assert r.data == {"id": 1}
        assert r.error == ""

    def test_failure_result(self):
        """失败结果应包含错误信息"""
        r = ToolResult(success=False, error="timeout", message="超时")
        assert r.success is False
        assert r.error == "timeout"


class TestBaseTool:
    """BaseTool 抽象接口测试（使用 conftest 中的 MockTool）"""

    def test_prompt_description_format(self):
        """to_prompt_description 应包含名称、描述、权限级别"""
        from tests.conftest import MockTool
        tool = MockTool(name="test_tool", description="A test", permission_level=2)
        desc = tool.to_prompt_description()
        assert "test_tool" in desc
        assert "A test" in desc
        assert "Level 2" in desc
        assert "写操作" in desc  # Level 2 对应 "写操作(需确认)"

    def test_prompt_description_with_params(self):
        """有参数的工具应在描述中显示参数列表"""
        from tests.conftest import MockTool
        tool = MockTool()
        # 覆盖 parameters 属性
        tool.__class__ = type("ParamTool", (MockTool,), {
            "parameters": property(lambda self: [
                ToolParameter(name="order_id", description="订单号", required=True),
                ToolParameter(name="remark", description="备注", required=False),
            ])
        })
        desc = tool.to_prompt_description()
        assert "order_id" in desc
        assert "(必填)" in desc
        assert "(可选)" in desc
