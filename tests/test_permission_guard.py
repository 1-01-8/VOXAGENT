"""Tests for agent/permission_guard.py — 权限拦截模块

系统组件: Agent — PermissionGuard 四级权限拦截
源文件:   voice_optimized_rag/agent/permission_guard.py
职责:     L1放行/L2确认/L3验证+确认/L4拒绝，通过事件总线实现确认流

测试覆盖：
- Level 1 工具直接放行
- Level 4 工具直接拒绝
- Level 2 确认流程（模拟确认/拒绝/超时）
- Level 3 身份验证 + 确认

使用 MockTool + asyncio 模拟事件确认流程。
"""

from __future__ import annotations

import asyncio

import pytest

from voice_optimized_rag.agent.permission_guard import PermissionGuard
from voice_optimized_rag.core.conversation_stream import ConversationStream, EventType
from voice_optimized_rag.dialogue.session import SessionContext, UserProfile
from tests.conftest import MockTool


@pytest.fixture
def guard(stream: ConversationStream) -> PermissionGuard:
    return PermissionGuard(stream, confirm_timeout=1.0)  # 1秒超时加速测试


class TestPermissionGuard:
    """权限拦截器测试"""

    @pytest.mark.asyncio
    async def test_level1_pass_through(self, guard: PermissionGuard, session: SessionContext):
        """Level 1 工具应直接放行"""
        tool = MockTool(permission_level=1)
        result = await guard.check_permission(tool, session)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_level4_blocked(self, guard: PermissionGuard, session: SessionContext):
        """Level 4 工具应直接拒绝"""
        tool = MockTool(permission_level=4)
        result = await guard.check_permission(tool, session)
        assert result.success is False
        assert "permission_denied" == result.error

    @pytest.mark.asyncio
    async def test_level2_confirmed(self, guard: PermissionGuard, session: SessionContext):
        """Level 2 工具在用户确认后应放行"""
        tool = MockTool(permission_level=2)

        # 后台任务模拟用户 100ms 后确认
        async def _confirm():
            await asyncio.sleep(0.1)
            await guard.handle_confirm_response(True)

        task = asyncio.create_task(_confirm())
        result = await guard.check_permission(tool, session)
        await task
        assert result.success is True

    @pytest.mark.asyncio
    async def test_level2_rejected(self, guard: PermissionGuard, session: SessionContext):
        """Level 2 工具被用户拒绝应返回失败"""
        tool = MockTool(permission_level=2)

        async def _reject():
            await asyncio.sleep(0.1)
            await guard.handle_confirm_response(False)

        task = asyncio.create_task(_reject())
        result = await guard.check_permission(tool, session)
        await task
        assert result.success is False
        assert "user_rejected" == result.error

    @pytest.mark.asyncio
    async def test_level2_timeout(self, guard: PermissionGuard, session: SessionContext):
        """Level 2 确认超时应视为拒绝"""
        tool = MockTool(permission_level=2)
        # 不发送确认响应，等超时（1秒）
        result = await guard.check_permission(tool, session)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_level3_with_user_id(self, guard: PermissionGuard, session: SessionContext):
        """Level 3 工具有 user_id 时身份验证应通过"""
        session.user_profile = UserProfile(user_id="U001")
        tool = MockTool(permission_level=3)

        # 模拟 Level 2 确认
        async def _confirm():
            await asyncio.sleep(0.1)
            await guard.handle_confirm_response(True)

        task = asyncio.create_task(_confirm())
        result = await guard.check_permission(tool, session)
        await task
        assert result.success is True

    @pytest.mark.asyncio
    async def test_confirm_event_published(self, guard: PermissionGuard, session: SessionContext, stream: ConversationStream):
        """确认请求应发布 CONFIRM_REQUIRED 事件（含 request_id）"""
        tool = MockTool(permission_level=2)

        async def _confirm():
            await asyncio.sleep(0.1)
            await guard.handle_confirm_response(True)

        task = asyncio.create_task(_confirm())
        await guard.check_permission(tool, session)
        await task

        confirm_events = [e for e in stream.history if e.event_type == EventType.CONFIRM_REQUIRED]
        assert len(confirm_events) >= 1
        assert "request_id" in confirm_events[0].metadata

    @pytest.mark.asyncio
    async def test_concurrent_confirmations(self, stream: ConversationStream, session: SessionContext):
        """并发确认请求不应互相覆盖（通过 request_id 隔离）"""
        guard = PermissionGuard(stream, confirm_timeout=2.0)
        tool_a = MockTool(permission_level=2, name="tool_a")
        tool_b = MockTool(permission_level=2, name="tool_b")

        async def _confirm_both():
            await asyncio.sleep(0.1)
            # 兼容模式：两次无 request_id 确认，按 FIFO 分配
            await guard.handle_confirm_response(True)
            await asyncio.sleep(0.05)
            await guard.handle_confirm_response(False)

        task = asyncio.create_task(_confirm_both())

        result_a, result_b = await asyncio.gather(
            guard.check_permission(tool_a, session),
            guard.check_permission(tool_b, session),
        )
        await task

        # 其中一个应成功、一个应失败（不应都成功或都失败）
        results = {result_a.success, result_b.success}
        assert True in results and False in results
