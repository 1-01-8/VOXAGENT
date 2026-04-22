"""Tests for dialogue/task_state_machine.py — 显式任务状态机。"""

from __future__ import annotations

import pytest

from voice_optimized_rag.agent.permission_guard import TextPermissionGuard
from voice_optimized_rag.core.conversation_stream import ConversationStream
from voice_optimized_rag.dialogue.session import AgentDomain, IntentType, SessionContext, TaskWorkflow
from voice_optimized_rag.dialogue.task_state_machine import BusinessTaskStateMachine


@pytest.fixture
def machine(stream: ConversationStream) -> BusinessTaskStateMachine:
    guard = TextPermissionGuard(stream, prompt_func=lambda prompt: "yes")
    return BusinessTaskStateMachine(guard)


class TestTaskStateMachine:
    @pytest.mark.asyncio
    async def test_refund_workflow_collects_then_executes(self, machine: BusinessTaskStateMachine):
        session = SessionContext(current_intent=IntentType.TASK, current_domain=AgentDomain.FINANCE)

        first = await machine.handle("我要退款", session)
        second = await machine.handle("我的订单号是10392", session)
        third = await machine.handle("因为买错了", session)

        assert first.handled is True
        assert "订单号" in first.reply_text
        assert second.handled is True
        assert "退款原因" in second.reply_text
        assert third.handled is True
        assert "退款申请已提交" in third.reply_text
        assert session.active_workflow == TaskWorkflow.NONE

    @pytest.mark.asyncio
    async def test_cancel_order_workflow_executes_deterministically(self, machine: BusinessTaskStateMachine):
        session = SessionContext(current_intent=IntentType.TASK, current_domain=AgentDomain.AFTER_SALES)

        first = await machine.handle("我要取消订单", session)
        second = await machine.handle("订单号10392", session)

        assert first.handled is True
        assert "订单号" in first.reply_text
        assert second.handled is True
        assert "已成功取消" in second.reply_text

    @pytest.mark.asyncio
    async def test_update_address_workflow_requires_new_address(self, machine: BusinessTaskStateMachine):
        session = SessionContext(current_intent=IntentType.TASK, current_domain=AgentDomain.AFTER_SALES)

        first = await machine.handle("我要修改地址", session)
        second = await machine.handle("订单号10392", session)
        third = await machine.handle("新地址改成上海市浦东新区张江路88号8楼", session)

        assert first.handled is True
        assert "订单号和新的收货地址" in first.reply_text
        assert second.handled is True
        assert "新的收货地址" in second.reply_text
        assert third.handled is True
        assert "收货地址已修改" in third.reply_text