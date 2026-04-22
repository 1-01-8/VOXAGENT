"""Explicit state machine for deterministic business task flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from voice_optimized_rag.agent.base_tool import BaseTool, ToolResult
from voice_optimized_rag.agent.permission_guard import PermissionGuard
from voice_optimized_rag.agent.tools.finance_tools import ApplyRefundTool
from voice_optimized_rag.agent.tools.write_tools import CancelOrderTool, UpdateAddressTool
from voice_optimized_rag.dialogue.session import AgentDomain, SessionContext, TaskStage, TaskStatus, TaskWorkflow
from voice_optimized_rag.dialogue.task_slots import sync_workflow_slots


@dataclass(frozen=True)
class TaskStateMachineResult:
    handled: bool
    reply_text: str = ""


@dataclass(frozen=True)
class WorkflowSpec:
    workflow: TaskWorkflow
    domain: AgentDomain
    tool_factory: Callable[[], BaseTool]
    required_slots: tuple[str, ...]
    trigger_keywords: tuple[str, ...]


WORKFLOW_SPECS: dict[TaskWorkflow, WorkflowSpec] = {
    TaskWorkflow.REFUND: WorkflowSpec(
        workflow=TaskWorkflow.REFUND,
        domain=AgentDomain.FINANCE,
        tool_factory=ApplyRefundTool,
        required_slots=("reason",),
        trigger_keywords=("退款", "退费", "退货退款"),
    ),
    TaskWorkflow.CANCEL_ORDER: WorkflowSpec(
        workflow=TaskWorkflow.CANCEL_ORDER,
        domain=AgentDomain.AFTER_SALES,
        tool_factory=CancelOrderTool,
        required_slots=("order_id",),
        trigger_keywords=("取消订单", "撤销订单", "取消这个订单"),
    ),
    TaskWorkflow.UPDATE_ADDRESS: WorkflowSpec(
        workflow=TaskWorkflow.UPDATE_ADDRESS,
        domain=AgentDomain.AFTER_SALES,
        tool_factory=UpdateAddressTool,
        required_slots=("order_id", "new_address"),
        trigger_keywords=("修改地址", "改地址", "收货地址", "地址改"),
    ),
}


class BusinessTaskStateMachine:
    """Deterministic workflow layer for high-value business operations."""

    def __init__(self, permission_guard: PermissionGuard) -> None:
        self._permission_guard = permission_guard
        self.last_trace = "尚无状态机决策"
        self._tools = {
            workflow: spec.tool_factory()
            for workflow, spec in WORKFLOW_SPECS.items()
        }

    async def handle(self, user_text: str, session: SessionContext) -> TaskStateMachineResult:
        workflow = self._detect_workflow(user_text, session)
        if workflow == TaskWorkflow.NONE:
            self.last_trace = "no deterministic workflow matched -> fallback to domain agent"
            return TaskStateMachineResult(handled=False)

        spec = WORKFLOW_SPECS[workflow]
        if workflow != session.active_workflow:
            self._clear_workflow_slots(session)
        session.active_workflow = workflow
        session.workflow_stage = TaskStage.COLLECTING
        session.task_description = workflow.value
        session.current_domain = spec.domain
        session.task_status = TaskStatus.PENDING
        self.last_trace = f"workflow={workflow.value} stage=collecting"

        sync_workflow_slots(session, workflow, user_text)

        missing_prompt = self._build_missing_prompt(session, workflow)
        if missing_prompt:
            self.last_trace = f"workflow={workflow.value} missing required slots -> collecting"
            return TaskStateMachineResult(handled=True, reply_text=missing_prompt)

        tool = self._tools[workflow]
        arguments = self._build_arguments(session, workflow)
        session.workflow_stage = TaskStage.CONFIRMING

        perm_result = await self._permission_guard.check_permission(
            tool=tool,
            session=session,
            **arguments,
        )
        if not perm_result.success:
            session.workflow_stage = TaskStage.CANCELLED
            session.task_status = TaskStatus.FAILED
            reply_text = self._build_permission_failure_reply(perm_result)
            self.last_trace = f"workflow={workflow.value} permission failed -> {perm_result.error}"
            if perm_result.error in {"user_rejected", "identity_verification_failed"}:
                self._reset_workflow(session)
            return TaskStateMachineResult(handled=True, reply_text=reply_text)

        session.workflow_stage = TaskStage.EXECUTING
        session.task_status = TaskStatus.IN_PROGRESS
        self.last_trace = f"workflow={workflow.value} confirmed -> executing"
        result = await tool.execute(**arguments)
        if result.success:
            session.workflow_stage = TaskStage.COMPLETED
            session.task_status = TaskStatus.COMPLETED
            reply_text = result.message
            self.last_trace = f"workflow={workflow.value} executed successfully -> completed"
            self._reset_workflow(session)
            return TaskStateMachineResult(handled=True, reply_text=reply_text)

        if result.error == "missing_params":
            session.workflow_stage = TaskStage.COLLECTING
            session.task_status = TaskStatus.PENDING
            self.last_trace = f"workflow={workflow.value} tool reported missing params -> collecting"
            return TaskStateMachineResult(handled=True, reply_text=result.message)

        session.workflow_stage = TaskStage.CANCELLED
        session.task_status = TaskStatus.FAILED
        self.last_trace = f"workflow={workflow.value} execution failed -> cancelled"
        return TaskStateMachineResult(
            handled=True,
            reply_text=result.message or "业务处理失败，请稍后重试或转人工处理。",
        )

    def _detect_workflow(self, user_text: str, session: SessionContext) -> TaskWorkflow:
        explicit_workflow = self._match_workflow(user_text)
        if explicit_workflow != TaskWorkflow.NONE:
            return explicit_workflow

        if session.active_workflow != TaskWorkflow.NONE and session.workflow_stage not in {
            TaskStage.COMPLETED,
            TaskStage.CANCELLED,
        }:
            return session.active_workflow

        return TaskWorkflow.NONE

    def _match_workflow(self, user_text: str) -> TaskWorkflow:
        normalized = user_text.lower()
        for workflow, spec in WORKFLOW_SPECS.items():
            if any(keyword in normalized for keyword in spec.trigger_keywords):
                return workflow
        return TaskWorkflow.NONE

    def _build_missing_prompt(self, session: SessionContext, workflow: TaskWorkflow) -> str:
        if workflow == TaskWorkflow.REFUND:
            has_order_ref = bool(session.slots.get("order_id") and session.slots["order_id"].value) or bool(
                session.slots.get("phone") and session.slots["phone"].value
            )
            has_reason = bool(session.slots.get("reason") and session.slots["reason"].value)
            if not has_order_ref and not has_reason:
                return "为了帮助您完成退款申请，请提供以下信息：订单号或购买商品的手机号，以及退款原因。"
            if not has_order_ref:
                return "为了帮助您完成退款申请，请提供订单号或购买商品的手机号。这样我才能为您发起退款请求。"
            if not has_reason:
                return "为了处理您的退款申请，请告知我具体的退款原因，以便我们继续进行。"
            return ""

        if workflow == TaskWorkflow.CANCEL_ORDER:
            if not (session.slots.get("order_id") and session.slots["order_id"].value):
                return "请提供要取消的订单号；如方便，也可以附上取消原因。"
            return ""

        if workflow == TaskWorkflow.UPDATE_ADDRESS:
            has_order = bool(session.slots.get("order_id") and session.slots["order_id"].value)
            has_address = bool(session.slots.get("new_address") and session.slots["new_address"].value)
            if not has_order and not has_address:
                return "请提供订单号和新的收货地址，我会按步骤为您办理修改。"
            if not has_order:
                return "请先提供需要修改地址的订单号。"
            if not has_address:
                return "请提供新的收货地址。"
            return ""

        return ""

    def _build_arguments(self, session: SessionContext, workflow: TaskWorkflow) -> dict[str, object]:
        if workflow == TaskWorkflow.REFUND:
            return {
                "order_id": session.slots.get("order_id").value if session.slots.get("order_id") else "",
                "reason": session.slots.get("reason").value if session.slots.get("reason") else "",
                "amount": session.slots.get("amount").value if session.slots.get("amount") and session.slots.get("amount").value is not None else None,
            }

        if workflow == TaskWorkflow.CANCEL_ORDER:
            return {
                "order_id": session.slots.get("order_id").value if session.slots.get("order_id") else "",
                "reason": session.slots.get("reason").value if session.slots.get("reason") and session.slots.get("reason").value else "用户主动取消",
            }

        return {
            "order_id": session.slots.get("order_id").value if session.slots.get("order_id") else "",
            "new_address": session.slots.get("new_address").value if session.slots.get("new_address") else "",
        }

    @staticmethod
    def _build_permission_failure_reply(result: ToolResult) -> str:
        if result.error == "user_rejected":
            return "您已取消本次操作。如需继续办理，请重新发起业务请求。"
        if result.error == "identity_verification_failed":
            return "身份验证未通过，当前操作已取消。如需继续，请重新验证后再试。"
        return result.message or "操作未完成，请稍后重试。"

    @staticmethod
    def _reset_workflow(session: SessionContext) -> None:
        BusinessTaskStateMachine._clear_workflow_slots(session)
        session.active_workflow = TaskWorkflow.NONE
        session.workflow_stage = TaskStage.IDLE
        session.task_status = TaskStatus.IDLE
        session.task_description = ""

    @staticmethod
    def _clear_workflow_slots(session: SessionContext) -> None:
        for key in ("order_id", "phone", "reason", "amount", "new_address"):
            session.slots.pop(key, None)