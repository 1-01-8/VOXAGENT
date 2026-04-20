"""
财务类工具（Level 3）—— 需身份验证 + 二次语音确认

包含：
- apply_refund: 发起退款申请流程

执行前需通过 PermissionGuard 的双重校验：
1. 用户语音确认操作
2. 身份验证（V1 简化为二次确认，V2 对接短信/声纹验证）
"""

from __future__ import annotations

from typing import Any

from voice_optimized_rag.agent.base_tool import BaseTool, ToolParameter, ToolResult
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("finance_tools")


class ApplyRefundTool(BaseTool):
    """发起退款申请"""

    @property
    def name(self) -> str:
        return "apply_refund"

    @property
    def description(self) -> str:
        return "为订单发起退款申请。需提供订单号、退款原因和退款金额。"

    @property
    def permission_level(self) -> int:
        return 3

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="order_id", description="订单号"),
            ToolParameter(name="reason", description="退款原因"),
            ToolParameter(
                name="amount",
                type="number",
                description="退款金额（元）",
                required=False,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        order_id = kwargs.get("order_id", "")
        reason = kwargs.get("reason", "")
        amount = kwargs.get("amount")

        if not order_id or not reason:
            return ToolResult(
                success=False,
                error="missing_params",
                message="请提供订单号和退款原因。",
            )

        # TODO: 对接退款系统 API
        logger.info(
            f"Applying refund for order {order_id}: reason={reason}, amount={amount}"
        )
        return ToolResult(
            success=True,
            data={
                "order_id": order_id,
                "refund_id": "RF-20260419-001",
                "reason": reason,
                "amount": amount or "全额",
                "status": "审核中",
            },
            message=(
                f"退款申请已提交（退款单号: RF-20260419-001），"
                f"预计1-3个工作日内审核完成，退款将原路返回。"
            ),
        )
