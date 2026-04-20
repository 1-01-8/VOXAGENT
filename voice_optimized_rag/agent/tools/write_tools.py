"""
写操作类工具（Level 2）—— 需用户语音确认

包含：
- update_address: 修改未发货订单的收货地址
- cancel_order: 取消未发货的订单

执行前通过 ConversationStream 发布 CONFIRM_REQUIRED 事件，
等待用户语音确认后才真正执行。
"""

from __future__ import annotations

from typing import Any

from voice_optimized_rag.agent.base_tool import BaseTool, ToolParameter, ToolResult
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("write_tools")


class UpdateAddressTool(BaseTool):
    """修改未发货订单的收货地址"""

    @property
    def name(self) -> str:
        return "update_address"

    @property
    def description(self) -> str:
        return "修改未发货订单的收货地址。需提供订单号和新地址。"

    @property
    def permission_level(self) -> int:
        return 2

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="order_id", description="订单号"),
            ToolParameter(name="new_address", description="新的收货地址"),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        order_id = kwargs.get("order_id", "")
        new_address = kwargs.get("new_address", "")

        if not order_id or not new_address:
            return ToolResult(
                success=False,
                error="missing_params",
                message="请提供订单号和新的收货地址。",
            )

        # TODO: 对接订单系统 API
        logger.info(f"Updating address for order {order_id}: {new_address}")
        return ToolResult(
            success=True,
            data={"order_id": order_id, "new_address": new_address},
            message=f"订单 {order_id} 的收货地址已修改为: {new_address}",
        )


class CancelOrderTool(BaseTool):
    """取消未发货的订单"""

    @property
    def name(self) -> str:
        return "cancel_order"

    @property
    def description(self) -> str:
        return "取消未发货的订单。需提供订单号和取消原因。"

    @property
    def permission_level(self) -> int:
        return 2

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="order_id", description="订单号"),
            ToolParameter(name="reason", description="取消原因", required=False),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        order_id = kwargs.get("order_id", "")
        reason = kwargs.get("reason", "用户主动取消")

        if not order_id:
            return ToolResult(
                success=False,
                error="missing_params",
                message="请提供要取消的订单号。",
            )

        # TODO: 对接订单系统 API
        logger.info(f"Cancelling order {order_id}: {reason}")
        return ToolResult(
            success=True,
            data={"order_id": order_id, "reason": reason},
            message=f"订单 {order_id} 已成功取消。如涉及退款，将在1-3个工作日内原路返回。",
        )
