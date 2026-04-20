"""
查询类工具（Level 1）—— 只读操作，无需用户确认

包含：
- query_order: 查询订单状态/物流信息
- query_inventory: 查询产品库存/到货时间
- get_customer_info: 获取客户信息与权益
- check_promotion: 查询当前可用优惠活动

所有工具均为 Level 1 权限，直接执行无需确认。
实际生产中需对接真实业务系统 API。
"""

from __future__ import annotations

from typing import Any

from voice_optimized_rag.agent.base_tool import BaseTool, ToolParameter, ToolResult
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("query_tools")


class QueryOrderTool(BaseTool):
    """查询订单状态和物流信息"""

    @property
    def name(self) -> str:
        return "query_order"

    @property
    def description(self) -> str:
        return "查询订单状态、物流信息。需提供订单号或手机号。"

    @property
    def permission_level(self) -> int:
        return 1

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="order_id", description="订单号", required=False),
            ToolParameter(name="phone", description="手机号", required=False),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        order_id = kwargs.get("order_id", "")
        phone = kwargs.get("phone", "")

        if not order_id and not phone:
            return ToolResult(
                success=False,
                error="missing_params",
                message="请提供订单号或手机号来查询订单。",
            )

        # TODO: 对接真实订单系统 API
        logger.info(f"Querying order: id={order_id}, phone={phone}")
        return ToolResult(
            success=True,
            data={
                "order_id": order_id or "ORD-2026041900001",
                "status": "已发货",
                "logistics": "顺丰速运 SF1234567890",
                "estimated_arrival": "2026-04-21",
            },
            message="订单已发货，顺丰速运 SF1234567890，预计4月21日到达。",
        )


class QueryInventoryTool(BaseTool):
    """查询产品库存和到货时间"""

    @property
    def name(self) -> str:
        return "query_inventory"

    @property
    def description(self) -> str:
        return "查询产品库存数量和预计到货时间。需提供产品名称或编号。"

    @property
    def permission_level(self) -> int:
        return 1

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="product_name", description="产品名称或编号"),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        product = kwargs.get("product_name", "")
        if not product:
            return ToolResult(
                success=False,
                error="missing_params",
                message="请告诉我您想查询哪个产品的库存？",
            )

        # TODO: 对接真实库存系统
        logger.info(f"Querying inventory: {product}")
        return ToolResult(
            success=True,
            data={"product": product, "stock": 128, "available": True},
            message=f"「{product}」当前有现货，库存充足。",
        )


class GetCustomerInfoTool(BaseTool):
    """获取客户信息与权益"""

    @property
    def name(self) -> str:
        return "get_customer_info"

    @property
    def description(self) -> str:
        return "获取客户会员等级、优惠券余额等信息。需提供手机号或客户ID。"

    @property
    def permission_level(self) -> int:
        return 1

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="customer_id", description="客户ID或手机号"),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        customer_id = kwargs.get("customer_id", "")
        if not customer_id:
            return ToolResult(
                success=False,
                error="missing_params",
                message="请提供您的手机号或客户编号。",
            )

        # TODO: 对接 CRM 系统
        logger.info(f"Fetching customer info: {customer_id}")
        return ToolResult(
            success=True,
            data={
                "customer_id": customer_id,
                "vip_level": 2,
                "coupons": 3,
                "points": 5680,
            },
            message=f"您是 VIP 2 级会员，当前有 3 张优惠券、5680 积分可用。",
        )


class CheckPromotionTool(BaseTool):
    """查询当前可用优惠活动"""

    @property
    def name(self) -> str:
        return "check_promotion"

    @property
    def description(self) -> str:
        return "查询当前可用的优惠活动和促销信息。"

    @property
    def permission_level(self) -> int:
        return 1

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="category",
                description="产品类别（可选）",
                required=False,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        category = kwargs.get("category", "全部")

        # TODO: 对接促销系统
        logger.info(f"Checking promotions: category={category}")
        return ToolResult(
            success=True,
            data={
                "promotions": [
                    {"name": "春季特惠", "discount": "8折", "end_date": "2026-04-30"},
                    {"name": "满减活动", "rule": "满500减50", "end_date": "2026-05-01"},
                ],
            },
            message="当前有两个优惠活动：春季特惠8折（截止4月30日）、满500减50（截止5月1日）。",
        )
