"""
权限分级拦截器 —— 4 级权限校验 + 事件确认触发

在 Agent 执行工具前进行权限检查：
- Level 1：直接放行
- Level 2：通过 ConversationStream 发布 CONFIRM_REQUIRED 事件，
          等待 CONFIRM_RESPONSE 事件返回用户确认结果
- Level 3：需身份验证 + 二次确认（双重校验）
- Level 4：直接拒绝，Agent 不可执行

关键设计：确认流程通过事件总线实现，避免 Agent → Voice 循环依赖。
"""

from __future__ import annotations

import asyncio
from typing import Optional

from voice_optimized_rag.agent.base_tool import BaseTool, ToolResult
from voice_optimized_rag.core.conversation_stream import (
    ConversationStream,
    EventType,
    StreamEvent,
)
from voice_optimized_rag.dialogue.session import SessionContext
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("permission_guard")


class PermissionGuard:
    """
    权限分级拦截器

    在 Agent 调用工具前对权限等级进行校验，
    必要时通过事件总线发起用户确认流程。

    使用 dict[str, Future] 按请求 ID 隔离并发确认，
    避免多个 L2+ 工具同时等待时的竞态条件。
    """

    def __init__(
        self,
        stream: ConversationStream,
        confirm_timeout: float = 30.0,
    ) -> None:
        self._stream = stream
        self._confirm_timeout = confirm_timeout
        # 按 request_id 存储 Future，支持并发确认
        self._pending_confirms: dict[str, asyncio.Future] = {}

    async def check_permission(
        self,
        tool: BaseTool,
        session: SessionContext,
        **tool_kwargs,
    ) -> ToolResult:
        """
        检查工具的权限等级并执行相应校验

        Returns:
            ToolResult - success=True 表示放行，success=False 表示拒绝
        """
        level = tool.permission_level

        if level >= 4:
            return ToolResult(
                success=False,
                error="permission_denied",
                message=f"操作「{tool.name}」为管理级权限(Level 4)，"
                        f"Agent 无法执行。请通过管理后台操作或联系管理员。",
            )

        if level >= 2:
            # 需要用户语音确认
            confirmed = await self._request_confirmation(
                tool=tool,
                session=session,
                **tool_kwargs,
            )
            if not confirmed:
                return ToolResult(
                    success=False,
                    error="user_rejected",
                    message=f"用户未确认操作「{tool.name}」，已取消执行。",
                )

        if level >= 3:
            # Level 3 额外需要身份验证（简化实现：再次确认）
            verified = await self._verify_identity(session)
            if not verified:
                return ToolResult(
                    success=False,
                    error="identity_verification_failed",
                    message="身份验证未通过，财务操作已取消。",
                )

        # 权限校验通过
        return ToolResult(success=True, message="permission_granted")

    async def _request_confirmation(
        self,
        tool: BaseTool,
        session: SessionContext,
        **tool_kwargs,
    ) -> bool:
        """
        通过事件总线发起用户语音确认

        1. 生成唯一 request_id
        2. 发布 CONFIRM_REQUIRED 事件（携带 request_id）
        3. 等待对应的 CONFIRM_RESPONSE（超时则视为拒绝）

        使用 per-request Future 避免并发确认竞态。
        """
        import uuid
        request_id = str(uuid.uuid4())[:8]

        description = f"即将执行操作「{tool.name}」: {tool.description}"
        if tool_kwargs:
            params_str = ", ".join(f"{k}={v}" for k, v in tool_kwargs.items())
            description += f"（参数: {params_str}）"

        # 创建 per-request Future
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        self._pending_confirms[request_id] = future

        # 发布确认请求事件（携带 request_id 供前端回传）
        await self._stream.publish(StreamEvent(
            event_type=EventType.CONFIRM_REQUIRED,
            text=description,
            metadata={
                "request_id": request_id,
                "tool_name": tool.name,
                "permission_level": tool.permission_level,
                "session_id": session.session_id,
            },
        ))

        logger.info(f"Confirmation requested: {tool.name} (req={request_id})")

        try:
            result = await asyncio.wait_for(future, timeout=self._confirm_timeout)
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Confirmation timeout: {tool.name} (req={request_id})")
            return False
        finally:
            self._pending_confirms.pop(request_id, None)

    async def handle_confirm_response(
        self, confirmed: bool, request_id: str = "",
    ) -> None:
        """
        处理用户确认响应

        Args:
            confirmed: 用户是否确认
            request_id: 确认请求 ID（精确匹配）；
                       为空时兜底设置最早的 pending Future（向下兼容）
        """
        if request_id and request_id in self._pending_confirms:
            fut = self._pending_confirms[request_id]
            if not fut.done():
                fut.set_result(confirmed)
            return

        # 兼容旧协议：无 request_id 时，设置最早的 pending Future
        for rid, fut in list(self._pending_confirms.items()):
            if not fut.done():
                fut.set_result(confirmed)
                return

    async def _verify_identity(self, session: SessionContext) -> bool:
        """
        身份验证（简化实现）

        V1: 通过会话中已有的用户信息进行基本验证
        V2: 对接实际的身份验证服务（短信验证码、语音声纹等）
        """
        if session.user_profile.user_id:
            return True
        # 无用户信息时请求二次确认作为替代
        return await self._request_confirmation(
            tool=_IdentityVerificationPseudoTool(),
            session=session,
        )


class _IdentityVerificationPseudoTool(BaseTool):
    """伪工具，仅用于身份验证的确认对话"""

    @property
    def name(self) -> str:
        return "identity_verification"

    @property
    def description(self) -> str:
        return "请确认您的身份信息以继续操作"

    @property
    def permission_level(self) -> int:
        return 2

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True)
