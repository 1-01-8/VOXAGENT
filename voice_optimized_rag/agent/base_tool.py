"""
工具抽象基类 —— 所有 Agent 可调用工具的统一接口

每个工具需声明：
- name：工具名称（供 LLM 选择）
- description：功能描述（嵌入 Prompt）
- permission_level：权限等级（1-4）
- parameters_schema：参数结构描述

权限等级说明：
  Level 1 - 只读查询：无需确认
  Level 2 - 写操作：需用户语音确认
  Level 3 - 财务操作：需身份验证 + 二次确认
  Level 4 - 管理操作：Agent 不可执行
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolParameter:
    """工具参数定义"""
    name: str
    type: str = "string"
    description: str = ""
    required: bool = True


@dataclass
class ToolResult:
    """工具执行结果"""
    success: bool
    data: Any = None
    error: str = ""
    message: str = ""


class BaseTool(ABC):
    """
    Agent 工具抽象基类

    所有业务工具继承此类，实现 execute() 方法。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称（唯一标识）"""

    @property
    @abstractmethod
    def description(self) -> str:
        """功能描述（嵌入 LLM Prompt）"""

    @property
    @abstractmethod
    def permission_level(self) -> int:
        """权限等级（1-4）"""

    @property
    def parameters(self) -> list[ToolParameter]:
        """参数列表（默认无参数）"""
        return []

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """
        执行工具逻辑

        Args:
            **kwargs: 工具参数（由 Agent 解析后传入）

        Returns:
            ToolResult 对象，包含成功/失败状态和数据
        """

    def to_prompt_description(self) -> str:
        """生成供 LLM 使用的工具描述文本"""
        params = ""
        if self.parameters:
            param_lines = []
            for p in self.parameters:
                req = "(必填)" if p.required else "(可选)"
                param_lines.append(f"    - {p.name} ({p.type}): {p.description} {req}")
            params = "\n  参数:\n" + "\n".join(param_lines)

        level_desc = {1: "只读", 2: "写操作(需确认)", 3: "财务(需验证+确认)", 4: "禁止"}
        return (
            f"- {self.name}: {self.description} "
            f"[权限: Level {self.permission_level} {level_desc.get(self.permission_level, '')}]"
            f"{params}"
        )

    def to_json_schema(self) -> dict[str, Any]:
        """导出 OpenAI-compatible 的 JSON Schema。"""
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []

        for parameter in self.parameters:
            schema: dict[str, Any] = {"type": parameter.type}
            if parameter.description:
                schema["description"] = parameter.description
            properties[parameter.name] = schema
            if parameter.required:
                required.append(parameter.name)

        result: dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            result["required"] = required
        return result

    def to_function_schema(self) -> dict[str, Any]:
        """导出工具定义，供原生 function calling 使用。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.to_json_schema(),
            },
        }

    def validate_arguments(self, arguments: dict[str, Any]) -> ToolResult | None:
        """Validate required arguments before permission checks or execution."""
        missing = [
            parameter.name
            for parameter in self.parameters
            if parameter.required and not arguments.get(parameter.name)
        ]
        if not missing:
            return None

        missing_text = "、".join(missing)
        return ToolResult(
            success=False,
            error="missing_params",
            message=f"请提供以下必要信息：{missing_text}。",
        )
