"""
ReAct Agent —— 推理-行动循环

采用 ReAct（Reasoning + Acting）模式：
  Thought → Action → Observation → ... → Final Answer

复用 llm/ 目录下的 LLM 提供商，不另建 LLM 调用。
失败兜底：
  - 单工具超时 3s 自动重试一次
  - 连续 3 次失败发布 TRANSFER_REQUEST 事件并移交上下文
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Optional

from voice_optimized_rag.agent.base_tool import BaseTool, ToolResult
from voice_optimized_rag.agent.permission_guard import PermissionGuard
from voice_optimized_rag.core.conversation_stream import (
    ConversationStream,
    EventType,
    StreamEvent,
)
from voice_optimized_rag.dialogue.session import SessionContext, TaskStatus
from voice_optimized_rag.llm.base import LLMProvider
from voice_optimized_rag.utils.logging import get_logger

logger = get_logger("react_agent")

REACT_SYSTEM_PROMPT = """你是一个智能客服 Agent，通过调用工具帮助用户完成业务操作。

你必须按照以下格式思考和行动：

Thought: 分析用户需求和当前状态
Action: 工具名称
Action Input: {{"参数名": "参数值"}}
Observation: （工具返回结果，由系统填充）
... （可重复多次 Thought/Action/Observation）
Thought: 任务完成，可以给用户最终回复
Final Answer: 对用户的最终回复

可用工具：
{tools}

重要规则：
1. 每次只调用一个工具
2. 参数缺失时向用户追问，不要猜测
3. 如果工具调用失败，尝试换一种方式或告知用户
4. Final Answer 必须用自然、友好的中文回复"""


class ReactAgent:
    """
    ReAct 推理循环 Agent

    执行流程：
    1. 接收用户意图和会话上下文
    2. 构建 ReAct Prompt（含可用工具列表）
    3. 循环执行 Thought → Action → Observation
    4. 达到最终回答或超过最大迭代次数时停止
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: list[BaseTool],
        permission_guard: PermissionGuard,
        stream: ConversationStream,
        max_iterations: int = 10,
        tool_timeout: float = 3.0,
        tool_retry: int = 1,
        max_scratchpad_chars: int = 6000,
    ) -> None:
        self._llm = llm
        self._tools = {tool.name: tool for tool in tools}
        self._permission_guard = permission_guard
        self._stream = stream
        self._max_iterations = max_iterations
        self._tool_timeout = tool_timeout
        self._tool_retry = tool_retry
        self._max_scratchpad_chars = max_scratchpad_chars

    async def execute(
        self,
        user_request: str,
        session: SessionContext,
        memory_context: str = "",
    ) -> str:
        """
        执行 ReAct 推理循环

        Args:
            user_request: 用户请求文本
            session: 会话上下文
            memory_context: 对话记忆上下文

        Returns:
            Agent 最终回复文本
        """
        session.task_status = TaskStatus.IN_PROGRESS

        # 构建工具列表描述
        tools_desc = "\n".join(
            tool.to_prompt_description()
            for tool in self._tools.values()
            if tool.permission_level < 4  # Level 4 不展示给 Agent
        )

        system = REACT_SYSTEM_PROMPT.format(tools=tools_desc)
        scratchpad = f"用户请求: {user_request}\n"
        if memory_context:
            scratchpad += f"\n对话上下文:\n{memory_context}\n"

        consecutive_failures = 0

        for iteration in range(self._max_iterations):
            # 防止 scratchpad 超出 LLM 上下文窗口
            if len(scratchpad) > self._max_scratchpad_chars:
                # 保留开头（用户请求）和最近的推理轨迹
                header_end = scratchpad.find("\n", 200)
                if header_end == -1:
                    header_end = 200
                header = scratchpad[:header_end]
                tail = scratchpad[-(self._max_scratchpad_chars - len(header) - 50):]
                scratchpad = header + "\n...(中间推理已省略)...\n" + tail

            # 让 LLM 生成下一步 Thought + Action
            prompt = f"{system}\n\n{scratchpad}\n"
            try:
                response = await self._llm.generate(prompt)
            except Exception as e:
                logger.error(f"LLM call failed in ReAct loop: {e}")
                consecutive_failures += 1
                session.record_agent_failure()
                if consecutive_failures >= 3:
                    return await self._handle_max_failures(session)
                continue

            scratchpad += response + "\n"

            # 检查是否有 Final Answer
            final = self._extract_final_answer(response)
            if final:
                session.task_status = TaskStatus.COMPLETED
                session.reset_agent_failures()
                return final

            # 提取 Action 和 Action Input
            action_name, action_input = self._parse_action(response)
            if not action_name:
                # 无法解析出 Action，追加提示让 LLM 重试
                scratchpad += "Observation: 请按照格式输出 Action 和 Action Input\n"
                continue

            # 执行工具
            tool = self._tools.get(action_name)
            if not tool:
                scratchpad += f"Observation: 工具 '{action_name}' 不存在，可用工具: {list(self._tools.keys())}\n"
                continue

            # 权限检查
            perm_result = await self._permission_guard.check_permission(
                tool=tool,
                session=session,
                **action_input,
            )
            if not perm_result.success:
                scratchpad += f"Observation: 权限校验失败 - {perm_result.message}\n"
                continue

            # 执行工具（带超时和重试）
            result = await self._execute_tool_with_retry(tool, action_input)
            if result.success:
                consecutive_failures = 0
                scratchpad += f"Observation: {result.message or json.dumps(result.data, ensure_ascii=False)}\n"
            else:
                consecutive_failures += 1
                session.record_agent_failure()
                scratchpad += f"Observation: 工具调用失败 - {result.error}: {result.message}\n"

                if consecutive_failures >= 3:
                    return await self._handle_max_failures(session)

        # 超过最大迭代次数
        session.task_status = TaskStatus.FAILED
        return "抱歉，当前操作比较复杂，我为您转接人工客服来处理。"

    async def _execute_tool_with_retry(
        self,
        tool: BaseTool,
        kwargs: dict[str, Any],
    ) -> ToolResult:
        """带超时和重试的工具执行"""
        for attempt in range(1 + self._tool_retry):
            try:
                result = await asyncio.wait_for(
                    tool.execute(**kwargs),
                    timeout=self._tool_timeout,
                )
                return result
            except asyncio.TimeoutError:
                logger.warning(
                    f"Tool '{tool.name}' timeout (attempt {attempt + 1})"
                )
                if attempt < self._tool_retry:
                    continue
                return ToolResult(
                    success=False,
                    error="timeout",
                    message=f"工具 '{tool.name}' 执行超时",
                )
            except Exception as e:
                logger.error(f"Tool '{tool.name}' error: {e}")
                if attempt < self._tool_retry:
                    continue
                return ToolResult(
                    success=False,
                    error="execution_error",
                    message=str(e),
                )

        return ToolResult(success=False, error="unknown", message="重试次数耗尽")

    async def _handle_max_failures(self, session: SessionContext) -> str:
        """连续失败达到上限，触发转人工"""
        session.task_status = TaskStatus.FAILED
        await self._stream.publish(StreamEvent(
            event_type=EventType.TRANSFER_REQUEST,
            text=f"Agent 连续失败 {session.agent_failure_count} 次",
            metadata=session.to_dict(),
        ))
        return "非常抱歉，系统暂时无法完成此操作，正在为您转接人工客服。"

    @staticmethod
    def _extract_final_answer(text: str) -> Optional[str]:
        """从 LLM 输出中提取 Final Answer"""
        match = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _parse_action(text: str) -> tuple[Optional[str], dict]:
        """
        从 LLM 输出中解析 Action 和 Action Input

        使用平衡括号计数法提取 JSON，正确处理嵌套对象。

        Returns:
            (action_name, action_input_dict) or (None, {})
        """
        action_match = re.search(r"Action:\s*(\S+)", text)
        if not action_match:
            return None, {}

        action_name = action_match.group(1).strip()

        # 使用括号平衡法提取完整 JSON（支持嵌套）
        input_label = re.search(r"Action Input:\s*", text)
        action_input = {}
        if input_label:
            start = input_label.end()
            brace_start = text.find("{", start)
            if brace_start != -1:
                depth = 0
                for i in range(brace_start, len(text)):
                    if text[i] == "{":
                        depth += 1
                    elif text[i] == "}":
                        depth -= 1
                    if depth == 0:
                        json_str = text[brace_start:i + 1]
                        try:
                            action_input = json.loads(json_str)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse Action Input JSON: {json_str}")
                        break

        return action_name, action_input
