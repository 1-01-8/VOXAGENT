"""Native function calling agent with ReAct fallback."""

from __future__ import annotations

import json

from voice_optimized_rag.dialogue.task_slots import (
    format_slot_context,
    merge_arguments_with_session,
    sync_task_slots_from_utterance,
)
from voice_optimized_rag.dialogue.session import SessionContext, TaskStatus
from voice_optimized_rag.llm.base import ToolCallingResponse
from voice_optimized_rag.utils.logging import get_logger
from voice_optimized_rag.agent.react_agent import ReactAgent

logger = get_logger("function_calling_agent")

FUNCTION_CALLING_SYSTEM_PROMPT = """你是{agent_name}，通过调用业务工具帮助用户完成销售、售后、财务相关请求。

可用工具：
{tools}

工作规则：
1. 你只处理销售、售后、财务相关业务内容。
2. 优先使用原生 function calling 调用工具，不要向用户暴露工具名和参数。
3. 如果当前模型不支持原生 function calling，请退回以下 ReAct 格式：
Thought: 分析当前业务目标
Action: 工具名称
Action Input: {{"参数名": "参数值"}}
Observation: 工具返回结果
Final Answer: 给用户的最终回复
4. 参数不足时不要猜测，直接向用户追问缺失的业务信息。
5. 非业务或闲聊请求，礼貌拒绝并引导用户提出销售、售后或财务问题。
6. 如果对话上下文里已经在处理某个业务任务，用户后续补充的订单号、手机号、地址、退款原因等信息应视为同一任务的继续，优先结合已有上下文补全参数，不要重复索要用户已经提供的信息。
7. 最终回复必须使用专业、自然、简洁的中文。"""


def _clean_user_answer(content: str) -> str:
    cleaned = content.strip()
    prefixes = ("Final Answer:", "Final Answer：", "最终答案:", "最终答案：")
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            return cleaned[len(prefix):].strip()
    return cleaned


class FunctionCallingAgent(ReactAgent):
    """Business agent that prefers native function calling."""

    async def execute(
        self,
        user_request: str,
        session: SessionContext,
        memory_context: str = "",
    ) -> str:
        if not self._llm.supports_function_calling:
            return await super().execute(user_request, session, memory_context)

        sync_task_slots_from_utterance(session, user_request)
        session.task_status = TaskStatus.IN_PROGRESS
        tools = [tool for tool in self._tools.values() if tool.permission_level < 4]
        tools_desc = "\n".join(tool.to_prompt_description() for tool in tools)
        tool_schemas = [tool.to_function_schema() for tool in tools]
        system = self._build_system_prompt(tools_desc)

        scratchpad = f"用户请求: {user_request}\n"
        if memory_context:
            scratchpad += f"\n对话上下文:\n{memory_context}\n"
        slot_context = format_slot_context(session)
        if slot_context:
            scratchpad += f"\n已收集业务参数:\n{slot_context}\n"

        consecutive_failures = 0

        for _ in range(self._max_iterations):
            if len(scratchpad) > self._max_scratchpad_chars:
                scratchpad = scratchpad[-self._max_scratchpad_chars :]

            prompt = f"{system}\n\n{scratchpad}\n"
            try:
                response = await self._llm.complete_with_tools(
                    prompt=prompt,
                    tools=tool_schemas,
                )
            except NotImplementedError:
                return await super().execute(user_request, session, memory_context)
            except Exception as error:
                logger.error(f"Function calling failed: {error}")
                consecutive_failures += 1
                session.record_agent_failure()
                if consecutive_failures >= 3:
                    return await self._handle_max_failures(session)
                continue

            if response.tool_calls:
                handled, scratchpad = await self._handle_tool_calls(response, session, scratchpad)
                if handled:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    session.record_agent_failure()
                if consecutive_failures >= 3:
                    return await self._handle_max_failures(session)
                continue

            content = _clean_user_answer(response.content)
            if content:
                session.task_status = TaskStatus.COMPLETED
                session.reset_agent_failures()
                return content

            scratchpad += "Observation: 请直接给出业务结论，或者继续调用工具。\n"

        session.task_status = TaskStatus.FAILED
        return "抱歉，当前业务处理未能在限定步骤内完成，我为您转接人工客服继续处理。"

    async def _handle_tool_calls(
        self,
        response: ToolCallingResponse,
        session: SessionContext,
        scratchpad: str,
    ) -> tuple[bool, str]:
        success = False

        for tool_call in response.tool_calls:
            tool = self._tools.get(tool_call.name)
            if not tool:
                scratchpad += f"Tool Call: {tool_call.name}\nObservation: 工具不存在。\n"
                continue

            resolved_arguments = merge_arguments_with_session(
                session,
                tool_call.arguments,
                [parameter.name for parameter in tool.parameters],
            )

            validation_result = tool.validate_arguments(resolved_arguments)
            if validation_result is not None:
                success = True
                scratchpad += (
                    f"Tool Call: {tool_call.name}({json.dumps(resolved_arguments, ensure_ascii=False)})\n"
                    f"Observation: 参数不足 - {validation_result.message}\n"
                )
                continue

            perm_result = await self._permission_guard.check_permission(
                tool=tool,
                session=session,
                **resolved_arguments,
            )
            if not perm_result.success:
                scratchpad += (
                    f"Tool Call: {tool_call.name}({json.dumps(resolved_arguments, ensure_ascii=False)})\n"
                    f"Observation: 权限校验失败 - {perm_result.message}\n"
                )
                continue

            result = await self._execute_tool_with_retry(tool, resolved_arguments)
            if result.success:
                success = True
                observation = result.message or json.dumps(result.data, ensure_ascii=False)
                scratchpad += (
                    f"Tool Call: {tool_call.name}({json.dumps(resolved_arguments, ensure_ascii=False)})\n"
                    f"Observation: {observation}\n"
                )
            else:
                if result.error == "missing_params":
                    success = True
                else:
                    session.record_agent_failure()
                scratchpad += (
                    f"Tool Call: {tool_call.name}({json.dumps(resolved_arguments, ensure_ascii=False)})\n"
                    f"Observation: 工具调用失败 - {result.error}: {result.message}\n"
                )

        return success, scratchpad